#!/usr/bin/env bash
#
# benchmark.sh — start the local serving stack as needed, then run qwen-benchmark.
#
# Usage:
#   ./benchmark.sh
#   ./benchmark.sh --target litellm --runs 5 --max-tokens 512
#   ./benchmark.sh --restart --target both --prompt-file ./prompt.txt
#   ./benchmark.sh --stop-after --target vllm
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_LITELLM_URL="http://127.0.0.1:11111/v1"
DEFAULT_VLLM_URL="http://127.0.0.1:11112/v1"

TARGET="both"
LITELLM_URL="$DEFAULT_LITELLM_URL"
VLLM_URL="$DEFAULT_VLLM_URL"
STARTUP_TIMEOUT=600
POLL_INTERVAL=2
RESTART=0
STOP_AFTER=0
LAST_STARTED_PID=""

BENCHMARK_ARGS=()
STARTED_PIDS=()
STARTED_LABELS=()
STARTED_PORTS=()

usage() {
    cat <<'EOF'
Start the local LiteLLM/vLLM stack as needed, wait for readiness, then run qwen-benchmark.

Options:
  --target {litellm|vllm|both}   Which path to benchmark. Default: both
  --litellm-url URL              LiteLLM base URL. Default: http://127.0.0.1:11111/v1
  --vllm-url URL                 vLLM base URL. Default: http://127.0.0.1:11112/v1
  --restart                      Kill the managed local service(s) first, then start fresh
  --stop-after                   Stop the service(s) this script started after the benchmark
  --startup-timeout SECONDS      How long to wait for health checks. Default: 600
  --poll-interval SECONDS        Health-check polling interval. Default: 2
  -h, --help                     Show this help text

All other arguments are passed through to qwen-benchmark.

Examples:
  ./benchmark.sh
  ./benchmark.sh --target litellm --runs 5 --max-tokens 512
  ./benchmark.sh --restart --target both --prompt-file ./prompt.txt
  ./benchmark.sh --stop-after --target vllm --runs 1 --warmup-runs 0
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --litellm-url)
            LITELLM_URL="$2"
            shift 2
            ;;
        --vllm-url)
            VLLM_URL="$2"
            shift 2
            ;;
        --restart)
            RESTART=1
            shift
            ;;
        --stop-after)
            STOP_AFTER=1
            shift
            ;;
        --startup-timeout)
            STARTUP_TIMEOUT="$2"
            shift 2
            ;;
        --poll-interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            BENCHMARK_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$TARGET" in
    litellm|vllm|both) ;;
    *)
        echo "❌ Invalid --target: $TARGET" >&2
        exit 1
        ;;
esac

if ! [[ "$STARTUP_TIMEOUT" =~ ^[0-9]+$ ]] || [[ "$STARTUP_TIMEOUT" -lt 1 ]]; then
    echo "❌ --startup-timeout must be a positive integer" >&2
    exit 1
fi

if ! [[ "$POLL_INTERVAL" =~ ^[0-9]+$ ]] || [[ "$POLL_INTERVAL" -lt 1 ]]; then
    echo "❌ --poll-interval must be a positive integer" >&2
    exit 1
fi

if [[ -x "$SCRIPT_DIR/venv/bin/python3" ]]; then
    VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
else
    VENV_PYTHON="python3"
fi

RUN_SERVER="$($VENV_PYTHON -c "import shutil; print(shutil.which('qwen-server') or '')" 2>/dev/null)"
RUN_PROXY="$($VENV_PYTHON -c "import shutil; print(shutil.which('qwen-compress') or '')" 2>/dev/null)"
RUN_BENCH="$($VENV_PYTHON -c "import shutil; print(shutil.which('qwen-benchmark') or '')" 2>/dev/null)"

RUN_SERVER="${RUN_SERVER:-$SCRIPT_DIR/qwen3_6_server.py}"
RUN_PROXY="${RUN_PROXY:-$SCRIPT_DIR/server_compress.py}"

mkdir -p logs

health_url() {
    local base_url="$1"
    base_url="${base_url%/}"
    if [[ "$base_url" == */v1 ]]; then
        printf '%s/health\n' "${base_url%/v1}"
    else
        printf '%s/health\n' "$base_url"
    fi
}

port_pid() {
    local port="$1"
    local pid=""

    if command -v ss >/dev/null 2>&1; then
        pid=$(ss -ltnp "( sport = :$port )" 2>/dev/null \
            | grep -o 'pid=[0-9]\+' \
            | head -1 \
            | cut -d= -f2 || true)
    fi

    if [[ -z "$pid" ]] && command -v lsof >/dev/null 2>&1; then
        pid=$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null | head -1 || true)
    fi

    printf '%s' "$pid"
}

stop_pid_tree() {
    local pid="$1"
    local children=()
    local child=""

    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    mapfile -t children < <(ps -o pid= --ppid "$pid" 2>/dev/null | awk '{print $1}')
    for child in "${children[@]}"; do
        stop_pid_tree "$child"
    done

    kill "$pid" 2>/dev/null || true
    for _ in {1..10}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done

    kill -9 "$pid" 2>/dev/null || true
}

stop_port_owner() {
    local port="$1"
    local label="$2"
    local pid=""

    pid="$(port_pid "$port")"
    if [[ -n "$pid" ]]; then
        echo "🛑 Stopping $label on port $port (PID $pid)..."
        stop_pid_tree "$pid"
    fi
}

http_ready() {
    local url="$1"
    curl -fsS --max-time 5 "$url" >/dev/null 2>&1
}

wait_for_health() {
    local url="$1"
    local label="$2"
    local pid="${3:-}"
    local log_file="${4:-}"
    local deadline=$((SECONDS + STARTUP_TIMEOUT))

    until http_ready "$url"; do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "❌ $label exited before becoming healthy." >&2
            if [[ -n "$log_file" && -f "$log_file" ]]; then
                echo "--- $label log tail ---" >&2
                tail -n 60 "$log_file" >&2 || true
            fi
            exit 1
        fi

        if (( SECONDS >= deadline )); then
            echo "❌ Timed out waiting for $label at $url" >&2
            if [[ -n "$log_file" && -f "$log_file" ]]; then
                echo "--- $label log tail ---" >&2
                tail -n 60 "$log_file" >&2 || true
            fi
            exit 1
        fi
        sleep "$POLL_INTERVAL"
    done
}

new_token_session() {
    "$VENV_PYTHON" - <<'PY' >/dev/null 2>&1
import qwen_token_tracker
qwen_token_tracker.new_session()
PY
}

start_command() {
    local label="$1"
    local log_file="$2"
    shift 2

    echo "🚀 Starting $label..." >&2
    "$@" >"$log_file" 2>&1 &
    local pid="$!"
    STARTED_PIDS+=("$pid")
    STARTED_LABELS+=("$label")
    LAST_STARTED_PID="$pid"
}

is_local_managed_litellm() {
    [[ "$LITELLM_URL" == "http://127.0.0.1:11111/v1" || "$LITELLM_URL" == "http://localhost:11111/v1" ]]
}

is_local_managed_vllm() {
    [[ "$VLLM_URL" == "http://127.0.0.1:11112/v1" || "$VLLM_URL" == "http://localhost:11112/v1" ]]
}

cleanup_started() {
    local idx=""
    local port=""
    local pid=""
    local label=""

    if [[ "$STOP_AFTER" != "1" ]]; then
        return 0
    fi

    for idx in "${!STARTED_PIDS[@]}"; do
        pid="${STARTED_PIDS[$idx]}"
        label="${STARTED_LABELS[$idx]}"
        if kill -0 "$pid" 2>/dev/null; then
            echo "🛑 Stopping $label (PID $pid)..."
            stop_pid_tree "$pid"
        fi
    done

    for port in "${STARTED_PORTS[@]}"; do
        stop_port_owner "$port" "service"
    done
}

trap cleanup_started EXIT

ensure_backend() {
    local url
    local log_file
    local pid=""

    url="$(health_url "$VLLM_URL")"
    if http_ready "$url"; then
        echo "✅ Reusing running vLLM backend at $url"
        return 0
    fi

    if ! is_local_managed_vllm; then
        echo "❌ vLLM is not healthy at $url and this script only auto-starts the local default vLLM endpoint." >&2
        exit 1
    fi

    log_file="$SCRIPT_DIR/logs/benchmark_backend.log"
    if [[ -x "$RUN_SERVER" && "$RUN_SERVER" != *.py ]]; then
        start_command "vLLM backend" "$log_file" "$RUN_SERVER"
    else
        start_command "vLLM backend" "$log_file" "$VENV_PYTHON" "$RUN_SERVER"
    fi
    pid="$LAST_STARTED_PID"
    STARTED_PORTS+=("11112")
    wait_for_health "$url" "vLLM backend" "$pid" "$log_file"
    echo "✅ vLLM backend ready at $url"
}

ensure_proxy() {
    local url
    local log_file
    local pid=""

    url="$(health_url "$LITELLM_URL")"
    if http_ready "$url"; then
        echo "✅ Reusing running LiteLLM proxy at $url"
        return 0
    fi

    if ! is_local_managed_litellm; then
        echo "❌ LiteLLM is not healthy at $url and this script only auto-starts the local default LiteLLM endpoint." >&2
        exit 1
    fi

    new_token_session
    log_file="$SCRIPT_DIR/logs/benchmark_proxy.log"
    if [[ -x "$RUN_PROXY" && "$RUN_PROXY" != *.py ]]; then
        start_command "LiteLLM proxy" "$log_file" "$RUN_PROXY"
    else
        start_command "LiteLLM proxy" "$log_file" "$VENV_PYTHON" "$RUN_PROXY"
    fi
    pid="$LAST_STARTED_PID"
    STARTED_PORTS+=("11111")
    wait_for_health "$url" "LiteLLM proxy" "$pid" "$log_file"
    echo "✅ LiteLLM proxy ready at $url"
}

if [[ "$RESTART" == "1" ]]; then
    if [[ "$TARGET" == "both" || "$TARGET" == "litellm" ]]; then
        if is_local_managed_litellm; then
            stop_port_owner "11111" "LiteLLM proxy"
        fi
    fi
    if [[ "$TARGET" == "vllm" || "$TARGET" == "both" || "$TARGET" == "litellm" ]]; then
        if is_local_managed_vllm; then
            stop_port_owner "11112" "vLLM backend"
        fi
    fi
fi

ensure_backend
if [[ "$TARGET" == "both" || "$TARGET" == "litellm" ]]; then
    ensure_proxy
fi

echo "📏 Running qwen-benchmark for target=$TARGET"
if [[ -n "$RUN_BENCH" ]]; then
    "$RUN_BENCH" \
        --target "$TARGET" \
        --litellm-url "$LITELLM_URL" \
        --vllm-url "$VLLM_URL" \
        "${BENCHMARK_ARGS[@]}"
else
    "$VENV_PYTHON" "$SCRIPT_DIR/qwen_benchmark.py" \
        --target "$TARGET" \
        --litellm-url "$LITELLM_URL" \
        --vllm-url "$VLLM_URL" \
        "${BENCHMARK_ARGS[@]}"
fi

if [[ "$STOP_AFTER" == "1" ]]; then
    echo "✅ Benchmark complete. Stopping services started by this script."
else
    echo "✅ Benchmark complete. Any reused or newly started services are still running."
fi
