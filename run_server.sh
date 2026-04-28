#!/usr/bin/env bash
#
# Full local Qwen3.6 serving stack starter.
#
# Starts:
#   11114 - vLLM MLX backend
#   11115 - LiteLLM proxy / compression server
#   11116 - token stats / logging server
#
# This script intentionally DOES NOT run qwen3_6_server.py.
# It uses the vllm-mlx path, because that is the path that works on your setup.
#
# Usage:
#   ./run_server.sh
#   ./run_server.sh start
#   ./run_server.sh stop
#   ./run_server.sh restart
#   ./run_server.sh status
#   ./run_server.sh logs
#   ./run_server.sh backend
#   ./run_server.sh proxy
#   ./run_server.sh stats
#
# Optional:
#   RUN_SERVER_NO_TAIL=1 ./run_server.sh start

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Model & ports ────────────────────────────────────────────────────────────

MODEL="${MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit}"

BACKEND_PORT="${BACKEND_PORT:-11114}"
PROXY_PORT="${PROXY_PORT:-11115}"
STATS_PORT="${STATS_PORT:-11116}"

HOST="${HOST:-0.0.0.0}"

# These are the settings from your working mlx-start.sh.
MAX_TOKENS="${MAX_TOKENS:-30000}"
CACHE_MEMORY_PERCENT="${CACHE_MEMORY_PERCENT:-20}"

# ── Paths ────────────────────────────────────────────────────────────────────

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

BACKEND_LOG="$LOG_DIR/vllm-mlx.log"
PROXY_LOG="$LOG_DIR/proxy.log"
STATS_LOG="$LOG_DIR/stats.log"

touch "$BACKEND_LOG" "$PROXY_LOG" "$STATS_LOG"

BACKEND_PID_FILE="$SCRIPT_DIR/.mlx-backend.pid"
PROXY_PID_FILE="$SCRIPT_DIR/.mlx-proxy.pid"
STATS_PID_FILE="$SCRIPT_DIR/.mlx-stats.pid"

# ── Binary discovery ─────────────────────────────────────────────────────────

find_vllm_mlx() {
    if [ -n "${VLLM_MLX_BIN:-}" ] && [ -x "$VLLM_MLX_BIN" ]; then
        echo "$VLLM_MLX_BIN"
        return 0
    fi

    if [ -x "$SCRIPT_DIR/venv/bin/vllm-mlx" ]; then
        echo "$SCRIPT_DIR/venv/bin/vllm-mlx"
        return 0
    fi

    if [ -x "/Users/sna/.venv-vllm-metal/bin/vllm-mlx" ]; then
        echo "/Users/sna/.venv-vllm-metal/bin/vllm-mlx"
        return 0
    fi

    local found
    found="$(command -v vllm-mlx || true)"
    if [ -n "$found" ] && [ -x "$found" ]; then
        echo "$found"
        return 0
    fi

    return 1
}

find_python() {
    if [ -n "${VENV_PYTHON:-}" ] && [ -x "$VENV_PYTHON" ]; then
        echo "$VENV_PYTHON"
        return 0
    fi

    if [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
        echo "$SCRIPT_DIR/venv/bin/python"
        return 0
    fi

    if [ -x "$SCRIPT_DIR/venv/bin/python3" ]; then
        echo "$SCRIPT_DIR/venv/bin/python3"
        return 0
    fi

    if [ -x "/Users/sna/.venv-vllm-metal/bin/python3" ]; then
        echo "/Users/sna/.venv-vllm-metal/bin/python3"
        return 0
    fi

    local found
    found="$(command -v python3 || true)"
    if [ -n "$found" ] && [ -x "$found" ]; then
        echo "$found"
        return 0
    fi

    return 1
}

VLLM_MLX_BIN="$(find_vllm_mlx || true)"
VENV_PYTHON="$(find_python || true)"

require_vllm_mlx() {
    if [ -z "$VLLM_MLX_BIN" ]; then
        echo "ERROR: Could not find vllm-mlx."
        echo ""
        echo "Checked:"
        echo "  $SCRIPT_DIR/venv/bin/vllm-mlx"
        echo "  /Users/sna/.venv-vllm-metal/bin/vllm-mlx"
        echo "  PATH"
        echo ""
        exit 1
    fi
}

require_python() {
    if [ -z "$VENV_PYTHON" ]; then
        echo "ERROR: Could not find Python for proxy/stats servers."
        echo ""
        echo "Checked:"
        echo "  $SCRIPT_DIR/venv/bin/python"
        echo "  $SCRIPT_DIR/venv/bin/python3"
        echo "  /Users/sna/.venv-vllm-metal/bin/python3"
        echo "  PATH"
        echo ""
        exit 1
    fi
}

require_file() {
    local path="$1"

    if [ ! -f "$path" ]; then
        echo "ERROR: Required file not found:"
        echo "  $path"
        exit 1
    fi
}

# ── Helpers ──────────────────────────────────────────────────────────────────

prompt_hf_token() {
    if [ -n "${HF_TOKEN:-}" ]; then
        export HF_TOKEN
        return 0
    fi

    # Do not block non-interactive runs.
    if [ ! -t 0 ]; then
        return 0
    fi

    echo "Enter your HuggingFace token if needed."
    echo "Leave empty if the model is already cached or does not require auth."
    printf "HF_TOKEN: "

    stty -echo
    IFS= read -r HF_TOKEN || true
    stty echo
    echo ""

    if [ -n "$HF_TOKEN" ]; then
        export HF_TOKEN
    fi
}

port_pids() {
    local port="$1"
    lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true
}

is_backend_ready() {
    curl -sf "http://localhost:$BACKEND_PORT/health" >/dev/null 2>&1
}

is_proxy_ready() {
    curl -sf "http://localhost:$PROXY_PORT/health" >/dev/null 2>&1
}

is_stats_ready() {
    curl -sf "http://localhost:$STATS_PORT/" >/dev/null 2>&1
}

wait_for_url() {
    local name="$1"
    local url="$2"
    local log_file="$3"
    local timeout_seconds="$4"

    echo -n "Waiting for $name"

    local i
    for i in $(seq 1 "$timeout_seconds"); do
        if curl -sf "$url" >/dev/null 2>&1; then
            echo " OK"
            return 0
        fi

        sleep 1
        echo -n "."
    done

    echo ""
    echo "ERROR: $name did not become ready within ${timeout_seconds}s."
    echo ""
    echo "Last log lines from $log_file:"
    echo "----------------------------------------------------------------"
    tail -n 80 "$log_file" 2>/dev/null || true
    echo "----------------------------------------------------------------"
    return 1
}

cleanup_pid_file() {
    local pid_file="$1"
    local name="$2"

    if [ ! -f "$pid_file" ]; then
        return 0
    fi

    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"

    if [ -z "$pid" ]; then
        rm -f "$pid_file"
        return 0
    fi

    if kill -0 "$pid" 2>/dev/null; then
        echo "Stopping $name PID $pid..."
        kill "$pid" 2>/dev/null || true

        local i
        for i in $(seq 1 10); do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing $name PID $pid..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi

    rm -f "$pid_file"
}

kill_port_listeners() {
    local port="$1"
    local name="$2"

    local pids
    pids="$(port_pids "$port")"

    if [ -z "$pids" ]; then
        return 0
    fi

    echo "Stopping remaining $name listener(s) on port $port: $pids"

    local pid
    for pid in $pids; do
        kill "$pid" 2>/dev/null || true
    done

    sleep 1

    pids="$(port_pids "$port")"
    if [ -n "$pids" ]; then
        echo "Force killing remaining $name listener(s) on port $port: $pids"
        for pid in $pids; do
            kill -9 "$pid" 2>/dev/null || true
        done
    fi
}

# ── Start services ───────────────────────────────────────────────────────────

start_backend() {
    require_vllm_mlx
    prompt_hf_token

    if is_backend_ready; then
        echo "Backend already running on port $BACKEND_PORT."
        return 0
    fi

    local existing
    existing="$(port_pids "$BACKEND_PORT")"
    if [ -n "$existing" ]; then
        echo "ERROR: Port $BACKEND_PORT is already in use by PID(s): $existing"
        echo "Run:"
        echo "  ./run_server.sh stop"
        exit 1
    fi

    echo "Starting vLLM MLX backend..."
    echo "  Model:                $MODEL"
    echo "  Port:                 $BACKEND_PORT"
    echo "  Binary:               $VLLM_MLX_BIN"
    echo "  Max tokens:           $MAX_TOKENS"
    echo "  Cache memory percent: $CACHE_MEMORY_PERCENT"
    echo "  Log:                  $BACKEND_LOG"

    nohup "$VLLM_MLX_BIN" serve "$MODEL" \
        --port "$BACKEND_PORT" \
        --host "$HOST" \
        --trust-remote-code \
        --max-tokens "$MAX_TOKENS" \
        --max-request-tokens "$MAX_TOKENS" \
        --cache-memory-percent "$CACHE_MEMORY_PERCENT" \
        --kv-cache-quantization \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --enable-auto-tool-choice \
        --default-temperature 0.7 \
        --default-top-p 0.8 \
        --stream-interval 8 \
        >> "$BACKEND_LOG" 2>&1 &

    local pid=$!
    echo "$pid" > "$BACKEND_PID_FILE"

    echo "  Backend PID: $pid"

    wait_for_url \
        "backend" \
        "http://localhost:$BACKEND_PORT/health" \
        "$BACKEND_LOG" \
        "${BACKEND_START_TIMEOUT:-240}"
}

start_stats() {
    require_python
    require_file "$SCRIPT_DIR/qwen_token_stats_server.py"

    if is_stats_ready; then
        echo "Stats/logging server already running on port $STATS_PORT."
        return 0
    fi

    local existing
    existing="$(port_pids "$STATS_PORT")"
    if [ -n "$existing" ]; then
        echo "ERROR: Port $STATS_PORT is already in use by PID(s): $existing"
        echo "Run:"
        echo "  ./run_server.sh stop"
        exit 1
    fi

    echo "Starting token stats/logging server..."
    echo "  Port:   $STATS_PORT"
    echo "  Python: $VENV_PYTHON"
    echo "  Log:    $STATS_LOG"

    nohup "$VENV_PYTHON" "$SCRIPT_DIR/qwen_token_stats_server.py" \
        >> "$STATS_LOG" 2>&1 &

    local pid=$!
    echo "$pid" > "$STATS_PID_FILE"

    echo "  Stats PID: $pid"

    wait_for_url \
        "stats/logging server" \
        "http://localhost:$STATS_PORT/" \
        "$STATS_LOG" \
        "${STATS_START_TIMEOUT:-60}"
}

start_proxy() {
    require_python
    require_file "$SCRIPT_DIR/server_compress.py"

    if is_proxy_ready; then
        echo "LiteLLM proxy already running on port $PROXY_PORT."
        return 0
    fi

    local existing
    existing="$(port_pids "$PROXY_PORT")"
    if [ -n "$existing" ]; then
        echo "ERROR: Port $PROXY_PORT is already in use by PID(s): $existing"
        echo "Run:"
        echo "  ./run_server.sh stop"
        exit 1
    fi

    echo "Starting LiteLLM proxy / compression server..."
    echo "  Port:   $PROXY_PORT"
    echo "  Python: $VENV_PYTHON"
    echo "  Log:    $PROXY_LOG"

    nohup "$VENV_PYTHON" "$SCRIPT_DIR/server_compress.py" \
        >> "$PROXY_LOG" 2>&1 &

    local pid=$!
    echo "$pid" > "$PROXY_PID_FILE"

    echo "  Proxy PID: $pid"

    wait_for_url \
        "LiteLLM proxy" \
        "http://localhost:$PROXY_PORT/health" \
        "$PROXY_LOG" \
        "${PROXY_START_TIMEOUT:-60}"
}

# ── Stop/status/logs ─────────────────────────────────────────────────────────

stop_all() {
    echo "Stopping Qwen3.6 local serving stack..."

    cleanup_pid_file "$PROXY_PID_FILE" "LiteLLM proxy"
    cleanup_pid_file "$STATS_PID_FILE" "stats/logging server"
    cleanup_pid_file "$BACKEND_PID_FILE" "vLLM MLX backend"

    kill_port_listeners "$PROXY_PORT" "LiteLLM proxy"
    kill_port_listeners "$STATS_PORT" "stats/logging server"
    kill_port_listeners "$BACKEND_PORT" "vLLM MLX backend"

    echo "All services stopped."
}

show_status() {
    echo ""
    echo "Qwen3.6 local serving stack status"
    echo "----------------------------------"

    echo ""
    echo "Backend: http://localhost:$BACKEND_PORT"
    if is_backend_ready; then
        echo "  OK - running"
        echo "  PID file: $(cat "$BACKEND_PID_FILE" 2>/dev/null || echo "missing")"
    else
        echo "  NOT RUNNING"
    fi

    echo ""
    echo "LiteLLM proxy: http://localhost:$PROXY_PORT"
    if is_proxy_ready; then
        echo "  OK - running"
        echo "  PID file: $(cat "$PROXY_PID_FILE" 2>/dev/null || echo "missing")"
    else
        echo "  NOT RUNNING"
    fi

    echo ""
    echo "Stats/logging: http://localhost:$STATS_PORT"
    if is_stats_ready; then
        echo "  OK - running"
        echo "  PID file: $(cat "$STATS_PID_FILE" 2>/dev/null || echo "missing")"
    else
        echo "  NOT RUNNING"
    fi

    echo ""
}

show_logs() {
    echo ""
    echo "Backend log: $BACKEND_LOG"
    echo "----------------------------------------------------------------"
    tail -n 80 "$BACKEND_LOG" 2>/dev/null || true

    echo ""
    echo "Proxy log: $PROXY_LOG"
    echo "----------------------------------------------------------------"
    tail -n 80 "$PROXY_LOG" 2>/dev/null || true

    echo ""
    echo "Stats/logging log: $STATS_LOG"
    echo "----------------------------------------------------------------"
    tail -n 80 "$STATS_LOG" 2>/dev/null || true
    echo ""
}

tail_live_logs() {
    if [ "${RUN_SERVER_NO_TAIL:-0}" = "1" ]; then
        return 0
    fi

    echo ""
    echo "Live logs. Press Ctrl+C to stop tailing."
    echo "Servers will keep running in the background."
    echo ""
    echo "Backend: http://localhost:$BACKEND_PORT"
    echo "Proxy:   http://localhost:$PROXY_PORT"
    echo "Stats:   http://localhost:$STATS_PORT"
    echo ""

    tail -n 40 -F "$BACKEND_LOG" "$PROXY_LOG" "$STATS_LOG"
}

start_all() {
    stop_all 2>/dev/null || true

    echo ""
    echo "Starting Qwen3.6 local serving stack..."
    echo ""

    start_backend
    echo ""

    start_stats
    echo ""

    start_proxy
    echo ""

    show_status
    tail_live_logs
}

usage() {
    echo "Usage: $0 [start|backend|proxy|stats|stop|restart|status|logs]"
    echo ""
    echo "Commands:"
    echo "  start     Start backend + stats/logging + proxy. Default."
    echo "  backend   Start only vLLM MLX backend on $BACKEND_PORT."
    echo "  stats     Start only stats/logging server on $STATS_PORT."
    echo "  proxy     Start only LiteLLM proxy on $PROXY_PORT."
    echo "  stop      Stop all services."
    echo "  restart   Stop all services, then start all."
    echo "  status    Show service status."
    echo "  logs      Show recent logs."
    echo ""
    echo "Environment overrides:"
    echo "  MODEL=$MODEL"
    echo "  BACKEND_PORT=$BACKEND_PORT"
    echo "  PROXY_PORT=$PROXY_PORT"
    echo "  STATS_PORT=$STATS_PORT"
    echo "  MAX_TOKENS=$MAX_TOKENS"
    echo "  CACHE_MEMORY_PERCENT=$CACHE_MEMORY_PERCENT"
    echo "  RUN_SERVER_NO_TAIL=1"
}

# ── Main ─────────────────────────────────────────────────────────────────────

case "${1:-start}" in
    start|both)
        start_all
        ;;

    backend)
        start_backend
        ;;

    stats)
        start_stats
        ;;

    proxy)
        start_proxy
        ;;

    stop)
        stop_all
        ;;

    restart)
        stop_all 2>/dev/null || true
        sleep 2
        start_all
        ;;

    status)
        show_status
        ;;

    logs)
        show_logs
        ;;

    help|-h|--help)
        usage
        ;;

    *)
        usage
        exit 1
        ;;
esac