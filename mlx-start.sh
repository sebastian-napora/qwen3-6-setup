#!/bin/bash
#
# Start Qwen3.6-35B-A3B-4bit MLX serving stack (Apple Silicon)
#
# Usage:
#   ./mlx-start.sh          # start all (backend + stats + proxy)
#   ./mlx-start.sh daemon   # start all without live log tailing
#   ./mlx-start.sh backend   # start only vLLM MLX backend (11114)
#   ./mlx-start.sh proxy     # start only LiteLLM proxy (11115)
#   ./mlx-start.sh stats     # start only token stats server (11116)
#   ./mlx-start.sh stop      # stop all services
#
# Architecture:
#   Copilot -> LiteLLM (11115) -> vLLM MLX (11114)
#                              -> Token stats (11116)
#
# Ports:
#   11114 - vLLM MLX backend (OpenAI-compatible API)
#   11115 - LiteLLM proxy (with compression callbacks)
#   11116 - Token stats server (per-request usage tracking)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE="${MLX_ENV_FILE:-$SCRIPT_DIR/.mlx.env}"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

# ── Model & System Config ────────────────────────────────────────────────────
MODEL="${MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit}"
BACKEND_PORT="${BACKEND_PORT:-11114}"
PROXY_PORT="${PROXY_PORT:-11115}"
STATS_PORT="${STATS_PORT:-11116}"
HOST="${HOST:-0.0.0.0}"

# MLX-specific settings
CACHE_MEMORY_PERCENT="${CACHE_MEMORY_PERCENT:-20}"       # % of unified memory for KV cache
MAX_TOKENS="${MAX_TOKENS:-30000}"
BACKEND_START_TIMEOUT="${BACKEND_START_TIMEOUT:-3600}"
STATS_START_TIMEOUT="${STATS_START_TIMEOUT:-10}"
PROXY_START_TIMEOUT="${PROXY_START_TIMEOUT:-10}"
ENABLE_REASONING_PARSER="${ENABLE_REASONING_PARSER:-no}"

# ── Paths ─────────────────────────────────────────────────────────────────────
VENV_PYTHON="${VENV_PYTHON:-$SCRIPT_DIR/venv/bin/python}"
VLLM_MLX_BIN="${VLLM_MLX_BIN:-vllm-mlx}"
RUN_PROXY="${RUN_PROXY:-$SCRIPT_DIR/src/server_compress.py}"
RUN_STATS="${RUN_STATS:-$SCRIPT_DIR/src/qwen_token_stats_server.py}"
RUN_PROXY_MODULE="${RUN_PROXY_MODULE:-src.server_compress}"
RUN_STATS_MODULE="${RUN_STATS_MODULE:-src.qwen_token_stats_server}"
CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-$SCRIPT_DIR/lite_llm_config.yaml}"
QWEN_LOG_DIR="${QWEN_LOG_DIR:-$SCRIPT_DIR/logs}"
PID_PREFIX="${PID_PREFIX:-.mlx}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CONFIG_FILE_PATH
export QWEN_LOG_DIR
export QWEN_COPILOT_MIN_OUTPUT_TOKENS="${QWEN_COPILOT_MIN_OUTPUT_TOKENS:-20000}"
export QWEN_STATS_PORT="$STATS_PORT"
export LITE_LLM_PROXY_PORT="$PROXY_PORT"

mkdir -p "$QWEN_LOG_DIR"
BACKEND_LOG="$QWEN_LOG_DIR/vllm-mlx.log"
PROXY_LOG="$QWEN_LOG_DIR/proxy.log"
STATS_LOG="$QWEN_LOG_DIR/stats.log"

# ── PID Helpers ───────────────────────────────────────────────────────────────
PID_FILE="$SCRIPT_DIR/$PID_PREFIX-pids"
BACKEND_PID_FILE="$SCRIPT_DIR/$PID_PREFIX-backend.pid"
STATS_PID_FILE="$SCRIPT_DIR/$PID_PREFIX-stats.pid"
PROXY_PID_FILE="$SCRIPT_DIR/$PID_PREFIX-proxy.pid"

port_listening() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
    else
        curl -sf --max-time 2 "http://localhost:$port/" >/dev/null 2>&1
    fi
}

start_detached() {
    local log_file="$1"
    shift

    # Keep services alive if the launcher terminal receives Ctrl+C while
    # tailing logs. TERM is intentionally not ignored so `stop` still works.
    nohup bash -c 'trap "" HUP INT; exec "$@"' bash "$@" \
        >> "$log_file" 2>&1 < /dev/null &
}

proxy_ready() {
    port_listening "$PROXY_PORT"
}

cleanup_pid() {
    local pidfile="$1"
    if [ -f "$pidfile" ]; then
        local pid
        pid=$(cat "$pidfile" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && echo "   Stopped PID $pid" || echo "   PID $pid not running"
        fi
        rm -f "$pidfile"
    fi
}

# ── Backend ─────────────────────────────────────────────────────────────────
start_backend() {
    # Guard: skip if backend is already responding on the port
    if port_listening "$BACKEND_PORT"; then
        echo "   ✅ Backend already running on port $BACKEND_PORT"
        return
    fi

    echo "🚀 Starting vLLM MLX backend on port $BACKEND_PORT..."
    echo "   Model: $MODEL"

    local reasoning_args=()
    if [ "$ENABLE_REASONING_PARSER" = "yes" ]; then
        reasoning_args=(--reasoning-parser qwen3)
    fi

    start_detached "$BACKEND_LOG" "$VLLM_MLX_BIN" serve "$MODEL" \
        --port "$BACKEND_PORT" \
        --host "$HOST" \
        --trust-remote-code \
        --max-tokens "$MAX_TOKENS" \
        --max-request-tokens "$MAX_TOKENS" \
        --cache-memory-percent "$CACHE_MEMORY_PERCENT" \
        --kv-cache-quantization \
        "${reasoning_args[@]}" \
        --tool-call-parser qwen3_coder \
        --enable-auto-tool-choice \
        --default-temperature 0.7 \
        --default-top-p 0.8 \
        --stream-interval 8

    local pid=$!
    echo $pid > "$BACKEND_PID_FILE"
    echo "   Backend PID: $pid"
    echo "   Log: $BACKEND_LOG"

    # Wait for server to start writing to log before confirming
    echo -n "   Waiting for server to come up"
    for i in $(seq 1 "$BACKEND_START_TIMEOUT"); do
        sleep 1
        echo -n "."
        if ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            echo "   ⚠️  Backend process exited — check $BACKEND_LOG"
            rm -f "$BACKEND_PID_FILE"
            tail -n 20 "$BACKEND_LOG" 2>/dev/null | tail -10
            return 1
        fi
        if port_listening "$BACKEND_PORT"; then
            echo ""
            echo "   ✅ Backend up on port $BACKEND_PORT"
            tail -n 10 "$BACKEND_LOG" 2>/dev/null | grep -E "INFO|WARNING|ERROR|loaded|Uvicorn" | tail -5
            return 0
        fi
    done
    echo ""
    echo "   ⚠️  Backend did not respond within ${BACKEND_START_TIMEOUT}s — check $BACKEND_LOG"
    tail -n 20 "$BACKEND_LOG" 2>/dev/null | tail -10
    return 1
}

# ── Stats Server ─────────────────────────────────────────────────────────────
start_stats() {
    if curl -sf "http://localhost:$STATS_PORT/" > /dev/null 2>&1; then
        echo "   ✅ Stats server already running on port $STATS_PORT"
        return
    fi

    echo "🚀 Starting token stats server on port $STATS_PORT..."
    if [ -n "$RUN_STATS_MODULE" ]; then
        start_detached "$STATS_LOG" env PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            "$VENV_PYTHON" -m "$RUN_STATS_MODULE"
    else
        start_detached "$STATS_LOG" env PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            "$VENV_PYTHON" "$RUN_STATS"
    fi

    local pid=$!
    echo $pid > "$STATS_PID_FILE"
    echo "   Stats PID: $pid"

    # Wait for stats server to come up
    echo -n "   Waiting for stats server"
    for i in $(seq 1 "$STATS_START_TIMEOUT"); do
        sleep 1
        echo -n "."
        if ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            echo "   ⚠️  Stats server process exited — check $STATS_LOG"
            rm -f "$STATS_PID_FILE"
            tail -n 20 "$STATS_LOG" 2>/dev/null | tail -10
            return 1
        fi
        if port_listening "$STATS_PORT"; then
            echo " ✅"
            return 0
        fi
    done
    echo ""
    echo "   ⚠️  Stats server did not respond — check $STATS_LOG"
    return 1
}

# ── LiteLLM Proxy ────────────────────────────────────────────────────────────
start_proxy() {
    if proxy_ready; then
        echo "   ✅ Proxy already running on port $PROXY_PORT"
        return
    fi

    echo "🚀 Starting LiteLLM proxy on port $PROXY_PORT..."
    if [ -n "$RUN_PROXY_MODULE" ]; then
        start_detached "$PROXY_LOG" env PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            "$VENV_PYTHON" -m "$RUN_PROXY_MODULE"
    else
        start_detached "$PROXY_LOG" env PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            "$VENV_PYTHON" "$RUN_PROXY"
    fi

    local pid=$!
    echo $pid > "$PROXY_PID_FILE"
    echo "   Proxy PID: $pid"

    # Wait for proxy to come up
    echo -n "   Waiting for proxy"
    for i in $(seq 1 "$PROXY_START_TIMEOUT"); do
        sleep 1
        echo -n "."
        if ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            echo "   ⚠️  Proxy process exited — check $PROXY_LOG"
            rm -f "$PROXY_PID_FILE"
            tail -n 20 "$PROXY_LOG" 2>/dev/null | tail -10
            return 1
        fi
        if proxy_ready; then
            echo " ✅"
            return 0
        fi
    done
    echo ""
    echo "   ⚠️  Proxy did not respond — check $PROXY_LOG"
    return 1
}

# ── Stop ─────────────────────────────────────────────────────────────────────
stop_all() {
    echo "🛑 Stopping MLX serving stack..."
    for pidfile in "$BACKEND_PID_FILE" "$STATS_PID_FILE" "$PROXY_PID_FILE"; do
        cleanup_pid "$pidfile"
    done
    rm -f "$PID_FILE"
    echo "✅ All services stopped"
}

# ── Status ───────────────────────────────────────────────────────────────────
show_status() {
    echo ""
    echo "=== MLX Stack Status ==="
    echo ""
    echo "vLLM MLX Backend (port $BACKEND_PORT):"
    if port_listening "$BACKEND_PORT"; then
        echo "   ✅ Running (PID: $(cat "$BACKEND_PID_FILE" 2>/dev/null || echo 'unknown'))"
    else
        echo "   ❌ Not running"
    fi

    echo ""
    echo "LiteLLM Proxy (port $PROXY_PORT):"
    if proxy_ready; then
        echo "   ✅ Running (PID: $(cat "$PROXY_PID_FILE" 2>/dev/null || echo 'unknown'))"
    else
        echo "   ❌ Not running"
    fi

    echo ""
    echo "Token Stats Server (port $STATS_PORT):"
    if port_listening "$STATS_PORT"; then
        echo "   ✅ Running (PID: $(cat "$STATS_PID_FILE" 2>/dev/null || echo 'unknown'))"
    else
        echo "   ❌ Not running"
    fi

    echo ""
}

start_all() {
    stop_all 2>/dev/null || true
    if ! start_backend; then
        echo ""
        echo "❌ Backend is not healthy, so the proxy was not started."
        echo "   Most likely the model download/load was interrupted or exceeded BACKEND_START_TIMEOUT."
        echo "   Resume with: ./mlx-start.sh start"
        exit 1
    fi
    echo ""
    echo "   ✅ Backend model is loaded and healthy."
    echo ""
    start_stats || echo "   ⚠️  Continuing without token stats server."
    sleep 1
    if ! start_proxy; then
        echo ""
        echo "❌ Proxy failed to start. Backend is still healthy; check $PROXY_LOG."
        exit 1
    fi
    sleep 1

    echo ""
    echo "   vLLM MLX backend:  http://$HOST:$BACKEND_PORT"
    echo "   LiteLLM proxy:    http://$HOST:$PROXY_PORT"
    echo "   Token stats:      http://$HOST:$STATS_PORT"
    echo ""
    echo "   Test:  curl http://localhost:$BACKEND_PORT/health"
    echo "   Logs:  ./mlx-start.sh logs"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────────────
case "${1:-start}" in
    start|both)
        start_all

        # Stream logs live in background — press Ctrl+C to stop tailing (servers keep running)
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📋 Live logs — press Ctrl+C to stop tailing (servers keep running)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        {
            echo "━━━ vLLM MLX ($BACKEND_PORT) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            tail -n 30 -f "$BACKEND_LOG" 2>/dev/null
        } &
        BACKEND_LOG_PID=$!

        {
            echo "━━━ LiteLLM Proxy ($PROXY_PORT) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            tail -n 15 -f "$PROXY_LOG" 2>/dev/null
        } &
        PROXY_LOG_PID=$!

        {
            echo "━━━ Token Stats ($STATS_PORT) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            tail -n 15 -f "$STATS_LOG" 2>/dev/null
        } &
        STATS_LOG_PID=$!

        # Wait for any log tail to die (user Ctrl+C), then clean up tail processes
        wait $BACKEND_LOG_PID $PROXY_LOG_PID $STATS_LOG_PID 2>/dev/null

        # Kill remaining tails (user pressed Ctrl+C)
        kill $BACKEND_LOG_PID $PROXY_LOG_PID $STATS_LOG_PID 2>/dev/null
        echo ""
        echo "✅ Servers still running in background. Restart: ./mlx-start.sh start"
        ;;

    daemon|start-no-tail)
        start_all
        echo "✅ Servers running detached. Stop: ./mlx-start.sh stop"
        ;;

    backend)
        start_backend
        ;;

    proxy)
        start_proxy
        ;;

    stats)
        start_stats
        ;;

    stop)
        stop_all
        ;;

    status)
        show_status
        ;;

    restart)
        stop_all 2>/dev/null || true
        sleep 2
        "$0" start
        ;;

    logs)
        echo "=== vLLM MLX Logs (last 50 lines) ==="
        tail -50 "$BACKEND_LOG" 2>/dev/null || echo "No log file"
        echo ""
        echo "=== Proxy Logs (last 30 lines) ==="
        tail -30 "$PROXY_LOG" 2>/dev/null || echo "No log file"
        echo ""
        echo "=== Stats Logs (last 20 lines) ==="
        tail -20 "$STATS_LOG" 2>/dev/null || echo "No log file"
        ;;

    *)
        echo "Usage: $0 [start|daemon|backend|proxy|stats|stop|restart|status|logs]"
        echo ""
        echo "  start     - start all services (default)"
        echo "  daemon    - start all services without live log tailing"
        echo "  backend   - start only vLLM MLX backend"
        echo "  proxy     - start only LiteLLM proxy"
        echo "  stats     - start only token stats server"
        echo "  stop      - stop all services"
        echo "  restart   - restart all services"
        echo "  status    - show service status"
        echo "  logs      - tail recent logs"
        exit 1
        ;;
esac
