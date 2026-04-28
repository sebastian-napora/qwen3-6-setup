#!/bin/bash
#
# Start Qwen3.6-35B-A3B-4bit MLX serving stack (Apple Silicon)
#
# Usage:
#   ./mlx-start.sh          # start all (backend + stats + proxy)
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

# ── Model & System Config ────────────────────────────────────────────────────
MODEL="mlx-community/Qwen3.6-35B-A3B-4bit"
BACKEND_PORT="11114"
PROXY_PORT="11115"
STATS_PORT="11116"
HOST="0.0.0.0"

# MLX-specific settings
CACHE_MEMORY_PERCENT="20"       # % of unified memory for KV cache
MAX_TOKENS="30000"

# ── Paths ─────────────────────────────────────────────────────────────────────
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"

mkdir -p logs

# ── PID Helpers ───────────────────────────────────────────────────────────────
PID_FILE="$SCRIPT_DIR/.mlx-pids"

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
    if curl -sf "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1; then
        echo "   ✅ Backend already running on port $BACKEND_PORT"
        return
    fi

    echo "🚀 Starting vLLM MLX backend on port $BACKEND_PORT..."
    echo "   Model: $MODEL"

    nohup vllm-mlx serve "$MODEL" \
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
        >> logs/vllm-mlx.log 2>&1 &

    local pid=$!
    echo $pid > "$SCRIPT_DIR/.mlx-backend.pid"
    echo "   Backend PID: $pid"
    echo "   Log: $SCRIPT_DIR/logs/vllm-mlx.log"

    # Wait for server to start writing to log before confirming
    echo -n "   Waiting for server to come up"
    for i in $(seq 1 40); do
        sleep 1
        echo -n "."
        if curl -sf "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1; then
            echo ""
            echo "   ✅ Backend up on port $BACKEND_PORT"
            tail -n 10 logs/vllm-mlx.log 2>/dev/null | grep -E "INFO|WARNING|ERROR|loaded|Uvicorn" | tail -5
            return
        fi
    done
    echo ""
    echo "   ⚠️  Backend did not respond in 20s — check logs/vllm-mlx.log"
    tail -n 20 logs/vllm-mlx.log 2>/dev/null | tail -10
}

# ── Stats Server ─────────────────────────────────────────────────────────────
start_stats() {
    if curl -sf "http://localhost:$STATS_PORT/" > /dev/null 2>&1; then
        echo "   ✅ Stats server already running on port $STATS_PORT"
        return
    fi

    echo "🚀 Starting token stats server on port $STATS_PORT..."
    nohup "$VENV_PYTHON" qwen_token_stats_server.py >> logs/stats.log 2>&1 &

    local pid=$!
    echo $pid > "$SCRIPT_DIR/.mlx-stats.pid"
    echo "   Stats PID: $pid"

    # Wait for stats server to come up
    echo -n "   Waiting for stats server"
    for i in $(seq 1 10); do
        sleep 1
        echo -n "."
        if curl -sf "http://localhost:$STATS_PORT/" > /dev/null 2>&1; then
            echo " ✅"
            return
        fi
    done
    echo ""
    echo "   ⚠️  Stats server did not respond — check logs/stats.log"
}

# ── LiteLLM Proxy ────────────────────────────────────────────────────────────
start_proxy() {
    if curl -sf "http://localhost:$PROXY_PORT/health" > /dev/null 2>&1; then
        echo "   ✅ Proxy already running on port $PROXY_PORT"
        return
    fi

    echo "🚀 Starting LiteLLM proxy on port $PROXY_PORT..."
    nohup "$VENV_PYTHON" server_compress.py >> logs/proxy.log 2>&1 &

    local pid=$!
    echo $pid > "$SCRIPT_DIR/.mlx-proxy.pid"
    echo "   Proxy PID: $pid"

    # Wait for proxy to come up
    echo -n "   Waiting for proxy"
    for i in $(seq 1 10); do
        sleep 1
        echo -n "."
        if curl -sf "http://localhost:$PROXY_PORT/health" > /dev/null 2>&1; then
            echo " ✅"
            return
        fi
    done
    echo ""
    echo "   ⚠️  Proxy did not respond — check logs/proxy.log"
}

# ── Stop ─────────────────────────────────────────────────────────────────────
stop_all() {
    echo "🛑 Stopping MLX serving stack..."
    for pidfile in .mlx-backend.pid .mlx-stats.pid .mlx-proxy.pid; do
        cleanup_pid "$SCRIPT_DIR/$pidfile"
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
    if curl -sf "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1; then
        echo "   ✅ Running (PID: $(cat "$SCRIPT_DIR/.mlx-backend.pid" 2>/dev/null || echo 'unknown'))"
    else
        echo "   ❌ Not running"
    fi

    echo ""
    echo "LiteLLM Proxy (port $PROXY_PORT):"
    if curl -sf "http://localhost:$PROXY_PORT/health" > /dev/null 2>&1; then
        echo "   ✅ Running (PID: $(cat "$SCRIPT_DIR/.mlx-proxy.pid" 2>/dev/null || echo 'unknown'))"
    else
        echo "   ❌ Not running"
    fi

    echo ""
    echo "Token Stats Server (port $STATS_PORT):"
    if curl -sf "http://localhost:$STATS_PORT/" > /dev/null 2>&1; then
        echo "   ✅ Running (PID: $(cat "$SCRIPT_DIR/.mlx-stats.pid" 2>/dev/null || echo 'unknown'))"
    else
        echo "   ❌ Not running"
    fi

    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────────────
case "${1:-start}" in
    start|both)
        stop_all 2>/dev/null || true
        start_backend
        echo ""
        echo "   ⏳ Waiting for backend model to load..."
        echo ""
        start_stats
        sleep 1
        start_proxy
        sleep 1

        # Stream logs live in background — press Ctrl+C to stop tailing (servers keep running)
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📋 Live logs — press Ctrl+C to stop tailing (servers keep running)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        {
            echo "━━━ vLLM MLX (11114) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            tail -n 30 -f logs/vllm-mlx.log 2>/dev/null
        } &
        BACKEND_LOG_PID=$!

        {
            echo "━━━ LiteLLM Proxy (11115) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            tail -n 15 -f logs/proxy.log 2>/dev/null
        } &
        PROXY_LOG_PID=$!

        {
            echo "━━━ Token Stats (11116) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            tail -n 15 -f logs/stats.log 2>/dev/null
        } &
        STATS_LOG_PID=$!

        echo ""
        echo "   vLLM MLX backend:  http://$HOST:$BACKEND_PORT"
        echo "   LiteLLM proxy:    http://$HOST:$PROXY_PORT"
        echo "   Token stats:      http://$HOST:$STATS_PORT"
        echo ""
        echo "   Test:  curl http://localhost:$BACKEND_PORT/health"
        echo "   Logs:  ./mlx-start.sh logs"
        echo ""

        # Wait for any log tail to die (user Ctrl+C), then clean up tail processes
        wait $BACKEND_LOG_PID $PROXY_LOG_PID $STATS_LOG_PID 2>/dev/null

        # Kill remaining tails (user pressed Ctrl+C)
        kill $BACKEND_LOG_PID $PROXY_LOG_PID $STATS_LOG_PID 2>/dev/null
        echo ""
        echo "✅ Servers still running in background. Restart: ./mlx-start.sh start"
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
        tail -50 logs/vllm-mlx.log 2>/dev/null || echo "No log file"
        echo ""
        echo "=== Proxy Logs (last 30 lines) ==="
        tail -30 logs/proxy.log 2>/dev/null || echo "No log file"
        echo ""
        echo "=== Stats Logs (last 20 lines) ==="
        tail -20 logs/stats.log 2>/dev/null || echo "No log file"
        ;;

    *)
        echo "Usage: $0 [start|backend|proxy|stats|stop|restart|status|logs]"
        echo ""
        echo "  start     - start all services (default)"
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