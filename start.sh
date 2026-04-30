#!/bin/bash
#
# Start Qwen3.6-35B-A3B-NVFP4 serving stack
#
# Usage:
#   ./start.sh          # start vLLM backend, token stats and LiteLLM proxy
#   ./start.sh backend  # start only vLLM backend (11112)
#   ./start.sh proxy    # start only LiteLLM proxy (11111)
#   ./start.sh stats    # start only token stats server (11113)
#
# Architecture:
#   Copilot -> LiteLLM (11111) -> vLLM (11112)
#                              -> Token stats (11113)
#   Local RAG + embeddings are mounted on the LiteLLM proxy (11111).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_RAG_EMBED_MODEL="unsloth/Qwen3-Embedding-4B"
DEFAULT_RAG_EMBED_BACKEND="hf"

export QWEN_RAG_EMBED_MODEL="${QWEN_RAG_EMBED_MODEL:-$DEFAULT_RAG_EMBED_MODEL}"
export QWEN_RAG_EMBED_BACKEND="${QWEN_RAG_EMBED_BACKEND:-$DEFAULT_RAG_EMBED_BACKEND}"

# Auto-cleanup any prior run so port-in-use errors don't recur.
# Skip with: SKIP_CLEAN=1 ./start.sh
if [[ "${SKIP_CLEAN:-0}" != "1" && -x "$SCRIPT_DIR/kill.sh" ]]; then
    echo "🧹 Cleaning up any prior serving processes..."
    bash "$SCRIPT_DIR/kill.sh" >/dev/null 2>&1 || true
    sleep 1
fi

# Detect or create venv
if [ -d "venv" ]; then
    VENV_PYTHON="venv/bin/python3"
else
    VENV_PYTHON="python3"
fi

# Entry point commands (available when package is installed via pip/wheel)
# Falls back to relative paths for development (running from source dir)
RUN_SERVER="$($VENV_PYTHON -c "import shutil; print(shutil.which('qwen-server') or '')" 2>/dev/null)"
RUN_PROXY="$($VENV_PYTHON -c "import shutil; print(shutil.which('qwen-compress') or '')" 2>/dev/null)"
RUN_STATS="$($VENV_PYTHON -c "import shutil; print(shutil.which('qwen-stats') or '')" 2>/dev/null)"

# Fall back to relative paths (development mode)
RUN_SERVER="${RUN_SERVER:-$SCRIPT_DIR/qwen3_6_server.py}"
RUN_PROXY="${RUN_PROXY:-$SCRIPT_DIR/server_compress.py}"
RUN_STATS="${RUN_STATS:-$SCRIPT_DIR/qwen_token_stats_server.py}"

mkdir -p logs

start_backend() {
    echo "🚀 Starting vLLM backend (port 11112)..."
    if [[ -x "$RUN_SERVER" ]] && [[ ! "$RUN_SERVER" == *.py ]]; then
        $RUN_SERVER &
    else
        $VENV_PYTHON "$RUN_SERVER" &
    fi
    echo "Backend PID: $!"
}

new_token_session() {
    # Create a fresh session ID and write it to the shared session file.
    # Both the proxy and the stats server read this file at startup, so
    # every ./start.sh run begins a clean session automatically.
    NEW_SID=$($VENV_PYTHON -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import qwen_token_tracker
sid = qwen_token_tracker.new_session()
print(sid, end='')
" 2>/dev/null)
    echo "📊 New token session: ${NEW_SID:-unknown}"
}

start_stats() {
    echo "🚀 Starting token stats server (port 11113)..."
    if [[ -x "$RUN_STATS" ]] && [[ ! "$RUN_STATS" == *.py ]]; then
        $RUN_STATS &
    else
        $VENV_PYTHON "$RUN_STATS" &
    fi
    echo "Stats PID: $!"
}

start_proxy() {
    echo "🚀 Starting LiteLLM proxy (port 11111)..."
    echo "Embedding model: $QWEN_RAG_EMBED_MODEL"
    echo "Embedding backend: $QWEN_RAG_EMBED_BACKEND"
    if [[ -x "$RUN_PROXY" ]] && [[ ! "$RUN_PROXY" == *.py ]]; then
        $RUN_PROXY &
    else
        $VENV_PYTHON "$RUN_PROXY" &
    fi
    echo "Proxy PID: $!"
    echo "Local RAG / embeddings: http://0.0.0.0:${LITE_LLM_PROXY_PORT:-11111}/v1/local_rag/health"
}

case "${1:-both}" in
    both)
        start_backend
        new_token_session
        echo "Waiting 5s for backend to initialize..."
        sleep 5
        start_stats
        sleep 1
        start_proxy
        echo ""
        echo "✅ All services started:"
        echo "   vLLM backend:   http://0.0.0.0:11112"
        echo "   LiteLLM proxy:  http://0.0.0.0:11111"
        echo "   Local RAG:      http://0.0.0.0:11111/v1/local_rag/health"
        echo "   Token stats:    http://0.0.0.0:11113"
        echo ""
        echo "Test with:"
        echo "  curl http://localhost:11112/health"
        echo "  curl http://localhost:11111/health"
        echo "  curl http://localhost:11111/v1/local_rag/health"
        ;;
    backend)
        start_backend
        ;;
    proxy)
        new_token_session
        start_proxy
        ;;
    stats)
        start_stats
        ;;
    *)
        echo "Usage: $0 [both|backend|proxy|stats]"
        exit 1
        ;;
esac
