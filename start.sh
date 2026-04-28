#!/bin/bash
#
# Start Qwen3.6-35B-A3B-NVFP4 serving stack
#
# Usage:
#   ./start.sh          # start vLLM backend, token stats and LiteLLM proxy
#   ./start.sh backend  # start only vLLM backend (11114)
#   ./start.sh proxy    # start only LiteLLM proxy (11115)
#   ./start.sh stats    # start only token stats server (11116)
#
# Architecture:
#   Copilot -> LiteLLM (11115) -> vLLM (11114)
#                              -> Token stats (11116)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use vllm-metal venv on macOS
if [ -d "/Users/sna/.venv-vllm-metal/bin/python3" ]; then
    VENV_PYTHON="/Users/sna/.venv-vllm-metal/bin/python3"
elif [ -d "venv" ]; then
    VENV_PYTHON="venv/bin/python3"
else
    VENV_PYTHON="python3"
fi

# If running from bundle (no .py files here), install the wheel first
if [ ! -f "$SCRIPT_DIR/src/qwen3_6_server.py" ] && [ -f "$SCRIPT_DIR/lunch_model-"*.whl ]; then
    echo "📦 Installing lunch-model package..."
    $VENV_PYTHON -m pip install --quiet "$SCRIPT_DIR"/lunch_model-*.whl
fi

# Entry point commands (available when package is installed via pip/wheel)
# Falls back to relative paths for development (running from source dir)
RUN_SERVER="$($VENV_PYTHON -c "import shutil; print(shutil.which('qwen-server') or '')" 2>/dev/null)"
RUN_PROXY="$($VENV_PYTHON -c "import shutil; print(shutil.which('qwen-compress') or '')" 2>/dev/null)"
RUN_STATS="$($VENV_PYTHON -c "import shutil; print(shutil.which('qwen-stats') or '')" 2>/dev/null)"

# Fall back to relative paths (development mode)
RUN_SERVER="${RUN_SERVER:-$SCRIPT_DIR/src/qwen3_6_server.py}"
RUN_PROXY="${RUN_PROXY:-$SCRIPT_DIR/src/server_compress.py}"
RUN_STATS="${RUN_STATS:-$SCRIPT_DIR/src/qwen_token_stats_server.py}"

mkdir -p logs

start_backend() {
    echo "🚀 Starting vLLM backend (port 11114)..."
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
import sys; sys.path.insert(0, '$SCRIPT_DIR/src')
from qwen_token_tracker import new_session
sid = new_session()
print(sid, end='')
" 2>/dev/null)
    echo "📊 New token session: ${NEW_SID:-unknown}"
}

start_stats() {
    echo "🚀 Starting token stats server (port 11116)..."
    if [[ -x "$RUN_STATS" ]] && [[ ! "$RUN_STATS" == *.py ]]; then
        $RUN_STATS &
    else
        $VENV_PYTHON "$RUN_STATS" &
    fi
    echo "Stats PID: $!"
}

start_proxy() {
    echo "🚀 Starting LiteLLM proxy (port 11115)..."
    if [[ -x "$RUN_PROXY" ]] && [[ ! "$RUN_PROXY" == *.py ]]; then
        $RUN_PROXY &
    else
        $VENV_PYTHON "$RUN_PROXY" &
    fi
    echo "Proxy PID: $!"
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
        echo "   vLLM backend:   http://0.0.0.0:11114"
        echo "   LiteLLM proxy:  http://0.0.0.0:11115"
        echo "   Token stats:    http://0.0.0.0:11116"
        echo ""
        echo "Test with:"
        echo "  curl http://localhost:11114/health"
        echo "  curl http://localhost:11115/health"
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
