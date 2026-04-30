#!/bin/bash
#
# Start the stack with the Apple Silicon MLX embedding backend.
#
# Usage:
#   ./mlx-start.sh          # start LiteLLM proxy + local RAG embeddings
#   ./mlx-start.sh proxy    # same as above
#   ./mlx-start.sh both     # start vLLM backend + stats + proxy
#   ./mlx-start.sh --download proxy
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DOWNLOAD_FIRST=0
if [[ "${1:-}" == "--download" ]]; then
    DOWNLOAD_FIRST=1
    shift
fi

export QWEN_RAG_EMBED_MODEL="${QWEN_RAG_EMBED_MODEL:-mlx-community/Qwen3-Embedding-8B-4bit-DWQ}"
export QWEN_RAG_EMBED_BACKEND="${QWEN_RAG_EMBED_BACKEND:-mlx}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "Warning: MLX embeddings need Apple Silicon macOS with Metal."
    echo "Use ./start.sh for the default Unsloth/HF embedding backend on DGX/NVIDIA."
fi

if [[ "$DOWNLOAD_FIRST" -eq 1 ]]; then
    ./download_embedding_model.sh --model "$QWEN_RAG_EMBED_MODEL"
fi

exec ./start.sh "${1:-proxy}"
