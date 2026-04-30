#!/usr/bin/env bash
#
# Pre-download the local RAG embedding model into the Hugging Face cache.
#
# Usage:
#   ./download_embedding_model.sh
#   ./download_embedding_model.sh --verify
#   ./download_embedding_model.sh --preset unsloth-4b
#   ./download_embedding_model.sh --model mlx-community/Qwen3-Embedding-0.6B
#   ./download_embedding_model.sh --cache-dir /path/to/hf-cache
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_ID="${QWEN_RAG_EMBED_MODEL:-mlx-community/Qwen3-Embedding-8B-4bit-DWQ}"
CACHE_DIR=""
REVISION=""
VERIFY=0

preset_model() {
    case "$1" in
        mlx-8b|mlx-qwen3-8b-4bit)
            echo "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"
            ;;
        unsloth-4b)
            echo "unsloth/Qwen3-Embedding-4B"
            ;;
        qwen-4b)
            echo "Qwen/Qwen3-Embedding-4B"
            ;;
        qwen-0.6b)
            echo "Qwen/Qwen3-Embedding-0.6B"
            ;;
        *)
            echo "Unknown preset: $1" >&2
            echo "Available presets: mlx-8b, unsloth-4b, qwen-4b, qwen-0.6b" >&2
            exit 1
            ;;
    esac
}

usage() {
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_ID="$2"
            shift 2
            ;;
        --preset)
            MODEL_ID="$(preset_model "$2")"
            shift 2
            ;;
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --revision)
            REVISION="$2"
            shift 2
            ;;
        --verify)
            VERIFY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "Warning: MLX embedding runtime needs Apple Silicon macOS with Metal."
    echo "This script can still download the model, but runtime embedding may not work here."
fi

if [[ -x "$SCRIPT_DIR/venv/bin/hf" ]]; then
    HF_CMD=("$SCRIPT_DIR/venv/bin/hf")
elif command -v hf >/dev/null 2>&1; then
    HF_CMD=("hf")
elif [[ -x "$SCRIPT_DIR/venv/bin/huggingface-cli" ]]; then
    HF_CMD=("$SCRIPT_DIR/venv/bin/huggingface-cli")
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD=("huggingface-cli")
else
    cat >&2 <<'EOF'
Could not find the Hugging Face CLI.

Run ./install.sh first, then try again. The installer pulls dependencies that
include the Hugging Face download tooling.
EOF
    exit 1
fi

DOWNLOAD_CMD=("${HF_CMD[@]}" download "$MODEL_ID")
if [[ -n "$REVISION" ]]; then
    DOWNLOAD_CMD+=(--revision "$REVISION")
fi
if [[ -n "$CACHE_DIR" ]]; then
    DOWNLOAD_CMD+=(--cache-dir "$CACHE_DIR")
fi

echo "Downloading embedding model:"
echo "  model: $MODEL_ID"
if [[ -n "$CACHE_DIR" ]]; then
    echo "  cache: $CACHE_DIR"
else
    echo "  cache: default Hugging Face cache"
fi
echo ""

DOWNLOAD_PATH="$("${DOWNLOAD_CMD[@]}")"

echo ""
echo "Embedding model is available at:"
echo "  $DOWNLOAD_PATH"
du -sh "$DOWNLOAD_PATH" 2>/dev/null || true

if [[ "$VERIFY" -eq 0 ]]; then
    cat <<EOF

Done.

To use it:
  ./start.sh proxy
  curl http://localhost:11111/v1/local_rag/health

To verify the selected local embedding backend can run the model:
  ./download_embedding_model.sh --verify

For DGX/NVIDIA with the Unsloth 4B embedding model:
  ./install.sh --hf-embeddings
  ./download_embedding_model.sh --preset unsloth-4b --verify
  QWEN_RAG_EMBED_MODEL=unsloth/Qwen3-Embedding-4B ./start.sh proxy
EOF
    exit 0
fi

if [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
elif [[ -x "$SCRIPT_DIR/venv/bin/python3" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python3"
else
    PYTHON_BIN="python3"
fi

echo ""
echo "Running MLX embedding smoke test..."
QWEN_RAG_EMBED_MODEL="$MODEL_ID" "$PYTHON_BIN" - <<'PY'
import os

import numpy as np

import qwen_local_rag

model_id = os.environ["QWEN_RAG_EMBED_MODEL"]
vectors = qwen_local_rag.embedder.embed_texts(
    ["local RAG embedding smoke test"],
    model_id=model_id,
    batch_size=1,
)
vector = vectors[0]
print("  model:", qwen_local_rag.embedder.loaded_model)
print("  shape:", vector.shape)
print("  norm:", round(float(np.linalg.norm(vector)), 6))
PY

echo "Done."
