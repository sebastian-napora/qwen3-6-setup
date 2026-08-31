#!/usr/bin/env bash
#
# Start the serving stack using only local model directories with Hugging Face offline mode enabled.
#
# Usage:
#   ./start_offline.sh
#   ./start_offline.sh both
#   ./start_offline.sh backend
#   ./start_offline.sh proxy
#   ./start_offline.sh asr
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TARGET="${1:-both}"
OFFLINE_MODELS_DIR="${QWEN_OFFLINE_MODELS_DIR:-$SCRIPT_DIR/models}"

export QWEN_BACKEND_MODEL="${QWEN_BACKEND_MODEL:-$OFFLINE_MODELS_DIR/backend}"
export QWEN_RAG_EMBED_MODEL="${QWEN_RAG_EMBED_MODEL:-$OFFLINE_MODELS_DIR/embedding}"
export QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-$OFFLINE_MODELS_DIR/asr}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export LITELLM_LOCAL_MODEL_COST_MAP="${LITELLM_LOCAL_MODEL_COST_MAP:-True}"

require_dir() {
    local label="$1"
    local path="$2"
    if [[ ! -d "$path" ]]; then
        echo "❌ Missing $label model directory: $path" >&2
        if [[ "${QWEN_OFFLINE_MODELS_DIR:-}" != "" ]]; then
            echo "Expected offline model root from QWEN_OFFLINE_MODELS_DIR: $OFFLINE_MODELS_DIR" >&2
            echo "Prepare offline models there with: QWEN_OFFLINE_MODELS_DIR=\"$OFFLINE_MODELS_DIR\" ./download_offline_models.sh" >&2
        else
            echo "Prepare offline models first with: ./download_offline_models.sh" >&2
        fi
        exit 1
    fi
}

case "$TARGET" in
    both)
        require_dir "backend" "$QWEN_BACKEND_MODEL"
        require_dir "embedding" "$QWEN_RAG_EMBED_MODEL"
        require_dir "ASR" "$QWEN_ASR_MODEL"
        ;;
    backend)
        require_dir "backend" "$QWEN_BACKEND_MODEL"
        ;;
    proxy)
        require_dir "embedding" "$QWEN_RAG_EMBED_MODEL"
        ;;
    asr)
        require_dir "ASR" "$QWEN_ASR_MODEL"
        ;;
    stats)
        ;;
    *)
        echo "Usage: $0 [both|backend|proxy|stats|asr]" >&2
        exit 1
        ;;
esac

echo "🔒 Offline mode enabled"
echo "Backend model:   $QWEN_BACKEND_MODEL"
echo "Embedding model: $QWEN_RAG_EMBED_MODEL"
echo "ASR model:       $QWEN_ASR_MODEL"

exec "$SCRIPT_DIR/start.sh" "$@"
