#!/usr/bin/env bash
#
# Pre-download the serving stack models into stable local directories for offline use.
#
# Usage:
#   ./download_offline_models.sh
#   ./download_offline_models.sh --models-dir /path/to/models
#   ./download_offline_models.sh --skip-asr
#   ./download_offline_models.sh --backend-model RedHatAI/Qwen3.6-35B-A3B-NVFP4
#   ./download_offline_models.sh --embedding-model unsloth/Qwen3-Embedding-4B
#   ./download_offline_models.sh --asr-model Qwen/Qwen3-ASR-1.7B
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODELS_DIR="${QWEN_OFFLINE_MODELS_DIR:-$SCRIPT_DIR/models}"
BACKEND_MODEL="${QWEN_BACKEND_MODEL_ID:-RedHatAI/Qwen3.6-35B-A3B-NVFP4}"
EMBEDDING_MODEL="${QWEN_RAG_EMBED_MODEL_ID:-unsloth/Qwen3-Embedding-4B}"
ASR_MODEL="${QWEN_ASR_MODEL_ID:-Qwen/Qwen3-ASR-1.7B}"

DOWNLOAD_BACKEND=1
DOWNLOAD_EMBEDDING=1
DOWNLOAD_ASR=1

usage() {
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --backend-model)
            BACKEND_MODEL="$2"
            shift 2
            ;;
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --asr-model)
            ASR_MODEL="$2"
            shift 2
            ;;
        --skip-backend)
            DOWNLOAD_BACKEND=0
            shift
            ;;
        --skip-embedding)
            DOWNLOAD_EMBEDDING=0
            shift
            ;;
        --skip-asr)
            DOWNLOAD_ASR=0
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

BACKEND_DIR="$MODELS_DIR/backend"
EMBEDDING_DIR="$MODELS_DIR/embedding"
ASR_DIR="$MODELS_DIR/asr"
mkdir -p "$MODELS_DIR"

download_model() {
    local label="$1"
    local model_id="$2"
    local target_dir="$3"

    echo "Downloading $label model:"
    echo "  model: $model_id"
    echo "  local dir: $target_dir"
    mkdir -p "$target_dir"
    "${HF_CMD[@]}" download "$model_id" --local-dir "$target_dir"
    echo ""
}

[[ "$DOWNLOAD_BACKEND" -eq 1 ]] && download_model "backend" "$BACKEND_MODEL" "$BACKEND_DIR"
[[ "$DOWNLOAD_EMBEDDING" -eq 1 ]] && download_model "embedding" "$EMBEDDING_MODEL" "$EMBEDDING_DIR"
[[ "$DOWNLOAD_ASR" -eq 1 ]] && download_model "ASR" "$ASR_MODEL" "$ASR_DIR"

cat <<EOF
Offline model prep complete.

Local model directories:
  backend:   $BACKEND_DIR
  embedding: $EMBEDDING_DIR
  asr:       $ASR_DIR

Start fully offline with:
  ./start_offline.sh both

The offline starter exports:
  QWEN_BACKEND_MODEL=$BACKEND_DIR
  QWEN_RAG_EMBED_MODEL=$EMBEDDING_DIR
  QWEN_ASR_MODEL=$ASR_DIR
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
EOF
