#!/usr/bin/env bash
#
# Prepare a separate MLX profile for unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit.
#
# This reuses the project venv and server code, but writes its own:
#   - .mlx-35b-a3b-3bit.env
#   - lite_llm_config_35b_a3b_3bit.yaml
#   - logs/qwen36-35b-a3b-3bit/
#   - .mlx-35b-a3b-3bit-*.pid files
#
# Usage:
#   ./setup-mlx-35b-a3b-3bit.sh
#   ./setup-mlx-35b-a3b-3bit.sh --no-install
#   ./setup-mlx-35b-a3b-3bit.sh --download-model
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit}"
export MODEL_NAME="${MODEL_NAME:-qwen3.6-35b-a3b-3bit}"
export MODEL_ALIAS="${MODEL_ALIAS:-qwen36-35b-a3b-3bit}"
export BACKEND_PORT="${BACKEND_PORT:-11134}"
export PROXY_PORT="${PROXY_PORT:-11135}"
export STATS_PORT="${STATS_PORT:-11136}"
export PID_PREFIX="${PID_PREFIX:-.mlx-35b-a3b-3bit}"
export CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-$SCRIPT_DIR/lite_llm_config_35b_a3b_3bit.yaml}"
export QWEN_LOG_DIR="${QWEN_LOG_DIR:-$SCRIPT_DIR/logs/qwen36-35b-a3b-3bit}"
export ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.mlx-35b-a3b-3bit.env}"
export SETUP_COMMAND="${SETUP_COMMAND:-./setup-mlx-35b-a3b-3bit.sh}"
export START_COMMAND="${START_COMMAND:-./mlx-start-35b-a3b-3bit.sh}"

exec "$SCRIPT_DIR/setup-mlx.sh" "$@"
