#!/usr/bin/env bash
#
# Prepare a separate MLX profile for mlx-community/Qwen3.6-27B-4bit.
#
# This reuses the project venv and server code, but writes its own:
#   - .mlx-27b.env
#   - lite_llm_config_27b.yaml
#   - logs/qwen36-27b/
#   - .mlx-27b-*.pid files
#
# Usage:
#   ./setup-mlx-27b.sh
#   ./setup-mlx-27b.sh --no-install
#   ./setup-mlx-27b.sh --download-model
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-mlx-community/Qwen3.6-27B-4bit}"
export MODEL_NAME="${MODEL_NAME:-qwen3.6-27b-4bit}"
export MODEL_ALIAS="${MODEL_ALIAS:-qwen36-27b}"
export BACKEND_PORT="${BACKEND_PORT:-11124}"
export PROXY_PORT="${PROXY_PORT:-11125}"
export STATS_PORT="${STATS_PORT:-11126}"
export PID_PREFIX="${PID_PREFIX:-.mlx-27b}"
export CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-$SCRIPT_DIR/lite_llm_config_27b.yaml}"
export QWEN_LOG_DIR="${QWEN_LOG_DIR:-$SCRIPT_DIR/logs/qwen36-27b}"
export ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.mlx-27b.env}"
export SETUP_COMMAND="${SETUP_COMMAND:-./setup-mlx-27b.sh}"
export START_COMMAND="${START_COMMAND:-./mlx-start-27b.sh}"

exec "$SCRIPT_DIR/setup-mlx.sh" "$@"
