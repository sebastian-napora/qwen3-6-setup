#!/usr/bin/env bash
#
# Start the separate Qwen3.6-27B-4bit MLX profile.
#
# Usage:
#   ./mlx-start-27b.sh start
#   ./mlx-start-27b.sh backend
#   ./mlx-start-27b.sh status
#   ./mlx-start-27b.sh stop
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MLX_ENV_FILE="${MLX_ENV_FILE:-$SCRIPT_DIR/.mlx-27b.env}"

if [ ! -f "$MLX_ENV_FILE" ]; then
    echo "ERROR: Missing 27B env file:"
    echo "  $MLX_ENV_FILE"
    echo ""
    echo "Create it first:"
    echo "  ./setup-mlx-27b.sh --no-install --skip-download"
    exit 1
fi

exec "$SCRIPT_DIR/mlx-start.sh" "$@"
