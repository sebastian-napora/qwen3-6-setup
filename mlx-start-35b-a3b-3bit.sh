#!/usr/bin/env bash
#
# Start the separate Qwen3.6-35B-A3B 3-bit MLX profile.
#
# Usage:
#   ./mlx-start-35b-a3b-3bit.sh start
#   ./mlx-start-35b-a3b-3bit.sh daemon
#   ./mlx-start-35b-a3b-3bit.sh status
#   ./mlx-start-35b-a3b-3bit.sh stop
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MLX_ENV_FILE="${MLX_ENV_FILE:-$SCRIPT_DIR/.mlx-35b-a3b-3bit.env}"

if [ ! -f "$MLX_ENV_FILE" ]; then
    echo "ERROR: Missing 35B-A3B 3-bit env file:"
    echo "  $MLX_ENV_FILE"
    echo ""
    echo "Create it first:"
    echo "  ./setup-mlx-35b-a3b-3bit.sh --no-install --skip-download"
    exit 1
fi

exec "$SCRIPT_DIR/mlx-start.sh" "$@"
