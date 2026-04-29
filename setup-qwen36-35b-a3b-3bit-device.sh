#!/usr/bin/env bash
#
# User-facing setup helper for a separate device.
#
# This prepares the Unsloth Qwen3.6-35B-A3B 3-bit MLX profile without touching
# the existing 27B profile. Run it from this repository on the target Mac.
#
# Usage:
#   ./setup-qwen36-35b-a3b-3bit-device.sh
#   ./setup-qwen36-35b-a3b-3bit-device.sh --download-model
#   ./setup-qwen36-35b-a3b-3bit-device.sh --download-model --start
#   ./setup-qwen36-35b-a3b-3bit-device.sh --no-install --download-model
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DOWNLOAD_MODEL="no"
START_AFTER="no"
INSTALL_DEPS="yes"
CACHE_DIR=""

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --download-model   Download unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit now.
  --start            Start the separate 35B-A3B 3-bit stack after setup.
  --no-install       Skip dependency installation and only write/update config.
  --cache-dir DIR    Hugging Face cache root. Default: ~/.cache/huggingface.
  -h, --help         Show this help.

Profile created:
  Model:        unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit
  LiteLLM name: qwen3.6-35b-a3b-3bit
  Alias:        qwen36-35b-a3b-3bit
  Backend:      http://localhost:11134/v1
  Proxy:        http://localhost:11135/v1
  Stats:        http://localhost:11136
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --download-model)
            DOWNLOAD_MODEL="yes"
            shift
            ;;
        --start)
            START_AFTER="yes"
            shift
            ;;
        --no-install)
            INSTALL_DEPS="no"
            shift
            ;;
        --cache-dir)
            CACHE_DIR="${2:?Missing value for --cache-dir}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo ""
            usage
            exit 1
            ;;
    esac
done

require_file() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "ERROR: Missing required file: $file"
        echo "Run this from the qwen3-6-setup repository on the target device."
        exit 1
    fi
}

require_file "$SCRIPT_DIR/setup-mlx.sh"
require_file "$SCRIPT_DIR/setup-mlx-35b-a3b-3bit.sh"
require_file "$SCRIPT_DIR/mlx-start.sh"
require_file "$SCRIPT_DIR/mlx-start-35b-a3b-3bit.sh"

echo ""
echo "==> Target profile"
echo "Model:   unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit"
echo "Proxy:   http://localhost:11135/v1"
echo "Backend: http://localhost:11134/v1"
echo ""

os_name="$(uname -s)"
arch_name="$(uname -m)"
echo "OS:   $os_name"
echo "Arch: $arch_name"
if [ "$os_name" != "Darwin" ] || [ "$arch_name" != "arm64" ]; then
    echo "WARNING: MLX serving is intended for Apple Silicon macOS."
fi

echo ""
echo "==> Disk space"
df -h "${CACHE_DIR:-$HOME/.cache/huggingface}" 2>/dev/null || df -h "$HOME"

setup_args=()
if [ "$INSTALL_DEPS" = "no" ]; then
    setup_args+=(--no-install)
fi
if [ -n "$CACHE_DIR" ]; then
    setup_args+=(--cache-dir "$CACHE_DIR")
fi
if [ "$DOWNLOAD_MODEL" = "yes" ]; then
    setup_args+=(--download-model)
else
    setup_args+=(--skip-download)
fi

echo ""
echo "==> Running profile setup"
"$SCRIPT_DIR/setup-mlx-35b-a3b-3bit.sh" "${setup_args[@]}"

if [ "$START_AFTER" = "yes" ]; then
    echo ""
    echo "==> Starting 35B-A3B 3-bit stack"
    "$SCRIPT_DIR/mlx-start-35b-a3b-3bit.sh" daemon
fi

echo ""
echo "==> Done"
echo "Start later with:"
echo "  ./mlx-start-35b-a3b-3bit.sh daemon"
echo ""
echo "Check status/logs:"
echo "  ./mlx-start-35b-a3b-3bit.sh status"
echo "  ./mlx-start-35b-a3b-3bit.sh logs"
echo ""
echo "Configure VS Code/Copilot gateway with:"
echo "  Base URL: http://localhost:11135/v1"
echo "  Model:    qwen3.6-35b-a3b-3bit"
