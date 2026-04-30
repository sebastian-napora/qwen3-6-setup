#!/usr/bin/env bash
#
# install.sh — One-shot setup for the Qwen3.6 serving stack.
#
# What it does:
#   1. Verifies a usable Python interpreter (>= 3.12).
#   2. Creates ./venv if missing.
#   3. Upgrades pip / setuptools / wheel inside the venv.
#   4. Installs this project editable, with [runtime] extras
#      (vllm, fastapi, uvicorn) and uvloop.
#   5. Prints next-step commands.
#
# Usage:
#   ./install.sh                 # default: full install (runtime + dev extras off)
#   ./install.sh --no-runtime    # skip vllm/fastapi/uvicorn (proxy-only host)
#   ./install.sh --dev           # also install dev extras (build, pyinstaller)
#   ./install.sh --recreate      # delete and rebuild ./venv from scratch
#   ./install.sh --python python3.12  # pin a specific interpreter
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
PYTHON_BIN=""
INSTALL_RUNTIME=1
INSTALL_DEV=0
RECREATE=0

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-runtime)  INSTALL_RUNTIME=0; shift ;;
        --dev)         INSTALL_DEV=1;     shift ;;
        --recreate)    RECREATE=1;        shift ;;
        --python)      PYTHON_BIN="$2";   shift 2 ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ── Detect Python ─────────────────────────────────────────────────────────────
if [[ -z "$PYTHON_BIN" ]]; then
    for cand in python3.14 python3.13 python3.12 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            PYTHON_BIN="$cand"
            break
        fi
    done
fi

if [[ -z "$PYTHON_BIN" ]] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "❌ No suitable python3 interpreter found. Use --python /path/to/python3" >&2
    exit 1
fi

PY_VER="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "🐍 Using interpreter: $PYTHON_BIN ($PY_VER)"

if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)'; then
    echo "❌ Python >= 3.12 required (found $PY_VER)" >&2
    exit 1
fi

# ── Venv ──────────────────────────────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/venv"

if [[ "$RECREATE" -eq 1 && -d "$VENV_DIR" ]]; then
    echo "🧹 Removing existing venv at $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "📦 Creating venv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "📦 Reusing existing venv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── Build base toolchain ──────────────────────────────────────────────────────
echo "⬆️  Upgrading pip / setuptools / wheel"
python -m pip install --upgrade pip setuptools wheel

# ── Build extras list ─────────────────────────────────────────────────────────
EXTRAS=()
[[ "$INSTALL_RUNTIME" -eq 1 ]] && EXTRAS+=("runtime")
[[ "$INSTALL_DEV"     -eq 1 ]] && EXTRAS+=("dev")

if [[ "${#EXTRAS[@]}" -gt 0 ]]; then
    EXTRA_SPEC="[$(IFS=,; echo "${EXTRAS[*]}")]"
else
    EXTRA_SPEC=""
fi

echo "📥 Installing project (editable) with extras: ${EXTRAS[*]:-none}"
python -m pip install -e ".${EXTRA_SPEC}"

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "🔎 Verifying installed console scripts"
for cmd in qwen-server qwen-compress qwen-stats; do
    if command -v "$cmd" >/dev/null 2>&1; then
        echo "   ✅ $cmd  ->  $(command -v "$cmd")"
    else
        echo "   ⚠️  $cmd not found on PATH"
    fi
done

echo "🔎 Verifying module imports"
python - <<'PY'
import importlib, sys
mods = ["qwen3_6_server", "server_compress", "qwen_token_stats_server", "uvloop"]
failed = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"   ✅ {m}")
    except Exception as e:
        failed.append((m, e))
        print(f"   ❌ {m}: {e}")
sys.exit(1 if failed else 0)
PY

# ── Done ──────────────────────────────────────────────────────────────────────
cat <<EOF

✅ Installation complete.

Next steps:
  source venv/bin/activate
  ./start.sh           # start vLLM backend + proxy + stats
  ./kill.sh            # stop everything

Health checks:
  curl http://localhost:11112/health   # vLLM backend
  curl http://localhost:11111/health   # LiteLLM proxy
EOF
