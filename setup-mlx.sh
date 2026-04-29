#!/usr/bin/env bash
#
# Prepare this checkout for the MLX serving path used by ./mlx-start.sh.
#
# What this does:
#   1. Creates/updates a Python virtual environment.
#   2. Installs the local package, vllm-mlx runtime, and Hugging Face CLI tools.
#   3. Writes .mlx.env so mlx-start.sh uses the selected MLX model and binaries.
#   4. Optionally pre-downloads the Hugging Face model into the local HF cache.
#
# Usage:
#   ./setup-mlx.sh
#   ./setup-mlx.sh --download-model
#   ./setup-mlx.sh --model mlx-community/Qwen3.6-35B-A3B-4bit
#   ./setup-mlx.sh --cache-dir "$HOME/.cache/huggingface"
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit}"
MODEL_NAME="${MODEL_NAME:-qwen3.6-35b-nvfp4}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen36flash}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/venv}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
BACKEND_PORT="${BACKEND_PORT:-11114}"
PROXY_PORT="${PROXY_PORT:-11115}"
STATS_PORT="${STATS_PORT:-11116}"
HOST="${HOST:-0.0.0.0}"
MAX_TOKENS="${MAX_TOKENS:-30000}"
CACHE_MEMORY_PERCENT="${CACHE_MEMORY_PERCENT:-20}"
BACKEND_START_TIMEOUT="${BACKEND_START_TIMEOUT:-3600}"
PID_PREFIX="${PID_PREFIX:-.mlx}"
CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-$SCRIPT_DIR/lite_llm_config.yaml}"
QWEN_LOG_DIR="${QWEN_LOG_DIR:-$SCRIPT_DIR/logs}"
SETUP_COMMAND="${SETUP_COMMAND:-./setup-mlx.sh}"
START_COMMAND="${START_COMMAND:-./mlx-start.sh}"

DOWNLOAD_MODEL="ask"
INSTALL_DEPS="yes"
WRITE_CONFIG="yes"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.mlx.env}"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --model MODEL          Hugging Face model repo or local model path.
                         Default: $MODEL
  --model-name NAME      LiteLLM primary model name. Default: $MODEL_NAME
  --model-alias NAME     LiteLLM alias model name. Default: $MODEL_ALIAS
  --venv DIR             Virtualenv directory. Default: $VENV_DIR
  --cache-dir DIR        Hugging Face cache root. Default: $HF_HOME
  --env-file FILE        Env file for mlx-start.sh. Default: $ENV_FILE
  --config-file FILE     LiteLLM config file. Default: $CONFIG_FILE_PATH
  --log-dir DIR          Logs and token stats directory. Default: $QWEN_LOG_DIR
  --pid-prefix PREFIX    PID file prefix. Default: $PID_PREFIX
  --download-model       Pre-download the model now.
  --skip-download        Do not ask to download the model.
  --no-install           Skip pip installs; only write config.
  --no-config            Do not update lite_llm_config.yaml.
  -h, --help             Show this help.

Examples:
  ./setup-mlx.sh --download-model
  MODEL=mlx-community/Qwen3.6-35B-A3B-4bit ./setup-mlx.sh
  HF_TOKEN=hf_xxx ./setup-mlx.sh --download-model
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model)
            MODEL="${2:?Missing value for --model}"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="${2:?Missing value for --model-name}"
            shift 2
            ;;
        --model-alias)
            MODEL_ALIAS="${2:?Missing value for --model-alias}"
            shift 2
            ;;
        --venv)
            VENV_DIR="${2:?Missing value for --venv}"
            shift 2
            ;;
        --cache-dir)
            HF_HOME="${2:?Missing value for --cache-dir}"
            shift 2
            ;;
        --env-file)
            ENV_FILE="${2:?Missing value for --env-file}"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE_PATH="${2:?Missing value for --config-file}"
            shift 2
            ;;
        --log-dir)
            QWEN_LOG_DIR="${2:?Missing value for --log-dir}"
            shift 2
            ;;
        --pid-prefix)
            PID_PREFIX="${2:?Missing value for --pid-prefix}"
            shift 2
            ;;
        --download-model)
            DOWNLOAD_MODEL="yes"
            shift
            ;;
        --skip-download)
            DOWNLOAD_MODEL="no"
            shift
            ;;
        --no-install)
            INSTALL_DEPS="no"
            shift
            ;;
        --no-config)
            WRITE_CONFIG="no"
            shift
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

VENV_PYTHON="$VENV_DIR/bin/python3"
VENV_PIP="$VENV_DIR/bin/pip"
VLLM_MLX_BIN="${VLLM_MLX_BIN:-$VENV_DIR/bin/vllm-mlx}"

say() {
    echo ""
    echo "==> $*"
}

require_command() {
    local name="$1"
    if ! command -v "$name" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $name"
        exit 1
    fi
}

check_platform() {
    say "Checking platform"

    local os_name arch
    os_name="$(uname -s)"
    arch="$(uname -m)"

    echo "OS:   $os_name"
    echo "Arch: $arch"

    if [ "$os_name" != "Darwin" ]; then
        echo "WARNING: vllm-mlx is intended for Apple Silicon macOS."
    fi

    if [ "$arch" != "arm64" ]; then
        echo "WARNING: MLX acceleration needs Apple Silicon arm64."
    fi
}

check_python_version() {
    require_command python3

    python3 -c '
import sys
version = ".".join(map(str, sys.version_info[:3]))
print(f"Python: {version}")
raise SystemExit(0 if sys.version_info >= (3, 12) else 1)
' || {
        echo "ERROR: Python 3.12+ is required by this package."
        exit 1
    }
}

setup_venv() {
    say "Preparing virtual environment"

    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating: $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    else
        echo "Using existing: $VENV_DIR"
    fi

    if [ ! -x "$VENV_PYTHON" ]; then
        echo "ERROR: Could not find venv Python at $VENV_PYTHON"
        exit 1
    fi

    if [ ! -x "$VENV_PIP" ]; then
        echo "ERROR: Could not find pip at $VENV_PIP"
        echo "Try recreating the venv: rm -rf '$VENV_DIR' && ./setup-mlx.sh"
        exit 1
    fi
}

install_dependencies() {
    if [ "$INSTALL_DEPS" != "yes" ]; then
        echo "Skipping dependency installation."
        return
    fi

    say "Installing MLX runtime dependencies"

    "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
    "$VENV_PYTHON" -m pip install -e ".[runtime]"
    "$VENV_PYTHON" -m pip install "huggingface_hub[cli]" hf_transfer

    resolve_runtime_paths

    if ! is_executable_command "$VLLM_MLX_BIN"; then
        echo "ERROR: vllm-mlx was not installed at $VLLM_MLX_BIN"
        echo "Check the pip output above; vllm-mlx must install successfully before starting."
        exit 1
    fi
}

is_executable_command() {
    local value="$1"

    if [[ "$value" == */* ]]; then
        [ -x "$value" ]
        return
    fi

    command -v "$value" >/dev/null 2>&1
}

resolve_runtime_paths() {
    if is_executable_command "$VLLM_MLX_BIN"; then
        return
    fi

    if [ -x "$VENV_DIR/bin/vllm-mlx" ]; then
        VLLM_MLX_BIN="$VENV_DIR/bin/vllm-mlx"
        return
    fi

    if command -v vllm-mlx >/dev/null 2>&1; then
        VLLM_MLX_BIN="$(command -v vllm-mlx)"
        return
    fi

    # Keep mlx-start.sh usable with PATH-based installs, and let it fail with a
    # normal command-not-found error if vllm-mlx truly is not installed yet.
    VLLM_MLX_BIN="vllm-mlx"
}

write_env_file() {
    say "Writing MLX start configuration"

    cat > "$ENV_FILE" <<EOF
# Generated by setup-mlx.sh. Safe to edit.
MODEL="$MODEL"
HF_HOME="$HF_HOME"
VENV_PYTHON="$VENV_PYTHON"
VLLM_MLX_BIN="$VLLM_MLX_BIN"
BACKEND_PORT="$BACKEND_PORT"
PROXY_PORT="$PROXY_PORT"
STATS_PORT="$STATS_PORT"
HOST="$HOST"
MAX_TOKENS="$MAX_TOKENS"
CACHE_MEMORY_PERCENT="$CACHE_MEMORY_PERCENT"
BACKEND_START_TIMEOUT="$BACKEND_START_TIMEOUT"
PID_PREFIX="$PID_PREFIX"
RUN_PROXY="$SCRIPT_DIR/src/server_compress.py"
RUN_STATS="$SCRIPT_DIR/src/qwen_token_stats_server.py"
RUN_PROXY_MODULE="src.server_compress"
RUN_STATS_MODULE="src.qwen_token_stats_server"
CONFIG_FILE_PATH="$CONFIG_FILE_PATH"
QWEN_LOG_DIR="$QWEN_LOG_DIR"
QWEN_COPILOT_MIN_OUTPUT_TOKENS="20000"
ENABLE_REASONING_PARSER="no"
EOF

    chmod 600 "$ENV_FILE"
    chmod +x "$SCRIPT_DIR/mlx-start.sh"

    echo "Wrote: $ENV_FILE"
}

update_litellm_config() {
    if [ "$WRITE_CONFIG" != "yes" ]; then
        echo "Skipping lite_llm_config.yaml update."
        return
    fi

    local config="$CONFIG_FILE_PATH"

    say "Updating LiteLLM model mapping"

    "$VENV_PYTHON" - "$config" "$MODEL" "$BACKEND_PORT" "$MODEL_NAME" "$MODEL_ALIAS" <<'PY'
from pathlib import Path
import shutil
import sys

path = Path(sys.argv[1])
model = sys.argv[2]
backend_port = sys.argv[3]
model_name = sys.argv[4]
model_alias = sys.argv[5]

def render_config() -> str:
    return f"""# Generated by setup-mlx.sh.
#
# Architecture:
#   Copilot -> LiteLLM -> vLLM MLX

model_list:
  - model_name: {model_name}
    litellm_params:
      model: openai/{model}
      api_base: http://localhost:{backend_port}/v1
      api_key: none
      max_tokens: 20000
      temperature: 0.7
      extra_body:
        chat_template_kwargs:
          enable_thinking: false

  - model_name: {model_alias}
    litellm_params:
      model: openai/{model}
      api_base: http://localhost:{backend_port}/v1
      api_key: none
      max_tokens: 20000
      temperature: 0.7
      extra_body:
        chat_template_kwargs:
          enable_thinking: false

litellm_settings:
  drop_params: true
  request_timeout: 300
  set_verbose: false
  system_prompt: |
    You are a helpful, concise coding assistant.
    Match response length to the request: trivial questions get trivial answers.
    Do not repeat reasoning or restart from scratch.
    Do not emit Error:/Exception:/Warning: labels unless there is a genuine error.
    When the user asks to update, edit, create, or fix a file, use the available
    file edit tool after minimal reading. Do not only describe the change.
    If the target file is already attached or named, read only the needed context
    and then perform the edit.
    For large file edits, never rewrite the whole file unless explicitly required.
    For large repository changes, first make a compact numbered todo plan, then
    implement one focused edit at a time.
    Prefer focused edits to functions/classes/sections and split large changes into
    several small tool calls. Keep each edit payload compact and valid.
    Do not create or update a TODO file unless the user explicitly asks for one.
    Stop as soon as the task is done.

general_settings:
  completion_model: "{model_name}"
"""

if not path.exists():
    path.write_text(render_config())
    print(f"Created {path}")
    raise SystemExit(0)

text = path.read_text()

if text.startswith("# Generated by setup-mlx.sh."):
    new_text = render_config()
    if new_text != text:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        path.write_text(new_text)
        print(f"Updated {path}")
        print(f"Backup:  {backup}")
    else:
        print(f"{path} already points at {model}")
    raise SystemExit(0)

updated = []
changed = False

for line in text.splitlines():
    stripped = line.lstrip()
    if stripped.startswith("model: openai/"):
        indent = line[: len(line) - len(stripped)]
        updated.append(f"{indent}model: openai/{model}")
        changed = True
    elif stripped.startswith("api_base: http://localhost:"):
        indent = line[: len(line) - len(stripped)]
        updated.append(f"{indent}api_base: http://localhost:{backend_port}/v1")
        changed = True
    elif stripped.startswith("max_tokens:"):
        indent = line[: len(line) - len(stripped)]
        updated.append(f"{indent}max_tokens: 20000")
        changed = True
    else:
        updated.append(line)

new_text = "\n".join(updated) + "\n"

if changed and new_text != text:
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    path.write_text(new_text)
    print(f"Updated {path}")
    print(f"Backup:  {backup}")
elif changed:
    print(f"{path} already points at {model}")
else:
    print(f"No generated or openai/ model entries found in {path}; left unchanged.")
PY
}

find_hf_cli() {
    if [ -x "$VENV_DIR/bin/hf" ]; then
        echo "$VENV_DIR/bin/hf"
        return 0
    fi

    if [ -x "$VENV_DIR/bin/huggingface-cli" ]; then
        echo "$VENV_DIR/bin/huggingface-cli"
        return 0
    fi

    if command -v hf >/dev/null 2>&1; then
        command -v hf
        return 0
    fi

    if command -v huggingface-cli >/dev/null 2>&1; then
        command -v huggingface-cli
        return 0
    fi

    return 1
}

show_download_info() {
    say "Model download information"

    local hf_login_cmd="$VENV_DIR/bin/hf"
    if [ ! -x "$hf_login_cmd" ] && find_hf_cli >/dev/null 2>&1; then
        hf_login_cmd="$(find_hf_cli)"
    fi

    echo "Model:      $MODEL"
    echo "HF_HOME:    $HF_HOME"
    echo "Cache path: $HF_HOME/hub"
    echo ""
    echo "How downloading works:"
    echo "  - vllm-mlx downloads missing Hugging Face model files on first start."
    echo "  - To download before starting the server, run:"
    echo "      $SETUP_COMMAND --download-model"
    echo "  - For private or gated repos, authenticate first:"
    echo "      $hf_login_cmd auth login"
    echo "    or run with HF_TOKEN in the environment:"
    echo "      HF_TOKEN=hf_xxx $SETUP_COMMAND --download-model"
    echo "  - This MLX model is large; keep tens of GB free in the HF cache."
}

download_model() {
    if [ "$DOWNLOAD_MODEL" = "ask" ]; then
        if [ -t 0 ]; then
            echo ""
            read -r -p "Download $MODEL now? [y/N] " reply
            case "$reply" in
                y|Y|yes|YES) DOWNLOAD_MODEL="yes" ;;
                *) DOWNLOAD_MODEL="no" ;;
            esac
        else
            DOWNLOAD_MODEL="no"
        fi
    fi

    if [ "$DOWNLOAD_MODEL" != "yes" ]; then
        echo ""
        echo "Skipping model download. First $START_COMMAND backend run will download if needed."
        return
    fi

    say "Downloading model into Hugging Face cache"

    local hf_cli
    hf_cli="$(find_hf_cli)" || {
        echo "ERROR: Hugging Face CLI not found."
        echo "Run without --no-install, or install huggingface_hub[cli]."
        exit 1
    }

    mkdir -p "$HF_HOME"
    export HF_HOME
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

    echo "CLI:   $hf_cli"
    echo "Model: $MODEL"
    echo "Cache: $HF_HOME"

    "$hf_cli" download "$MODEL"
}

print_next_steps() {
    say "Setup complete"

    echo "Start the full stack:"
    if [ "$ENV_FILE" = "$SCRIPT_DIR/.mlx.env" ] || [ "$START_COMMAND" != "./mlx-start.sh" ]; then
        echo "  $START_COMMAND start"
    else
        echo "  MLX_ENV_FILE=\"$ENV_FILE\" $START_COMMAND start"
    fi
    echo ""
    echo "Or only start the MLX backend:"
    if [ "$ENV_FILE" = "$SCRIPT_DIR/.mlx.env" ] || [ "$START_COMMAND" != "./mlx-start.sh" ]; then
        echo "  $START_COMMAND backend"
    else
        echo "  MLX_ENV_FILE=\"$ENV_FILE\" $START_COMMAND backend"
    fi
    echo ""
    echo "Check status and logs:"
    echo "  $START_COMMAND status"
    echo "  $START_COMMAND logs"
}

check_platform
check_python_version
setup_venv
install_dependencies
resolve_runtime_paths
write_env_file
update_litellm_config
show_download_info
download_model
print_next_steps
