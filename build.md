# Build Guide

This guide covers how to build and deploy the `lunch-model` package.

---

## Prerequisites

- **Python 3.12+**

> The build script automatically creates a `venv/` if one does not exist, so no manual venv setup is needed for building.

---

## Project Structure

```
qwen3-6-setup/
├── src/                          # Python source package
│   ├── __init__.py
│   ├── qwen3_6_server.py         # vLLM backend entry point
│   ├── server_compress.py        # LiteLLM proxy entry point
│   ├── qwen_token_stats_server.py # Token stats server entry point
│   ├── qwen_token_tracker.py     # Token tracking
│   ├── qwen36_compress.py        # History compression
│   ├── qwen_compress.py          # History sanitization
│   └── request_logging.py        # HTTP request logging
├── start.sh                      # Starts all services
├── kill.sh                       # Stops all services
├── lite_llm_config.yaml          # LiteLLM configuration
├── pyproject.toml                # Package metadata
├── MANIFEST.in                   # Build manifest
└── build.sh                      # Build script
```

---

## Step 1 — Build

Run from the project root directory:

```bash
./build.sh
```

This script performs 4 steps:

1. **[0/4] Setup** — creates `venv/` if missing, installs `build` package
2. **[1/4] Clean** — removes old `build/`, `dist/`, and `*.egg-info/` directories
3. **[2/4] Build wheel** — creates a Python wheel via `python3 -m build --wheel`
4. **[3/4] Copy assets** — copies config files, scripts, and `src/` into `dist/bundle/`
5. **[4/4] Create tarball** — packages everything into `dist/lunch-model-X.X.X.tar.gz`

**Output files:**

| File | Description |
|------|-------------|
| `dist/lunch_model-X.X.X-py3-none-any.whl` | Python wheel |
| `dist/lunch-model-X.X.X.tar.gz` | Deployable tarball (use this) |

---

## Step 3 — Deploy

Copy the tarball to the target machine:

```bash
scp dist/lunch-model-0.1.0.tar.gz user@target:/path/to/deploy/
```

On the target device, extract the tarball and enter the bundle directory:

```bash
cd /path/to/deploy
tar -xzf lunch-model-0.1.0.tar.gz
cd bundle/
```

Then run the deploy helper script:

```bash
./deploy.sh        # interactive (asks for confirmation)
./deploy.sh --yes  # non-interactive (skip prompts)
```

The deploy script will:

- Detect if already installed (has `venv/`)
- Show detected files and what will be installed
- Ask for confirmation (unless `--yes` is passed)
- Run `./install.sh` automatically

---

## Step 4 — Run

```bash
./start.sh
```

This starts all three services:

| Service | Port | Command entry |
|---------|------|---------------|
| vLLM backend | 11114 | `qwen-server` |
| Token stats | 11116 | `qwen-stats` |
| LiteLLM proxy | 11115 | `qwen-compress` |

To start individual services:

```bash
./start.sh backend  # vLLM only
./start.sh proxy    # LiteLLM proxy only
./start.sh stats    # Token stats only
```

---

## Step 5 — Stop

```bash
./kill.sh
```

This kills processes by name and frees ports 11114, 11115, 11116.

---

## Entry Points (installed via pip)

After running `pip install lunch_model-*.whl`, these commands become available system-wide:

| Command | Invokes | Purpose |
|---------|---------|---------|
| `qwen-server` | `src.qwen3_6_server:run` | vLLM backend (port 11114) |
| `qwen-compress` | `src.server_compress:main` | LiteLLM proxy (port 11115) |
| `qwen-stats` | `src.qwen_token_stats_server:main` | Token stats web UI (port 11116) |

---

## Rebuild During Development

If you modify source files in `src/`, rebuild the package:

```bash
./build.sh
```

Then reinstall in a fresh venv to test:

```bash
python3 -m venv test_venv
source test_venv/bin/activate
pip install dist/lunch_model-0.1.0-py3-none-any.whl
```

---

## Local Development Setup

To run the services locally (without building), you need a venv with all dependencies.

### Create and activate venv

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install --upgrade pip
pip install litellm fastapi uvicorn aiohttp uvloop websockets backoff \
    python-multipart orjson apscheduler pydantic-settings pyjwt \
    cryptography boto3 "email-validator[pyyaml]" fastapi-sso
```

> **Note:** `vllm` cannot be installed on macOS via pip. On macOS, skip it and run vLLM via Docker (see [vLLM on macOS](#vllm-on-macos) below).

### Run services

```bash
./start.sh
```

### Delete venv

```bash
deactivate
rm -rf venv
```

### vLLM on macOS

Since `vllm` is Linux-only, on macOS you need to run it in Docker:

```bash
docker run -d --gpus all -p 11114:8000 \
  -v ~/.cache/huggingflow:/root/.cache/huggingflow \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-35B-A3B-NVFP4-TTS
```

This exposes the vLLM API on port 11114. Then start the remaining services:

```bash
./start.sh proxy    # LiteLLM proxy on 11115
./start.sh stats    # Token stats on 11116
```

---

## Wheel-Only Build (no tarball)

To build just the wheel without the full bundle:

```bash
python3 -m pip install build
python3 -m build --wheel --outdir dist/
```

Manually copy config files and `src/` as needed.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

The wheel was built with an empty `src/` directory. Make sure `src/` contains all `.py` files before building. Run:

```bash
ls src/*.py
```

If empty, the files are still in the project root — move them into `src/` as shown in the Project Structure above.

### `ModuleNotFoundError: No module named 'qwen3_6_server'`

Entry points reference the old root-level module names. Update `pyproject.toml`:

```toml
[project.scripts]
qwen-server = "src.qwen3_6_server:run"
qwen-compress = "src.server_compress:main"
qwen-stats = "src.qwen_token_stats_server:main"
```

### Slow first request on vLLM

The first request triggers model loading into GPU memory and may take 2–5 minutes. Subsequent requests are fast.
