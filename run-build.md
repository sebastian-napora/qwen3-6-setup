# Running the build on a target device

This document describes how to deploy and run `lunch-model` on a new device.

## Transfer

Copy `lunch-model-X.X.X.tar.gz` to the target device, then extract:

```bash
tar -xzf lunch-model-X.X.X.tar.gz
cd bundle
```

## Install

Run the install script:

```bash
./install.sh
```

This will:
1. Check Python version (requires 3.12+)
2. Check for NVIDIA GPU via `nvidia-smi`
3. Create a Python virtual environment (`venv/`)
4. Install the `lunch-model` wheel
5. Install runtime dependencies: vllm, litellm, fastapi, uvicorn, aiohttp

## Configure

Edit `lite_llm_config.yaml` if needed. Default points to:
- LiteLLM proxy: `http://localhost:11111`
- vLLM backend: `http://localhost:11112`

## Start services

```bash
./start.sh
```

This starts all three services:
- vLLM backend on port **11112**
- LiteLLM proxy on port **11111**
- Token stats server on port **11113**

## Stop services

```bash
./kill.sh
```

## Direct Python execution

If you prefer to run Python directly without the install script:

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Install package
pip install lunch_model-*.whl

# Or install from PyPI (if published)
pip install lunch-model

# Install runtime deps
pip install vllm litellm fastapi uvicorn aiohttp

# Run manually
python3 qwen3_6_server.py
python3 server_compress.py
python3 qwen_token_stats_server.py
```

## Architecture

```
Copilot → LiteLLM (11111) → vLLM (11112)
                              ↕
                        Token stats (11113)
```

## Requirements

- NVIDIA GPU with CUDA drivers
- Python 3.12+
- 128GB+ system RAM recommended for Qwen3.6-35B-A3B-NVFP4

## Linux (NVIDIA)

vllm has pre-built wheels for Linux on NVIDIA GPUs. The install script handles this automatically.

```bash
./install.sh
```

## macOS / No GPU

vllm **requires Linux + NVIDIA GPU**. On macOS or systems without NVIDIA:

1. Run install.sh (vllm will be skipped)
2. Use Docker for vllm instead:

```bash
docker run -d \
  --name vllm \
  -p 11112:8000 \
  --runtime nvidia \
  --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.6-35B-A3B-NVFP4
```

Or use a cloud GPU instance (Lambda, RunPod, etc.) as the backend, and point `lite_llm_config.yaml` to that remote URL.

## Custom model path

If the model is not in the default HuggingFace cache, set the path in `qwen3_6_server.py`:

```python
llm = LLM(
    model="path/to/your/model",
    ...
)
```