# Qwen3.6-35B-NVFP4 vLLM Backend

This is the **vLLM backend** for `RedHatAI/Qwen3.6-35B-A3B-NVFP4`. It serves the model on port `11112` with an LLM-powered `/compress` endpoint.

**This server is not called directly by clients.** It is started by LiteLLM's `comstart` command and hosts the `/compress` endpoint used by the auto-compression middleware.

## Architecture

```
Copilot → LiteLLM (11111) → vLLM qwen3.6 (11112)
                                   ↑
                            /compress @ 11112
```

## Quick Start

```bash
# Option A: via LiteLLM proxy (recommended)
cd /home/sna/ai-projects/lite-llm
uv run main.py comstart       # starts vLLM (11112) + LiteLLM proxy with compression (11111)

# Option B: standalone (for testing /compress directly)
cd /home/sna/ai-projects/lunch-model
python3 qwen3-6-server.py
```

## Standalone Server Flags

```bash
python3 qwen3-6-server.py \
  --model RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
  --port 11112 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --quantization fp4
```

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Standard OpenAI-compatible chat API |
| `/health` | GET | Health check |
| `/compress` | POST | LLM-powered context compression |
| `/compress/stream` | POST | Streaming compression |

## /compress

Compresses long conversation histories to save context tokens.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "target_tokens": 16384
}
```

**Response:**
```json
{
  "compressed": {
    "summary": "...",
    "preserved_messages": [...],
    "token_budget_used": 0.42
  },
  "original_message_count": 50,
  "target_tokens": 16384
}
```

## Packaging & Installation

### 1. Editable install (recommended for development)

```bash
pip install -e /home/sna/ai-projects/lunch-model
qwen-server
```

### 2. Build a distributable wheel

```bash
cd /home/sna/ai-projects/lunch-model
python3 -m pip install build --upgrade
python3 -m build
pip install dist/*.whl
qwen-server
```

### 3. Docker image

```bash
cd /home/sna/ai-projects/lunch-model
docker build -t qwen-server .
docker run --gpus all -p 11112:11112 qwen-server
```

### 4. PyInstaller standalone binary

```bash
cd /home/sna/ai-projects/lunch-model
pip install pyinstaller
pyinstaller qwen3-6-server.py --name qwen-server-bin --onefile
./dist/qwen-server-bin
```
