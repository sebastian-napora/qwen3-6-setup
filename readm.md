# Qwen3.6 MLX vLLM Backend

This project serves **Qwen3.6** models on Apple Silicon via vLLM-MLX, with an LLM-powered `/compress` endpoint. Two model profiles are supported:

| Model | Hugging Face | Port | Config |
|---|---|---|---|
| **Qwen3.6-35B-A3B-4bit** (MoE) | `mlx-community/Qwen3.6-35B-A3B-4bit` | 11114 (backend) / 11115 (proxy) | `lite_llm_config.yaml` |
| **Qwen3.6-35B-A3B-UD-MLX-3bit** (MoE) | `unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit` | 11134 (backend) / 11135 (proxy) | `lite_llm_config_35b_a3b_3bit.yaml` |
| **Qwen3.6-27B-4bit** | `mlx-community/Qwen3.6-27B-4bit` | 11124 (backend) / 11125 (proxy) | `lite_llm_config_27b.yaml` |

**These servers are not called directly by clients.** They are started manually (see Quick Start) and sit behind the LiteLLM proxy which VS Code Copilot talks to.

---

## Fixes Applied (April 2026)

Five bugs that caused degraded output, `</think>` leaks, and random crashes were diagnosed and fixed:

### 1. Token degradation (random words / symbols / emojis)
**Cause:** Sampling params `temperature=1.0, presence_penalty=1.5, frequency_penalty=1.0` caused the model to devolve into a 1500-token word-stream that never produced a real answer.  
**Fix:** Changed to Qwen3-recommended values in both `qwen3_6_server.py` (`--override-generation-config`) and `lite_llm_config.yaml`:
```
temperature: 0.7 | top_p: 0.8 | top_k: 20 | min_p: 0.0
repetition_penalty: 1.05 | presence_penalty: 0 | frequency_penalty: 0
```

### 2. `</think>` tags leaking into chat output
**Cause:** `_THINK_PAT` regex in `qwen36_compress.py` used Qwen2 syntax `<|think|>` instead of Qwen3 syntax `<think>`. The stripper replaced the literal twice and did nothing useful.  
**Fix:** Rewrote `strip_thinking_blocks()` to correctly match `<think>...</think>` (DOTALL) and strip any orphan tags.

### 3. Reasoning appearing outside reasoning pane / over-thinking
**Cause:** `TodoApprovalPromptCallback.async_pre_call_hook` was duplicating system messages (one from config, one injected) and prepending a 454-token `FIRST_MSG_RULES` block to every "first" message — forcing the model into deep reasoning on trivial prompts.  
**Fix:** Rewrote the hook to augment the existing system message in place (no duplication). Gated `FIRST_MSG_RULES` injection behind env var `LITE_LLM_INJECT_FIRST_MSG_RULES` (default `false`).

### 4. Unknown stop token `<|done|>`
**Cause:** `<|done|>` in the stop list and system prompt — the model was never trained on this token and would spin endlessly waiting to emit it.  
**Fix:** Removed from `lite_llm_config.yaml` stops list and all prompt text. Standard stop tokens `<|im_end|>` and `<|endoftext|>` kept.

### 5. Silent crashes / no error info in logs
**Cause:** `litellm.failure_callback` was never wired; `ProxyIOLogger.log_pre_api_call` emitted a WARNING on every *successful* call drowning real errors; `RequestLoggingMiddleware` was defined but never installed.  
**Fix:** Registered `_proxy_io` on both `litellm.success_callback` and `litellm.failure_callback` in `server_compress.py`. Downgraded per-call kwargs log from WARNING → DEBUG.  
**Note:** `RequestLoggingMiddleware` (BaseHTTPMiddleware) was incompatible with vLLM's `listen_for_disconnect()` task — it corrupted non-streaming responses to `"null"`. It was intentionally removed; proxy-side callbacks cover logging instead.

---

## Architecture

```
Copilot → LiteLLM Proxy → vLLM MLX Backend
              ↑                ↑
   Compression callbacks    /compress endpoint
```

Two independent serving stacks run side by side:

```
Stack 35B (default):      Copilot → LiteLLM (11115) → vLLM MLX (11114)
Stack 35B-A3B-3bit:       Copilot → LiteLLM (11135) → vLLM MLX (11134)
Stack 27B:                Copilot → LiteLLM (11125) → vLLM MLX (11124)
```

### Ports

| Service | 35B-A3B | 35B-A3B-3bit | 27B |
|---|---|---|---|
| vLLM MLX backend | 11114 | 11134 | 11124 |
| LiteLLM proxy | 11115 | 11135 | 11125 |
| Token stats | 11116 | 11136 | 11126 |

## Quick Start

### Setup (one-time)

```bash
# 35B-A3B model
./setup-mlx.sh

# 35B-A3B-UD-MLX-3bit (unsloth variant)
./setup-mlx-35b-a3b-3bit.sh

# 27B model (separate env, logs, and PID files)
./setup-mlx-27b.sh
```

### Start a model

```bash
# 35B-A3B (default)
./mlx-start.sh          # start all (backend + stats + proxy)
./mlx-start.sh backend  # backend only
./mlx-start.sh proxy    # proxy only
./mlx-start.sh stop     # stop all

# 35B-A3B-UD-MLX-3bit (unsloth variant)
./mlx-start-35b-a3b-3bit.sh      # start all
./mlx-start-35b-a3b-3bit.sh stop # stop all

# 27B
./mlx-start-27b.sh      # start all
./mlx-start-27b.sh stop # stop all
```

### LiteLLM proxy (with compression)

```bash
# From a separate lite-llm project directory:
cd /home/sna/ai-projects/lite-llm
uv run main.py comstart       # starts vLLM + LiteLLM proxy with compression
```

## Standalone Server Flags

### 35B-A3B

```bash
python3 qwen3-6-server.py \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 11114 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --quantization fp4
```

### 27B

```bash
python3 qwen3-6-server.py \
  --model mlx-community/Qwen3.6-27B-4bit \
  --port 11124 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 20000 \
  --quantization fp4
```

> **Note:** The 27B model runs with `enable_thinking: false` in its LiteLLM config (no reasoning parser).

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
