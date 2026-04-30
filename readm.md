# Qwen3.6-35B-NVFP4 vLLM Backend

This is the **vLLM backend** for `RedHatAI/Qwen3.6-35B-A3B-NVFP4`. It serves the model on port `11112` with an LLM-powered `/compress` endpoint.

**This server is not called directly by clients.** It is started manually (see Quick Start) and sits behind the LiteLLM proxy which VS Code Copilot talks to.

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
| `/v1/local/embeddings` | POST | Local Qwen3 embedding endpoint |
| `/v1/local_rag/ingest` | POST | Ingest local documents into SQLite vector storage |
| `/v1/local_rag/search` | POST | Search ingested local document chunks |
| `/v1/local_rag/query` | POST | Retrieve chunks and answer with Qwen via LiteLLM |

## Local RAG

The LiteLLM proxy installs local RAG endpoints on port `11111`. Embeddings are
generated with MLX and default to:

```
mlx-community/Qwen3-Embedding-8B-4bit-DWQ
```

The embedding model is loaded lazily on the first embedding, ingest, search, or
query request. If the model is not cached yet, the first request downloads it
from Hugging Face.

Embedding model choices:

| Preset | Model | Backend | Good for |
|---|---|---|---|
| `mlx-8b` | `mlx-community/Qwen3-Embedding-8B-4bit-DWQ` | `mlx` | Apple Silicon / Metal |
| `unsloth-4b` | `unsloth/Qwen3-Embedding-4B` | `hf` | DGX Spark / NVIDIA |
| `qwen-4b` | `Qwen/Qwen3-Embedding-4B` | `hf` | DGX Spark / NVIDIA |
| `qwen-0.6b` | `Qwen/Qwen3-Embedding-0.6B` | `hf` | smaller CPU/GPU tests |

To pre-download it during machine setup:

```bash
./download_embedding_model.sh
```

To also verify the selected local embedding backend can execute the model:

```bash
./download_embedding_model.sh --verify
```

For DGX Spark / NVIDIA, install the Torch/Transformers embedding backend and
select the Unsloth 4B model:

```bash
./install.sh --hf-embeddings
./download_embedding_model.sh --preset unsloth-4b --verify
QWEN_RAG_EMBED_MODEL=unsloth/Qwen3-Embedding-4B ./start.sh proxy
```

You can also select it per request:

```json
{
  "embedding_model": "unsloth/Qwen3-Embedding-4B",
  "embedding_backend": "hf"
}
```

By default path ingestion is limited to `./documents` for safety because the
proxy binds to `0.0.0.0`. Override with:

```bash
export QWEN_RAG_DOCUMENT_ROOT=/path/to/documents
# or, only on trusted machines:
export QWEN_RAG_ALLOW_OUTSIDE_ROOT=1
```

Ingest a folder:

```bash
curl -X POST http://localhost:11111/v1/local_rag/ingest \
  -H 'Content-Type: application/json' \
  -d '{"path": ".", "collection": "default"}'
```

Search:

```bash
curl -X POST http://localhost:11111/v1/local_rag/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "What does this project expose?", "collection": "default", "top_k": 5}'
```

Ask with retrieved context:

```bash
curl -X POST http://localhost:11111/v1/local_rag/query \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-nvfp4",
    "collection": "default",
    "messages": [{"role": "user", "content": "What endpoints are available?"}]
  }'
```

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
