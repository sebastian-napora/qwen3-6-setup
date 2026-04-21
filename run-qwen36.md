# Running Qwen3.6-35B-NVFP4 with Auto-Compression

This guide covers the complete setup for serving `RedHatAI/Qwen3.6-35B-A3B-NVFP4`
with automatic context compression at 50K tokens.

---

## Architecture

```
VS Code Copilot → LiteLLM (11111) → vLLM (11112)
                                      ↑
                               /compress @ 11112
```

- **Port 11111** — LiteLLM proxy (public API, Copilot connects here)
- **Port 11112** — vLLM backend (internal, serves model + `/compress` endpoint)

---

## Terminal 1 — Start vLLM backend

```bash
cd /home/sna/ai-projects/lunch-model
python3 qwen3_6_server.py
```

Wait for `INFO:     Uvicorn running on ...`. First load takes a few minutes.

---

## Terminal 2 — Start LiteLLM proxy with auto-compression

```bash
cd /home/sna/ai-projects/lunch-model
python3 server_compress.py
```

LiteLLM listens on **11111**. Copilot connects here. When a conversation exceeds
~50K tokens, LiteLLM's callback intercepts the request, calls `/compress` at 11112,
and forwards the compressed context to vLLM.

---

## Verify

```bash
# vLLM backend health
curl http://localhost:11112/health

# LiteLLM proxy health
curl http://localhost:11111/health

# Test completion via LiteLLM
curl -s -X POST http://localhost:11111/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-nvfp4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Test /compress Directly

```bash
curl -s -X POST http://localhost:11112/compress \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is Python?"},
      {"role": "assistant", "content": "Python is a programming language."},
      {"role": "user", "content": "What is a decorator?"}
    ],
    "target_tokens": 2048
  }'
```

---

## VS Code Copilot

Update `~/.copilot/config.json`:

```json
{
  "allowed_urls": ["http://localhost:11111"],
  "overrideEndpoint": "http://localhost:11111/v1"
}
```

Restart VS Code or the Copilot extension. All Copilot requests go through LiteLLM
on 11111, which forwards to vLLM on 11112. Long conversations (>50K tokens) are
auto-compressed.

---

## Auto-Compression Settings

Adjust via environment variables before starting `server_compress.py`:

| Variable | Default | Description |
|---|---|---|
| `LITE_LLM_COMPRESS_THRESHOLD_TOKENS` | `50000` | Trigger compression above this many input tokens |
| `LITE_LLM_COMPRESS_TARGET_TOKENS` | `16384` | Target token count for compressed output |
| `LITE_LLM_COMPRESS_MODELS` | `qwen3.6-35b-nvfp4` | Models that trigger compression (comma-separated) |
| `LITE_LLM_COMPRESS_PRESERVE_RECENT` | `5` | Always keep this many recent messages verbatim |

Example:
```bash
LITE_LLM_COMPRESS_THRESHOLD_TOKENS=30000 python3 server_compress.py
```

---

## Troubleshooting

**`vllm not found`** — Ensure vLLM is installed in the venv:
```bash
cd /home/sna/ai-projects/lunch-model/venv/bin
./pip install vllm
```

**Timeout on first load** — Wait longer. First inference after cold start
can take 1-2 minutes.

**Compression not triggering** — Short conversations (<10 messages) are never
compressed. The callback only acts when total input tokens exceed the threshold.

**Copilot not connecting** — Check `~/.copilot/config.json` and restart VS Code.

---

## File Map

| File | Purpose |
|---|---|
| `qwen3_6_server.py` | vLLM backend on port 11112 |
| `server_compress.py` | LiteLLM proxy on port 11111 with compression callback |
| `qwen36_compress.py` | LiteLLM CustomLogger — token counting + /compress calls |
| `lite_llm_config.yaml` | LiteLLM model routing config |
