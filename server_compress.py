#!/usr/bin/env python3
"""
LiteLLM proxy with Qwen3.6-35B auto-compression.

Architecture:
    Copilot → LiteLLM (11111) → vLLM (11112)
                                         ↑
                                  /compress @ 11112

Usage:
    # Terminal 1: start vLLM backend on 11112
    python3 qwen3_6_server.py

    # Terminal 2: start LiteLLM proxy on 11111
    python3 server_compress.py
"""

import os
import sys
import logging
import asyncio
import threading
import time
from pathlib import Path

import litellm

_io_logger = logging.getLogger("proxy_io")
_io_logger.setLevel(logging.INFO)
_io_logger.handlers.clear()
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter("%(message)s"))
_io_logger.addHandler(_ch)

# Custom I/O logger — logs every request + response to console
class ProxyIOLogger(litellm.integrations.custom_logger.CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        stream = kwargs.get("stream")
        msg_preview = self._summarize(messages)
        _io_logger.info(f"▶ REQUEST  model={model}  temp={temperature}  max_t={max_tokens}  stream={stream}  msgs={len(messages)}")
        if msg_preview:
            _io_logger.info(f"  {msg_preview}")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            choices = kwargs.get("messages", [])
            content = ""
            try:
                resp = dict(response_obj) if hasattr(response_obj, "__iter__") else {}
            except Exception:
                resp = {}
            choices = resp.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "") or str(msg)[:300]
            if not content:
                content = str(response_obj)[:300]
        except Exception:
            content = str(response_obj)[:300]
        _io_logger.info(f"◀ RESPONSE {content[:400]}")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        _io_logger.error(f"✗ FAILURE  {str(response_obj)[:300]}")

    def _summarize(self, messages):
        parts = []
        for m in messages[-3:]:
            role = m.get("role", "?")
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(p.get("text", "") or str(p) for p in content if isinstance(p, dict))
            if len(content) > 120:
                content = content[:120] + "..."
            parts.append(f"[{role}] {content}")
        return " | ".join(parts)

_proxy_io = ProxyIOLogger()
litellm.callbacks.append(_proxy_io)

# Must import + register BEFORE the proxy starts processing requests.
# uvicorn imports the app as a module-level object, so callbacks must be
# registered before that import happens.  Importing qwen36_compress here
# (before the uvicorn import chain) ensures the singleton is created in the
# same process that will handle requests.
# Compression disabled for diagnosis — uncomment to re-enable
# import qwen36_compress  # noqa: F401 — must import before register()
# qwen36_compress.register()

# Todo/Approval/Summary prompt injection — always enabled for lunch-model
import qwen36_compress
qwen36_compress.register_todo_callback()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)-25s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("server_compress")
_proxy_logger = logging.getLogger("proxy_io")
_proxy_logger.setLevel(logging.INFO)
_proxy_logger.handlers.clear()
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter("%(message)s"))
_proxy_logger.addHandler(_ch)

LITELLM_PORT = os.environ.get("LITE_LLM_PROXY_PORT", "11111")
LITELLM_HOST = os.environ.get("LITE_LLM_PROXY_HOST", "0.0.0.0")
CONFIG_PATH = Path(__file__).parent / "lite_llm_config.yaml"
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.info("Starting LiteLLM proxy on %s:%s", LITELLM_HOST, LITELLM_PORT)
logger.info("Config: %s", CONFIG_PATH)
logger.info(
    "Auto-compression: threshold=%s tokens, target=%s tokens",
    os.environ.get("LITE_LLM_COMPRESS_THRESHOLD_TOKENS", "99999999999"),
    os.environ.get("LITE_LLM_COMPRESS_TARGET_TOKENS", "16384"),
)

# ── Configure LiteLLM ────────────────────────────────────────────────────────
os.environ["CONFIG_FILE_PATH"] = str(CONFIG_PATH)
os.environ.pop("LITELLM_MASTER_KEY", None)
os.environ.pop("LITELLM_SALT_KEY", None)

# LiteLLM logs every request by default — direct it to our log file
os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("LITELLM_LOG_FILE", str(LOG_DIR / "litellm_proxy.log"))
os.environ.setdefault("LITELLM_LOG_LEVEL", "ERROR")

# ── Health-check loop (logs every 30s so we know the proxy is alive) ─────────
def _health_loop():
    while True:
        time.sleep(30)
        logger.debug("proxy heartbeat — callbacks=%d", len(litellm.callbacks))

_heartbeat = threading.Thread(target=_health_loop, daemon=True)
_heartbeat.start()

# ── Launch uvicorn in-process ────────────────────────────────────────────────
# This replaces os.execvpe so that registered callbacks survive.
# We still use uvicorn's low-level serve() API to mirror CLI behaviour.
if __name__ == "__main__":
    import uvicorn
    from litellm.proxy.proxy_server import app as litellm_app

    # Health-check log so we can confirm this is the right process
    logger.info(
        "LiteLLM callbacks active: %s",
        [type(cb).__name__ for cb in litellm.callbacks],
    )

    uvicorn.run(
        litellm_app,
        host=LITELLM_HOST,
        port=int(LITELLM_PORT),
        log_level="info",
        access_log=False,
    )
