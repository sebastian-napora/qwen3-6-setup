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
        import uuid
        req_id = kwargs.get("request_id") or str(uuid.uuid4())[:8]
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        stream = kwargs.get("stream")
        msg_count = len(messages)
        msg_preview = self._summarize(messages)
        _io_logger.info(f"▶ REQUEST  req={req_id}  model={model}  temp={temperature}  max_t={max_tokens}  stream={stream}  msgs={msg_count}")
        if msg_preview:
            _io_logger.info(f"  {msg_preview}")
        # Per-call kwargs at debug level — was WARNING which drowned real errors.
        _io_logger.debug(f"  [PRE_CALL] req_id={req_id}  kwargs={ {k: v for k, v in kwargs.items() if k not in ('messages',)} }")

    def log_pre_call(self, model, messages, kwargs):
        import uuid
        req_id = kwargs.get("request_id") or str(uuid.uuid4())[:8]
        msg_count = len(messages) if messages else 0
        _io_logger.info(f"▶ PRE_CALL  req={req_id}  model={model}  msgs={msg_count}  kwargs={ {k: v for k, v in kwargs.items() if k not in ('messages',)} }")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        req_id = kwargs.get("request_id") or "-"
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
        duration_ms = round((end_time - start_time) * 1000) if start_time and end_time else 0
        _io_logger.info(f"◀ RESPONSE req={req_id}  dur={duration_ms}ms  {content[:400]}")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        import traceback
        req_id = kwargs.get("request_id") or "-"
        error_type = type(response_obj).__name__ if response_obj else "unknown"
        error_msg = str(response_obj)[:1000] if response_obj else "no error object"
        duration_ms = round((end_time - start_time) * 1000) if start_time and end_time else 0
        tb = traceback.format_stack()[-4:-1]
        _io_logger.error(f"✗ FAILURE  req={req_id}  dur={duration_ms}ms  type={error_type}  error={error_msg}")
        _io_logger.error(f"  [CALL STACK] {' | '.join(tb)}")

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
# Wire the same logger into success/failure channels so failures actually
# reach log_failure_event (otherwise crashes vanish silently).
litellm.success_callback.append(_proxy_io)
litellm.failure_callback.append(_proxy_io)

# Must import + register BEFORE the proxy starts processing requests.
# uvicorn imports the app as a module-level object, so callbacks must be
# registered before that import happens.  Importing qwen36_compress here
# (before the uvicorn import chain) ensures the singleton is created in the
# same process that will handle requests.
# Compression disabled for diagnosis — uncomment to re-enable
# import qwen36_compress  # noqa: F401 — must import before register()
# qwen36_compress.register()

# Todo/Approval prompt injection disabled — removed, was injecting conflicting
# system messages and forcing TODO-list behaviour on every Copilot request.
# import qwen36_compress
# qwen36_compress.register_todo_callback()

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
