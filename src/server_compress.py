#!/usr/bin/env python3
"""
LiteLLM proxy for Qwen3.6-35B-A3B-NVFP4.

Architecture:
    Copilot → LiteLLM (11111) → vLLM (11112)

Usage:
    # Terminal 1: start vLLM backend on 11112
    python3 qwen3_6_server.py

    # Terminal 2: start LiteLLM proxy on 11111
    python3 server_compress.py
"""

import os
import logging
import asyncio
import threading
import time
from pathlib import Path
import sys
import litellm

from src import qwen_compress  # noqa: F401 — strips thinking tokens from history
from src import qwen_token_tracker  # noqa: F401 — records per-request token usage
qwen_compress.register()
qwen_token_tracker.register()

ROOT_DIR = Path(__file__).resolve().parent.parent

# Setup detailed logging
LOG_DIR = Path(os.environ.get("QWEN_LOG_DIR", ROOT_DIR / "logs")).expanduser()
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_LEVEL_NAME = os.environ.get("LITELLM_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)

litellm_logger = logging.getLogger("litellm.image_request")
litellm_logger.setLevel(LOG_LEVEL)
fh = logging.FileHandler(os.path.join(LOG_DIR, "litellm_image_requests.log"))
fh.setLevel(LOG_LEVEL)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
litellm_logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(LOG_LEVEL)
ch.setFormatter(logging.Formatter("%(asctime)s %(name)-25s %(levelname)-8s %(message)s"))
litellm_logger.addHandler(ch)

litellm_logger.info("=" * 60)
litellm_logger.info("LiteLLM Qwen3.6-35B-A3B Proxy Started")

os.environ.setdefault("LITELLM_LOG", LOG_LEVEL_NAME)
os.environ.setdefault("LITELLM_REQUEST_LOGGING", "false")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(name)-25s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "litellm_detailed.log")),
        logging.StreamHandler(),
    ],
)

litellm_main_logger = logging.getLogger("litellm")
litellm_main_logger.setLevel(LOG_LEVEL)

logger = logging.getLogger("server_compress")
_proxy_logger = logging.getLogger("proxy_io")
_proxy_logger.setLevel(logging.INFO)
_proxy_logger.handlers.clear()
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter("%(message)s"))
_proxy_logger.addHandler(_ch)

LITELLM_PORT = os.environ.get("LITE_LLM_PROXY_PORT", "11115")
LITELLM_HOST = os.environ.get("LITE_LLM_PROXY_HOST", "0.0.0.0")
CONFIG_PATH = Path(
    os.environ.get("CONFIG_FILE_PATH", ROOT_DIR / "lite_llm_config.yaml")
).expanduser()
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Starting LiteLLM proxy on %s:%s", LITELLM_HOST, LITELLM_PORT)
logger.info("Config: %s", CONFIG_PATH)
logger.info("Assistant history sanitization enabled")

os.environ.pop("LITELLM_MASTER_KEY", None)
os.environ.pop("LITELLM_SALT_KEY", None)
os.environ["CONFIG_FILE_PATH"] = str(CONFIG_PATH)

registered_callbacks = [cb for cb in litellm.callbacks if hasattr(cb, "__class__")]
logger.info("Registered custom callbacks: %d", len(registered_callbacks))
for cb in registered_callbacks:
    logger.info("  - %s", type(cb).__name__)

def main():
    import uvicorn
    uvicorn.run(
        "litellm.proxy.proxy_server:app",
        host=LITELLM_HOST,
        port=int(LITELLM_PORT),
        reload=False,
        log_level=os.environ.get("UVICORN_LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
