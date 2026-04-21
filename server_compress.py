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
from pathlib import Path

import litellm

import qwen36_compress  # noqa: F401 — must import before register()
qwen36_compress.register()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("server_compress")

LITELLM_PORT = os.environ.get("LITE_LLM_PROXY_PORT", "11111")
LITELLM_HOST = os.environ.get("LITE_LLM_PROXY_HOST", "0.0.0.0")
CONFIG_PATH = Path(__file__).parent / "lite_llm_config.yaml"

logger.info("Starting LiteLLM proxy on %s:%s", LITELLM_HOST, LITELLM_PORT)
logger.info("Config: %s", CONFIG_PATH)
logger.info(
    "Auto-compression: threshold=%s tokens, target=%s tokens",
    os.environ.get("LITE_LLM_COMPRESS_THRESHOLD_TOKENS", "50000"),
    os.environ.get("LITE_LLM_COMPRESS_TARGET_TOKENS", "16384"),
)

os.environ["CONFIG_FILE_PATH"] = str(CONFIG_PATH)
os.environ.pop("LITELLM_MASTER_KEY", None)
os.environ.pop("LITELLM_SALT_KEY", None)

os.execvpe(
    sys.executable,
    [
        sys.executable,
        "-m", "uvicorn",
        "litellm.proxy.proxy_server:app",
        "--host", LITELLM_HOST,
        "--port", LITELLM_PORT,
    ],
    os.environ.copy(),
)
