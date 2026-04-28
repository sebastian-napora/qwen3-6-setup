#!/usr/bin/env python3
"""
Local vLLM API Server for Qwen3.6-35B-A3B-4bit on Apple Silicon / vLLM-Metal.

Endpoints:
  /v1/chat/completions     OpenAI-compatible chat API from vLLM
  /compress                LLM-powered context compression
  /compress/stream         Streaming context compression
  /health                  Simple health endpoint
  /v1/chat/image           Returns 501: image input not supported on this backend
  /v1/chat/image_base64    Returns 501: image input not supported on this backend

This version is tuned for the error you hit:
  - Paged attention OFF by default.
  - Prefix caching OFF by default.
  - Text-only compatibility mode ON.
  - max_model_len default lowered from 30000 to 28512.
  - Chunked prefill is NOT forcibly disabled.

Optional experimental overrides:
  QWEN_ALLOW_EXPERIMENTAL_PAGED_ATTENTION=1
  QWEN_ALLOW_EXPERIMENTAL_PREFIX_CACHE=1

Recommended run:
  source ~/.venv-vllm-metal/bin/activate
  python qwen3_6_server.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import traceback
from typing import Any


# =============================================================================
# Environment setup - must happen before importing vLLM / vllm-metal
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Prefer project-local ./venv when present. Harmless if absent.
_LOCAL_VENV_BIN = os.path.join(BASE_DIR, "venv", "bin")
_LOCAL_VENV_LIB = os.path.join(
    BASE_DIR,
    "venv",
    "lib",
    "python3.12",
    "site-packages",
)

if os.path.isdir(_LOCAL_VENV_BIN):
    current_path = os.environ.get("PATH", "")
    if _LOCAL_VENV_BIN not in current_path.split(os.pathsep):
        os.environ["PATH"] = _LOCAL_VENV_BIN + os.pathsep + current_path

if os.path.isdir(_LOCAL_VENV_LIB) and _LOCAL_VENV_LIB not in sys.path:
    sys.path.insert(0, _LOCAL_VENV_LIB)


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# Allow long max_model_len.
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

# Avoid your previous "unknown vLLM environment variable" warning.
os.environ.pop("VLLM_WORKER_LOGGING_LEVEL", None)

# Avoid slow/noisy Gloo hostname fallback on macOS. Override externally if needed.
os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")

# Qwen3.6 hybrid SDPA + GDN-linear/MoE on vLLM-Metal is sensitive to the paged
# KV path. Your original crash was an assertion in kv_cache_utils while paged
# attention/prefix-cache was active. Keep paged attention off by default.
_ALLOW_EXPERIMENTAL_PAGED_ATTENTION = _env_true(
    "QWEN_ALLOW_EXPERIMENTAL_PAGED_ATTENTION"
)
if _ALLOW_EXPERIMENTAL_PAGED_ATTENTION:
    os.environ.setdefault("VLLM_METAL_USE_PAGED_ATTENTION", "1")
    os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "auto")
else:
    os.environ["VLLM_METAL_USE_PAGED_ATTENTION"] = "0"
    os.environ["VLLM_METAL_MEMORY_FRACTION"] = "auto"

# Your logs show the model is detected as VLM-capable, but vLLM-Metal bypasses
# vision. Force the known-safe text-only path.
os.environ["VLLM_METAL_MULTIMODAL_MODE"] = "text-only-compat"

# Do not force prefix caching for this hybrid model unless explicitly requested.
_ALLOW_EXPERIMENTAL_PREFIX_CACHE = _env_true("QWEN_ALLOW_EXPERIMENTAL_PREFIX_CACHE")
if not _ALLOW_EXPERIMENTAL_PREFIX_CACHE:
    os.environ.pop("VLLM_METAL_PREFIX_CACHE", None)


# =============================================================================
# Logging
# =============================================================================

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

vllm_logger = logging.getLogger("vllm.qwen3_6_server")
vllm_logger.setLevel(logging.DEBUG)
vllm_logger.propagate = True

_log_file = os.path.join(LOG_DIR, "vllm_server_requests.log")
if not any(
    isinstance(handler, logging.FileHandler)
    and getattr(handler, "baseFilename", None) == _log_file
    for handler in vllm_logger.handlers
):
    file_handler = logging.FileHandler(_log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    )
    vllm_logger.addHandler(file_handler)

vllm_logger.info("=" * 80)
vllm_logger.info("Qwen3.6 vLLM-Metal server logger started")
vllm_logger.info(
    "VLLM_METAL_USE_PAGED_ATTENTION=%s",
    os.environ.get("VLLM_METAL_USE_PAGED_ATTENTION"),
)
vllm_logger.info(
    "VLLM_METAL_MEMORY_FRACTION=%s",
    os.environ.get("VLLM_METAL_MEMORY_FRACTION"),
)
vllm_logger.info(
    "VLLM_METAL_MULTIMODAL_MODE=%s",
    os.environ.get("VLLM_METAL_MULTIMODAL_MODE"),
)


# =============================================================================
# Configuration
# =============================================================================

MODEL = os.environ.get("QWEN_MODEL", "mlx-community/Qwen3.6-35B-A3B-4bit")
HOST = os.environ.get("QWEN_HOST", "0.0.0.0")
PORT = int(os.environ.get("QWEN_PORT", "11114"))

# Your latest log estimated max possible length as 29568, while 30000 failed.
# 28512 equals 27 * 1056 and leaves one 1056-token block of safety.
MAX_MODEL_LEN = int(os.environ.get("QWEN_MAX_MODEL_LEN", "28512"))

# Do not default generation to the full context window. Clients can request more.
DEFAULT_MAX_TOKENS = int(os.environ.get("QWEN_DEFAULT_MAX_TOKENS", "4096"))
DEFAULT_MAX_TOKENS = min(DEFAULT_MAX_TOKENS, MAX_MODEL_LEN)

MAX_NUM_SEQS = int(os.environ.get("QWEN_MAX_NUM_SEQS", "1"))
SEED = int(os.environ.get("QWEN_SEED", "5678"))

TOOL_CALL_PARSER = os.environ.get("QWEN_TOOL_CALL_PARSER", "qwen3_coder")
REASONING_PARSER = os.environ.get("QWEN_REASONING_PARSER", "qwen3")


# =============================================================================
# Compression prompt
# =============================================================================

COMPRESS_PROMPT = """You are a context compression assistant. Your task is to produce a **lossy but semantically faithful** summary of the conversation below.

## Rules
- Preserve ALL technical decisions, code snippets, file paths, and command outputs verbatim when possible.
- Preserve user preferences, constraints, and requirements.
- Preserve any errors, fixes, and their solutions.
- Summarize repetitive or redundant exchanges into concise bullet points.
- Preserve the most recent few turns in full detail (they contain the current context).
- Output ONLY a JSON object with this exact schema — no preamble, no explanation, no markdown:

{
  "summary": "A comprehensive but condensed summary of the entire conversation history. Include key facts, decisions, and current state.",
  "preserved_messages": [
    {"role": "user|assistant", "content": "Full verbatim content of the most recent N turns that must be kept verbatim."}
  ],
  "token_budget_used": 0.42
}

## Conversation to compress
"""


# =============================================================================
# Helpers
# =============================================================================

def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _shorten(value: Any, max_chars: int = 300) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def _extract_chat_content_and_reasoning(result: Any) -> tuple[str, str | None]:
    """
    Extract assistant content and optional reasoning from either dict-style or
    object-style vLLM OpenAI responses.
    """
    if isinstance(result, dict):
        choices = result.get("choices") or []
        if not choices:
            return "", None

        choice0 = choices[0]
        if isinstance(choice0, dict):
            message = choice0.get("message") or {}
            if isinstance(message, dict):
                return message.get("content") or "", message.get("reasoning")
            return str(message), None

        return str(choice0), None

    choices = getattr(result, "choices", None)
    if not choices:
        return str(result), None

    message = getattr(choices[0], "message", None)
    if message is None:
        return str(choices[0]), None

    content = getattr(message, "content", "") or ""
    reasoning = getattr(message, "reasoning", None)
    return content, reasoning


def _strip_json_fences(raw: str) -> str:
    raw = raw.strip()
    for fence in ("```json", "```JSON", "```"):
        if raw.startswith(fence):
            raw = raw[len(fence):].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    return raw


def _fallback_request_log(
    level: int,
    request_id: str,
    event: str,
    **kwargs: Any,
) -> None:
    payload = " ".join(f"{k}={_shorten(v, 160)}" for k, v in kwargs.items())
    vllm_logger.log(level, "[%s] %s %s", request_id, event, payload)


def _install_optional_request_logging(app: Any):
    """
    Use local request_logging.py if present; otherwise use a fallback logger.
    This does not install BaseHTTPMiddleware here.
    """
    try:
        from src import request_logging  # type: ignore

        install = getattr(request_logging, "install", None)
        if callable(install):
            install(app)

        custom_log = getattr(request_logging, "_log", None)
        if callable(custom_log):
            vllm_logger.info("Installed optional request_logging module")
            return custom_log

        vllm_logger.info("request_logging module found but _log missing; using fallback")
        return _fallback_request_log

    except Exception as exc:
        vllm_logger.info("Optional request_logging unavailable: %s", exc)
        return _fallback_request_log


def _patch_qwen3xml_tool_parser_warning() -> None:
    """
    Some vLLM builds have a noisy qwen3xml_tool_parser warning path where the
    log format args do not match. Silence only that known warning.
    """
    try:
        import vllm.tool_parsers.qwen3xml_tool_parser as qxt  # type: ignore

        original_warning = qxt.logger.warning

        def patched_warning(msg: Any, *args: Any, **kwargs: Any) -> None:
            if "is not a float in tool" in str(msg):
                return
            original_warning(msg, *args, **kwargs)

        qxt.logger.warning = patched_warning
        vllm_logger.info("Patched qwen3xml_tool_parser float warning")
    except Exception as exc:
        vllm_logger.info("Skipping qwen3xml_tool_parser warning patch: %s", exc)


def _build_generation_config() -> dict[str, Any]:
    """
    Qwen-style conservative defaults. Keep presence/frequency penalties at zero;
    your earlier note said non-zero penalties degraded output badly.
    """
    return {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "repetition_penalty": 1.05,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }


def _build_vllm_argv() -> list[str]:
    """
    Startup-safe argv for vLLM-Metal + Qwen3.6 hybrid model.

    Deliberately NOT included:
      --enable-prefix-caching
      --kv-cache-dtype fp8_e4m3
      --max-num-batched-tokens 8192
      --moe-backend cutlass
      --gpu-memory-utilization 0.20
      --no-enable-chunked-prefill
    """
    return [
        MODEL,
        "--trust-remote-code",
        "--dtype", "auto",
        "--max-model-len", str(MAX_MODEL_LEN),

        # Single local user/session.
        "--max-num-seqs", str(MAX_NUM_SEQS),

        # Conservative for Apple Metal debugging.
        "--enforce-eager",
        "--disable-log-stats",

        # Qwen tool/reasoning parsers.
        "--tool-call-parser", TOOL_CALL_PARSER,
        "--reasoning-parser", REASONING_PARSER,
        "--enable-auto-tool-choice",

        # Disable thinking by default. Clients can override per request with:
        # extra_body={"chat_template_kwargs": {"enable_thinking": True}}
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": False}, separators=(",", ":")),

        # Keep model in text-only mode.
        "--limit-mm-per-prompt",
        json.dumps({"image": 0, "video": 0, "audio": 0}, separators=(",", ":")),

        "--port", str(PORT),
        "--host", HOST,
        "--seed", str(SEED),

        "--override-generation-config",
        json.dumps(_build_generation_config(), separators=(",", ":")),
    ]


def _force_safe_arg_overrides(args: Any) -> None:
    """
    Force only the settings that matter for the Qwen3.6 hybrid KV-cache crash.
    Do not disable chunked prefill; your latest log warned that this model does
    not officially support disabling chunked prefill.
    """
    if hasattr(args, "enable_prefix_caching") and not _ALLOW_EXPERIMENTAL_PREFIX_CACHE:
        args.enable_prefix_caching = False

    # Avoid CUDA-oriented MoE backend on Metal if some default/config sets it.
    if hasattr(args, "moe_backend"):
        current = getattr(args, "moe_backend", None)
        if current == "cutlass":
            args.moe_backend = None


def _log_chat_request(request: Any) -> None:
    messages = _get(request, "messages", []) or []

    vllm_logger.info("=" * 80)
    vllm_logger.info("/v1/chat/completions request received")
    vllm_logger.info("Model: %s", _get(request, "model", "unknown"))
    vllm_logger.info("Stream: %s", _get(request, "stream", None))
    vllm_logger.info("Message count: %d", len(messages))

    for i, msg in enumerate(messages):
        role = _get(msg, "role", "unknown")
        content = _get(msg, "content", "")

        if isinstance(content, list):
            image_count = 0
            text_count = 0

            for part in content:
                part_type = _get(part, "type", "")
                if part_type == "image_url":
                    image_count += 1
                elif part_type == "text":
                    text_count += 1

            vllm_logger.info(
                "  msg[%d] role=%s: %d image_url items, %d text items",
                i,
                role,
                image_count,
                text_count,
            )

            for j, part in enumerate(content):
                part_type = _get(part, "type", "")
                if part_type == "image_url":
                    image_url = _get(part, "image_url", {})
                    url = _get(image_url, "url", "")
                    detail = _get(image_url, "detail", "not_set")
                    vllm_logger.info(
                        "    image_url[%d]: url_len=%d detail=%s url=%s",
                        j,
                        len(str(url)),
                        detail,
                        _shorten(url, 100),
                    )
                elif part_type == "text":
                    vllm_logger.info(
                        "    text[%d]: %s",
                        j,
                        _shorten(_get(part, "text", ""), 300),
                    )

        else:
            vllm_logger.info(
                "  msg[%d] role=%s: %s",
                i,
                role,
                _shorten(content, 300),
            )

    extra_body = _get(request, "extra_body", None)
    if extra_body:
        vllm_logger.info("Extra body: %s", _shorten(extra_body, 1000))

    vllm_logger.info("=" * 80)


# =============================================================================
# Main server
# =============================================================================

async def main() -> None:
    # Import vLLM only after env vars are set.
    try:
        from vllm.entrypoints.openai.api_server import (
            build_async_engine_client,
            build_app,
            init_app_state,
            setup_server,
            serve_http,
        )
    except ImportError:
        from vllm.entrypoints.openai.api_server import (
            build_async_engine_client,
            build_app,
            init_app_state,
            setup_server,
        )
        from vllm.entrypoints.launcher import serve_http  # type: ignore

    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    try:
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    except ImportError:
        OpenAIServingChat = Any  # type: ignore

    try:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    except ImportError:
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest  # type: ignore

    from fastapi import Request
    from fastapi.responses import JSONResponse, StreamingResponse

    _patch_qwen3xml_tool_parser_warning()

    parser = FlexibleArgumentParser(prog="qwen3-6-server")
    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser("serve")
    serve_parser = make_arg_parser(serve_parser)

    argv = _build_vllm_argv()

    args = serve_parser.parse_args(argv)
    args.command = "serve"
    args.model_tag = MODEL
    args.model = MODEL

    _force_safe_arg_overrides(args)
    validate_parsed_serve_args(args)

    print("Loading Qwen3.6 on vLLM-Metal...")
    print(f"Model: {MODEL}")
    print(f"Host:  {HOST}")
    print(f"Port:  {PORT}")
    print(f"Max model len: {MAX_MODEL_LEN}")
    print(f"Default max tokens: {DEFAULT_MAX_TOKENS}")
    print(f"Paged attention: {os.environ.get('VLLM_METAL_USE_PAGED_ATTENTION')}")
    print(f"Metal memory fraction: {os.environ.get('VLLM_METAL_MEMORY_FRACTION')}")
    print(f"Metal multimodal mode: {os.environ.get('VLLM_METAL_MULTIMODAL_MODE')}")
    print(f"Experimental prefix cache: {_ALLOW_EXPERIMENTAL_PREFIX_CACHE}")
    print(f"Experimental paged attention: {_ALLOW_EXPERIMENTAL_PAGED_ATTENTION}")
    print()

    async with build_async_engine_client(args) as engine_client:
        supported_tasks = await engine_client.get_supported_tasks()
        model_config = engine_client.model_config

        app = build_app(args, supported_tasks, model_config)
        request_log = _install_optional_request_logging(app)

        await init_app_state(engine_client, app.state, args, supported_tasks)

        model_name = args.model
        serving_chat: OpenAIServingChat = app.state.openai_serving_chat

        original_create_chat_completion = serving_chat.create_chat_completion

        async def logged_create_chat_completion(
            request: ChatCompletionRequest,
            raw_request: Request | None = None,
            **kwargs: Any,
        ):
            _log_chat_request(request)

            try:
                result = await original_create_chat_completion(
                    request,
                    raw_request,
                    **kwargs,
                )
                vllm_logger.info("Chat completion request completed successfully")
                return result
            except Exception as exc:
                vllm_logger.error("Chat completion request failed: %s", exc)
                vllm_logger.error(traceback.format_exc())
                raise

        serving_chat.create_chat_completion = logged_create_chat_completion

        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "model": model_name,
                "backend": "vllm-metal",
                "text_only": True,
                "max_model_len": MAX_MODEL_LEN,
                "default_max_tokens": DEFAULT_MAX_TOKENS,
                "paged_attention": os.environ.get("VLLM_METAL_USE_PAGED_ATTENTION"),
                "metal_memory_fraction": os.environ.get("VLLM_METAL_MEMORY_FRACTION"),
                "metal_multimodal_mode": os.environ.get("VLLM_METAL_MULTIMODAL_MODE"),
                "experimental_prefix_cache": _ALLOW_EXPERIMENTAL_PREFIX_CACHE,
                "experimental_paged_attention": _ALLOW_EXPERIMENTAL_PAGED_ATTENTION,
            }

        @app.post("/compress", response_model_exclude_none=True)
        async def compress(request: Request):
            """
            LLM-powered context compression.

            POST body:
              {
                "messages": [...chat history...],
                "target_tokens": 8192
              }
            """
            try:
                body = await request.json()
            except Exception as exc:
                return JSONResponse(
                    content={"error": f"Invalid JSON body: {exc}"},
                    status_code=400,
                )

            messages = body.get("messages", [])
            if not isinstance(messages, list):
                return JSONResponse(
                    content={"error": "`messages` must be a list"},
                    status_code=400,
                )

            target_tokens = int(body.get("target_tokens", 8192))
            target_tokens = max(256, min(target_tokens, 16384))

            vllm_logger.info("=" * 80)
            vllm_logger.info("/compress request received")
            vllm_logger.info("Message count: %d", len(messages))
            vllm_logger.info("Target tokens: %d", target_tokens)

            for i, msg in enumerate(messages):
                role = _get(msg, "role", "unknown")
                content = _get(msg, "content", "")
                vllm_logger.info(
                    "  msg[%d] role=%s content=%s",
                    i,
                    role,
                    _shorten(content, 300),
                )

            vllm_logger.info("=" * 80)

            request_log(
                logging.INFO,
                "N/A",
                "compress_request_start",
                msg_count=len(messages),
                target_tokens=target_tokens,
                input_preview=f"[{len(messages)} messages]",
            )

            compress_messages = [
                {"role": "system", "content": COMPRESS_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(messages, indent=2, ensure_ascii=False),
                },
            ]

            chat_req = ChatCompletionRequest(
                model=model_name,
                messages=compress_messages,
                temperature=0.1,
                max_tokens=target_tokens,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            try:
                result = await serving_chat.create_chat_completion(chat_req)
            except Exception as exc:
                tb = traceback.format_exc()
                request_log(
                    logging.ERROR,
                    "N/A",
                    "compress_error",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    traceback=tb,
                    messages_len=len(messages),
                )
                return JSONResponse(
                    content={"error": str(exc), "error_type": type(exc).__name__},
                    status_code=500,
                )

            if hasattr(result, "error"):
                err = getattr(result, "error")
                request_log(
                    logging.ERROR,
                    "N/A",
                    "compress_result_error",
                    error=str(err),
                    messages_len=len(messages),
                )
                return JSONResponse(
                    content={"error": str(err)},
                    status_code=getattr(err, "code", 500),
                )

            raw_content, raw_reasoning = _extract_chat_content_and_reasoning(result)
            raw = _strip_json_fences(raw_content)

            try:
                compressed = json.loads(raw)
                if not isinstance(compressed, dict):
                    raise ValueError("compression result was not a JSON object")
            except Exception:
                compressed = {
                    "summary": raw,
                    "preserved_messages": [],
                    "token_budget_used": None,
                }

            if raw_reasoning:
                compressed.setdefault("_reasoning_present", True)

            request_log(
                logging.INFO,
                "N/A",
                "compress_response",
                summary_chars=len(str(compressed.get("summary", ""))),
                preserved=len(compressed.get("preserved_messages", [])),
                token_budget=compressed.get("token_budget_used"),
                output_preview=_shorten(compressed.get("summary", ""), 300),
            )

            return {
                "compressed": compressed,
                "original_message_count": len(messages),
                "target_tokens": target_tokens,
            }

        @app.post("/compress/stream")
        async def compress_stream(request: Request):
            """
            Streaming compression endpoint. Yields vLLM/OpenAI SSE events.
            """
            try:
                body = await request.json()
            except Exception as exc:
                return JSONResponse(
                    content={"error": f"Invalid JSON body: {exc}"},
                    status_code=400,
                )

            messages = body.get("messages", [])
            if not isinstance(messages, list):
                return JSONResponse(
                    content={"error": "`messages` must be a list"},
                    status_code=400,
                )

            target_tokens = int(body.get("target_tokens", 8192))
            target_tokens = max(256, min(target_tokens, 16384))

            compress_messages = [
                {"role": "system", "content": COMPRESS_PROMPT},
                {"role": "user", "content": json.dumps(messages, ensure_ascii=False)},
            ]

            chat_req = ChatCompletionRequest(
                model=model_name,
                messages=compress_messages,
                temperature=0.1,
                max_tokens=target_tokens,
                stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            try:
                generator = await serving_chat.create_chat_completion(chat_req)
            except Exception as exc:
                vllm_logger.error("/compress/stream failed: %s", exc)
                vllm_logger.error(traceback.format_exc())
                return JSONResponse(
                    content={"error": str(exc), "error_type": type(exc).__name__},
                    status_code=500,
                )

            if hasattr(generator, "error"):
                err = getattr(generator, "error")
                return JSONResponse(
                    content={"error": str(err)},
                    status_code=getattr(err, "code", 500),
                )

            return StreamingResponse(generator, media_type="text/event-stream")

        @app.post("/v1/chat/image")
        async def analyze_image_not_supported(request: Request):
            """
            Kept for backward compatibility, but this backend is text-only.
            """
            return JSONResponse(
                status_code=501,
                content={
                    "error": "image_input_not_supported",
                    "message": (
                        "This server is running Qwen3.6 through vLLM-Metal in "
                        "text-only compatibility mode. Image inputs are not "
                        "supported by this backend."
                    ),
                    "backend": "vllm-metal",
                    "text_only": True,
                },
            )

        @app.post("/v1/chat/image_base64")
        async def analyze_image_base64_not_supported(request: Request):
            """
            Kept for backward compatibility, but this backend is text-only.
            """
            return JSONResponse(
                status_code=501,
                content={
                    "error": "image_input_not_supported",
                    "message": (
                        "This server is running Qwen3.6 through vLLM-Metal in "
                        "text-only compatibility mode. Base64 image inputs are "
                        "not supported by this backend."
                    ),
                    "backend": "vllm-metal",
                    "text_only": True,
                },
            )

        listen_address, sock = setup_server(args)

        print()
        print("Qwen3.6 vLLM-Metal Server")
        print(f"Chat API:    http://{HOST}:{PORT}/v1/chat/completions")
        print(f"Compress:    http://{HOST}:{PORT}/compress")
        print(f"Health:      http://{HOST}:{PORT}/health")
        print("Image API:   disabled/text-only backend returns 501")
        print(f"Parsers:     {TOOL_CALL_PARSER} + {REASONING_PARSER}")
        print(f"Listening:   {listen_address}")
        print()

        await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=30,
        )


def run() -> None:
    """
    Synchronous entry point for console_scripts.
    """
    try:
        import uvloop

        uvloop.run(main())
    except ImportError:
        asyncio.run(main())


if __name__ == "__main__":
    run()