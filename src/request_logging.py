"""
Structured request/response logging middleware for the vLLM backend (port 11112).

Logs every /v1/chat/completions and /v1/completions request with:
  - Unique request ID
  - Timestamp (ISO)
  - Request body (model, messages, params)
  - Response text / error
  - Token counts (estimated)
  - Duration in seconds

Output goes to:
  - Rotating file: logs/vllm_requests.log (max 10 MB, 5 backups)
  - Console (DEBUG level only)

Also exposes a lightweight HTTP logging endpoint at /log to let
LiteLLM callbacks forward their logs to the same file.
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "vllm_requests.log"
CONSOLE_LOG_FILE = LOG_DIR / "vllm_console.log"
REASONING_LOG_FILE = LOG_DIR / "reasoning.log"

# ── File-based reasoning logger ───────────────────────────────────────────────

_reasoning_fh = logging.FileHandler(REASONING_LOG_FILE, encoding="utf-8")
_reasoning_fh.setLevel(logging.DEBUG)
_reasoning_fh.setFormatter(logging.Formatter("%(message)s\n"))
_reasoning_logger = logging.getLogger("reasoning")
_reasoning_logger.setLevel(logging.DEBUG)
_reasoning_logger.addHandler(_reasoning_fh)
_reasoning_logger.propagate = False


def log_reasoning(req_id: str, role: str, chunk: str, is_final: bool = False) -> None:
    """Append a reasoning chunk (or final reasoning block) to reasoning.log."""
    marker = "【FINAL】" if is_final else "【THINK】"
    _reasoning_logger.info(
        "%s req=%s role=%s %s",
        marker,
        req_id,
        role,
        chunk[:500],
    )


def log_reasoning_separator(req_id: str, event: str, **fields) -> None:
    """Log a separator line between reasoning passes."""
    payload = json.dumps(
        {"req_id": req_id, "event": event, **fields},
        ensure_ascii=False,
        default=str,
    )
    _reasoning_logger.info("─── %s ───", payload)

# ── Logger setup ──────────────────────────────────────────────────────────────

_req_logger = logging.getLogger("vllm_requests")
_req_logger.handlers.clear()

# File handler — rotate at 10 MB
_fh = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter(
    "%(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
))

# Console handler
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
))

_req_logger.addHandler(_fh)
_req_logger.addHandler(_ch)
_req_logger.propagate = False


def _log(
    level: int,
    req_id: str,
    event: str,
    **fields: Any,
) -> None:
    """Write a structured JSON log line."""
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "req_id": req_id,
        "event": event,
        **fields,
    }
    _req_logger.log(level, json.dumps(payload, ensure_ascii=False, default=str))


# ── Token counter (fallback, no HF deps) ──────────────────────────────────────

def _approx_tokens(text: str) -> int:
    """Very rough estimate: ~4 chars per token on average."""
    return max(1, len(text) // 4)


def _count_message_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") or p.get("inline_data", {}).get("data", "")
                for p in content if isinstance(p, dict)
            )
        elif not isinstance(content, str):
            content = str(content)
        total += _approx_tokens(f"{role}: {content}")
    return total


# ── HTTP request middleware ─────────────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that logs every request + response to vllm_requests.log.
    Skips health-check and metrics endpoints for noise reduction.
    """

    SKIP_PATHS = {"/health", "/metrics", "/v1/models", "/log", "/docs", "/openapi.json"}

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        path = request.url.path
        if path in self.SKIP_PATHS or request.url.path.startswith("/docs"):
            return await call_next(request)

        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())[:8]
        start = time.monotonic()

        body = b""
        if request.method in ("POST", "PUT", "PATCH"):
            body = await request.body()
            # Re-bind body so downstream handlers can read it
            async def receive():
                return {"type": "http.request", "body": body}
            request._receive = receive

        try:
            body_decoded: Any = None
            if body:
                try:
                    body_decoded = json.loads(body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    body_decoded = body.decode("utf-8", errors="replace")[:500]

            _log(
                logging.INFO,
                req_id,
                "request_start",
                method=request.method,
                path=path,
                model=body_decoded.get("model") if isinstance(body_decoded, dict) else None,
                msg_count=(
                    len(body_decoded.get("messages", []))
                    if isinstance(body_decoded, dict) and "messages" in body_decoded
                    else None
                ),
                msg_tokens=(
                    _count_message_tokens(body_decoded.get("messages", []))
                    if isinstance(body_decoded, dict) and "messages" in body_decoded
                    else None
                ),
            )

            response = await call_next(request)

            duration = time.monotonic() - start
            status = response.status_code

            # Note: response body capture disabled; iterating the body_iterator
            # of vLLM's non-streaming responses corrupts the payload (returns "null").
            # For response inspection use the LiteLLM proxy logs instead.
            response_text = ""

            _log(
                logging.INFO if status < 400 else logging.WARNING,
                req_id,
                "request_end",
                method=request.method,
                path=path,
                status=status,
                duration_s=round(duration, 3),
                response_preview=(response_text[:500] if response_text else ""),
            )

            return response

        except Exception as exc:
            duration = time.monotonic() - start
            _log(
                logging.ERROR,
                req_id,
                "request_crash",
                method=request.method,
                path=path,
                error=str(exc),
                error_type=type(exc).__name__,
                traceback=traceback.format_exc(),
                duration_s=round(duration, 3),
            )
            return JSONResponse(
                {"error": "internal error", "detail": str(exc)},
                status_code=500,
            )


# ── HTTP log endpoint (for LiteLLM callbacks) ─────────────────────────────────

_log_buffer: list[dict] = []
_BUFFER_LOCK = asyncio.Lock()


async def log_endpoint(request: Request) -> JSONResponse:
    """
    Lightweight POST /log endpoint that accepts log records from LiteLLM
    callbacks and forwards them to the same vllm_requests.log.

    Body:
      {
        "level": "INFO|WARNING|ERROR|DEBUG",
        "event": "lite_llm_pre_call|...",
        "req_id": "abc123",
        ...any extra fields...
      }
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    level_name = body.pop("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    event = body.pop("event", "unknown")
    req_id = body.pop("req_id", "-")

    # Compute token info if messages are present
    msg_tokens = None
    msg_count = None
    model = None
    if "messages" in body:
        msg_count = len(body["messages"])
        msg_tokens = _count_message_tokens(body["messages"])
    if "model" in body:
        model = body["model"]

    log_entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "req_id": req_id,
        "event": event,
        "source": "litellm_callback",
        "msg_count": msg_count,
        "msg_tokens": msg_tokens,
        "model": model,
        **body,
    }

    _log(level, req_id, event, **body)

    async with _BUFFER_LOCK:
        _log_buffer.append(log_entry)
        if len(_log_buffer) > 5000:
            _log_buffer[:] = _log_buffer[-3000:]

    return JSONResponse({"status": "ok", "msg_tokens": msg_tokens, "msg_count": msg_count})


async def get_recent_logs(request: Request) -> JSONResponse:
    """GET /log — return last N log entries as JSON for debugging."""
    n = int(request.query_params.get("n", 100))
    async with _BUFFER_LOCK:
        return JSONResponse(_log_buffer[-n:])


# ── Streaming response wrapper ──────────────────────────────────────────────────

async def _log_streaming_response(
    req_id: str,
    method: str,
    path: str,
    generator,
):
    """Consume a streaming response and log the final text + duration."""
    chunks = []
    start = time.monotonic()
    error = None
    try:
        async for chunk in generator:
            yield chunk
            if isinstance(chunk, str):
                chunks.append(chunk)
            elif hasattr(chunk, "text"):
                chunks.append(getattr(chunk, "text", ""))
    except Exception as exc:
        error = str(exc)
        error_type = type(exc).__name__
        tb = traceback.format_exc()
    finally:
        duration = time.monotonic() - start
        full_text = "".join(chunks)
        log_fn = logging.WARNING if error else logging.INFO
        _log(
            log_fn,
            req_id,
            "streaming_response",
            method=method,
            path=path,
            response_chars=len(full_text),
            response_tokens=_approx_tokens(full_text),
            duration_s=round(duration, 3),
            error=error,
            error_type=error_type if error else None,
            traceback=tb if error else None,
            response_preview=full_text[:500] if full_text else None,
        )


# ── Installer ──────────────────────────────────────────────────────────────────

def install(app: ASGIApp) -> None:
    """
    Add the /log endpoint to the FastAPI app for LiteLLM callback forwarding.
    Middleware is skipped (app already started) — logging is done via
    wrapping the serving_chat methods in qwen3_6_server.py.
    """
    from fastapi import APIRouter
    router = APIRouter()
    router.add_api_route("/log", log_endpoint, methods=["POST"], include_in_schema=False)
    router.add_api_route("/log", get_recent_logs, methods=["GET"], include_in_schema=False)
    app.include_router(router)
    _req_logger.info(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "logging_installed",
        "log_file": str(LOG_FILE),
    }))
