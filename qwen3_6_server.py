#!/usr/bin/env python3
"""
Local vLLM API Server for Qwen3.6-35B-A3B-NVFP4 on NVIDIA GB10
with LLM-powered context compression on the same port.

Features:
  --tool-call-parser qwen3_xml   (parses XML tool call format)
  --reasoning-parser qwen3        (routes <think> tokens into reasoning field)
  --enable-auto-tool-choice       (auto-selects tools when needed)
  /compress                        (LLM-powered context summarization)
"""

import logging
import os
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _log(level: int, req_id: str, event: str, **fields: Any) -> None:
    """Log to terminal and logs/vllm_requests.log."""
    from datetime import datetime, timezone

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "req_id": req_id,
        "event": event,
        **fields,
    }
    line = json.dumps(payload, ensure_ascii=False, default=str)

    # Terminal
    print(line)

    # File
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "vllm_requests.log"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# Allow long max_model_len (model's native limit is 262144)
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

# Ensure venv packages take priority
_venv_bin = os.path.join(os.path.dirname(__file__), "venv", "bin")
if _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

_venv_lib = os.path.join(os.path.dirname(__file__), "venv", "lib", "python3.12", "site-packages")
if _venv_lib not in sys.path:
    sys.path.insert(0, _venv_lib)


# ─── Compression prompt ───────────────────────────────────────────────────────
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


# ─── Main ─────────────────────────────────────────────────────────────────────
async def main():
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client,
        build_app,
        init_app_state,
        setup_server,
        serve_http,
    )
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    # ── Patch vLLM's broken logger.warning before engine loads ─────────────────
    # vLLM's qwen3xml_tool_parser._convert_param_value logs a 3-arg warning but
    # passes only 2 args, causing a cascading logging loop.  Silence it.
    import vllm.tool_parsers.qwen3xml_tool_parser as _qxt
    _orig_warn = _qxt.logger.warning
    def _silence_float_warn(msg, *args, **kwargs):
        if "is not a float in tool" in str(msg):
            return
        _orig_warn(msg, *args, **kwargs)
    _qxt.logger.warning = _silence_float_warn

    # ── Fallback system prompt (applies when client doesn't send one) ──────────
    ANTI_LOOP_SYSTEM = (
        "You are a helpful coding assistant. "
        "Provide direct, concise responses. Move forward with each step."
    )

    # ── Build vLLM args ─────────────────────────────────────────────────────────
    parser = FlexibleArgumentParser(prog="qwen3-6-server")
    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser("serve")
    serve_parser = make_arg_parser(serve_parser)

    argv = [
        "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--max-model-len", "220000",
        "--gpu-memory-utilization", "0.35",
        "--max-num-batched-tokens", "4096",
        "--moe-backend", "cutlass",
        "--enforce-eager",
        "--disable-log-stats",
        "--enable-prefix-caching",
        "--tool-call-parser", "qwen3_xml",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice",
        "--port", "11112",
        "--host", "0.0.0.0",
    ]

    args = serve_parser.parse_args(argv)
    args.command = "serve"
    args.model_tag = argv[0]
    args.model = args.model_tag
    validate_parsed_serve_args(args)

    # ── Start engine (blocking until model is loaded) ──────────────────────────
    print("⏳ Loading model… this may take a few minutes.")
    async with build_async_engine_client(args) as engine_client:
        supported_tasks = await engine_client.get_supported_tasks()
        model_config = engine_client.model_config

        app = build_app(args, supported_tasks, model_config)

        import request_logging
        request_logging.install(app)
        _log = request_logging._log

        await init_app_state(engine_client, app.state, args, supported_tasks)

        from fastapi import Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        model_name = args.model
        serving_chat: OpenAIServingChat = app.state.openai_serving_chat

        # ── Patch serving_chat.create_chat_completion to log reasoning ─────────
        import request_logging as _rl
        _orig_create = serving_chat.create_chat_completion

        async def _create_with_reasoning_logging(chat_request, raw_request=None, **kwargs):
            # Inject anti-loop system prompt if none provided
            msgs = getattr(chat_request, "messages", []) or []
            has_system = any(m.get("role") == "system" and m.get("content", "").strip()
                            for m in msgs)
            if not has_system:
                msgs.insert(0, {"role": "system", "content": ANTI_LOOP_SYSTEM})
                chat_request.messages = msgs

            req_id = getattr(chat_request, "extra_query_params", {}).get("request_id", "unknown")
            _rl.log_reasoning_separator(req_id, "request_start",
                                        model=getattr(chat_request, "model", "?"),
                                        msgs=len(getattr(chat_request, "messages", [])))
            _pending_reasoning = []

            result = await _orig_create(chat_request, **kwargs)

            # Non-streaming: log the reasoning field directly
            if not getattr(result, "is_streaming", False):
                try:
                    choices = result.choices
                    reasoning = None
                    if choices:
                        msg = choices[0].get("message", {})
                        reasoning = msg.get("reasoning", "")
                    if reasoning:
                        _rl.log_reasoning(req_id, "assistant", reasoning, is_final=True)
                except Exception:
                    pass
                return result

            # Streaming: wrap the generator to intercept reasoning chunks
            async def _wrapped_stream():
                reasoning_buf = ""
                content_buf = ""

                # result is a StreamingResponse — extract the async iterator
                iterator = result.body_iterator if hasattr(result, "body_iterator") else result

                async for raw_bytes in iterator:
                    raw = raw_bytes.decode("utf-8") if isinstance(raw_bytes, bytes) else str(raw_bytes)
                    if raw.startswith("data: "):
                        raw = raw[6:]
                    if raw.strip() in ("[DONE]", ""):
                        yield raw_bytes
                        continue
                    try:
                        chunk = json.loads(raw)
                    except Exception:
                        yield raw_bytes
                        continue
                    delta = (
                        chunk.get("choices", [{}])[0].get("delta", {})
                        if chunk.get("choices")
                        else {}
                    )
                    delta_reasoning = delta.get("reasoning", "") or ""
                    delta_content = delta.get("content", "") or ""
                    if delta_reasoning:
                        reasoning_buf += delta_reasoning
                        _rl.log_reasoning(req_id, "assistant", delta_reasoning)
                    if delta_content:
                        content_buf += delta_content
                    yield raw_bytes

                # Flush remaining reasoning
                if reasoning_buf:
                    _rl.log_reasoning(req_id, "assistant",
                                      f"[...{len(reasoning_buf)} chars total...]", is_final=True)
                _rl.log_reasoning_separator(req_id, "request_end",
                                            content_chars=len(content_buf))

            return _wrapped_stream()

        serving_chat.create_chat_completion = _create_with_reasoning_logging

        @app.post("/compress", response_model_exclude_none=True)
        async def compress(request: Request):
            """
            LLM-powered context compression.
            POST body:
              {
                "messages": [...chat history...],
                "target_tokens": 8192   # optional, default 8192
              }
            Returns:
              {
                "compressed": {
                  "summary": "...",
                  "preserved_messages": [...],
                  "token_budget_used": 0.42
                },
                "original_message_count": 15,
                "target_tokens": 8192,
              }
            """
            body = await request.json()
            messages = body.get("messages", [])
            target_tokens = body.get("target_tokens", 8192)

            input_preview = f"[{len(messages)} msgs, target_tokens={target_tokens}]"
            _log(logging.INFO, "N/A", "compress_request_start",
                 msg_count=len(messages), target_tokens=target_tokens,
                 input_preview=input_preview)

            compress_messages = [
                {"role": "system", "content": COMPRESS_PROMPT},
                {"role": "user", "content": json.dumps(messages, indent=2, ensure_ascii=False)},
            ]

            chat_req = ChatCompletionRequest(
                model=model_name,
                messages=compress_messages,
                temperature=0.1,
                max_tokens=8192,
                stream=False,
            )

            try:
                result = await serving_chat.create_chat_completion(chat_req)
            except Exception as e:
                tb = traceback.format_exc()
                _log(logging.ERROR, "N/A", "compress_error", error=str(e),
                     error_type=type(e).__name__, traceback=tb, messages_len=len(messages))
                return JSONResponse(
                    content={"error": str(e), "error_type": type(e).__name__},
                    status_code=500,
                )

            if hasattr(result, "error"):
                _log(logging.ERROR, "N/A", "compress_result_error",
                     error=str(result.error), messages_len=len(messages))
                return JSONResponse(
                    content={"error": str(result.error)},
                    status_code=getattr(result.error, "code", 500),
                )

            raw = result.choices[0].message.content.strip()
            for fence in ("```json", "```JSON", "```"):
                if raw.startswith(fence):
                    raw = raw[len(fence):]
                if raw.endswith(fence):
                    raw = raw[: -len(fence)]
            raw = raw.strip()

            try:
                compressed = json.loads(raw)
            except json.JSONDecodeError:
                compressed = {
                    "summary": raw,
                    "preserved_messages": [],
                    "token_budget_used": None,
                }

            _log(logging.INFO, "N/A", "compress_response",
                 summary_chars=len(compressed.get("summary", "")),
                 preserved=len(compressed.get("preserved_messages", [])),
                 token_budget=compressed.get("token_budget_used"),
                 output_preview=compressed.get("summary", "")[:300])

            return {
                "compressed": compressed,
                "original_message_count": len(messages),
                "target_tokens": target_tokens,
            }

        @app.post("/compress/stream")
        async def compress_stream(request: Request):
            """Streaming compression — yields SSE events."""
            body = await request.json()
            messages = body.get("messages", [])

            compress_messages = [
                {"role": "system", "content": COMPRESS_PROMPT},
                {"role": "user", "content": json.dumps(messages)},
            ]

            chat_req = ChatCompletionRequest(
                model=model_name,
                messages=compress_messages,
                temperature=0.1,
                max_tokens=8192,
                stream=True,
            )

            generator = await serving_chat.create_chat_completion(chat_req)
            return StreamingResponse(generator, media_type="text/event-stream")

        # ── Serve ───────────────────────────────────────────────────────────────
        listen_address, sock = setup_server(args)
        print(f"\n🚀 Blackwell NVFP4 Server @ 200K Context")
        print(f"📡 API:        http://0.0.0.0:{args.port}/v1")
        print(f"📦 Compress:   http://0.0.0.0:{args.port}/compress")
        print(f"🔧 Parsers:    qwen3_xml + qwen3 reasoning")
        print()
        await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=30,
        )


if __name__ == "__main__":
    import uvloop
    uvloop.run(main())


def run():
    """Synchronous entry point for console_scripts."""
    import uvloop
    uvloop.run(main())
