#!/usr/bin/env python3
"""
Local vLLM API Server for Qwen3.6-35B-A3B-NVFP4 on NVIDIA GB10
with LLM-powered context compression on the same port.

Features:
  --tool-call-parser qwen3_xml   (parses XML tool call format)
  --reasoning-parser qwen3        (routes <think> tokens into reasoning field)
  --enable-auto-tool-choice       (auto-selects tools when needed)
  /compress                        (LLM-powered context summarization)

Think-budget instructions:
  - You have a limited budget of thinking tokens.
  - Use them wisely. Plan your reasoning before generating.
  - When you reach a confident answer or conclusion, STOP thinking and output it.
  - Do NOT loop or second-guess yourself unnecessarily.
  - If you've already answered the question, do NOT add more reasoning.
  - Think once, conclude once — do not re-think the same point.
"""

import logging
import os
import json
import logging
import traceback
import base64
from datetime import datetime

# Allow long max_model_len (model's native limit is 262144)
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

# Enable VLLM request logging
os.environ["VLLM_WORKER_LOGGING_LEVEL"] = "DEBUG"

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
vllm_logger = logging.getLogger("vllm.image_request")
vllm_logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(LOG_DIR, "vllm_image_requests.log"))
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
vllm_logger.addHandler(fh)
vllm_logger.info("=" * 60)
vllm_logger.info("vLLM Image Request Logger Started")

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

    # Injected into every request that lacks a system message. Kept short so it
    # doesn't force the model to over-think trivial prompts.
    ANTI_LOOP_SYSTEM = (
        "You are a helpful, concise coding assistant. "
        "Match response length to the request: trivial questions get trivial answers. "
        "Do not repeat reasoning or restart from scratch. "
        "Do not emit Error:/Exception:/Warning: labels unless there is a genuine error. "
        "Stop as soon as the task is done."
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
        "--max-model-len", "232144",          # model native limit (256K)
        "--gpu-memory-utilization", "0.35",    # reduced for multimodal encoder cache
        "--moe-backend", "cutlass",
        "--enforce-eager",
        "--disable-log-stats",
        "--enable-prefix-caching",
        "--tool-call-parser", "qwen3_xml",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice",
        "--port", "11112",
        "--host", "0.0.0.0",
        # ── Sampling defaults (seed is engine-level; rest via --override-generation-config) ─
        "--seed", "5678",
        "--reasoning-config", '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}',
        # Qwen3 team-recommended sampling. presence/frequency penalties >0
        # cause severe degradation (random word streams that never close
        # </think>) — keep them at 0. repetition_penalty 1.05 is mild and safe.
        "--override-generation-config",
        '{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"max_tokens":102400,"repetition_penalty":1.05,"presence_penalty":0.0,"frequency_penalty":0.0,"thinking_token_budget":16384}',
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
        # NOTE: RequestLoggingMiddleware is intentionally NOT installed here.
        # BaseHTTPMiddleware re-binds request._receive after reading the body,
        # which conflicts with vLLM's listen_for_disconnect() task and corrupts
        # non-streaming responses (returns "null" with content-length=4).
        # Per-request logging is handled by the LiteLLM proxy callbacks instead.
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

        # Wrap the original method to log all requests
        original_create = serving_chat.create_chat_completion

        async def logged_create_chat_completion(request: ChatCompletionRequest, raw_request: Request = None, **kwargs):
            # Log incoming chat completion request
            vllm_logger.info("=" * 60)
            vllm_logger.info("/v1/chat/completions request received")
            vllm_logger.info("Model: %s", request.model)
            vllm_logger.info("Stream: %s", request.stream)
            vllm_logger.info("Message count: %d", len(request.messages))

            for i, msg in enumerate(request.messages):
                # Handle both dict and object messages
                if isinstance(msg, dict):
                    msg_role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                else:
                    msg_role = getattr(msg, "role", "unknown")
                    content = getattr(msg, "content", "")

                if isinstance(content, list):
                    image_types = [c for c in content if c.get("type") == "image_url"]
                    text_parts = [c for c in content if c.get("type") == "text"]
                    vllm_logger.info(
                        "  msg[%d] role=%s: %d image_url items, %d text items",
                        i, msg_role, len(image_types), len(text_parts)
                    )
                    for j, part in enumerate(content):
                        if part.get("type") == "image_url":
                            img_url = part.get("image_url", {})
                            if isinstance(img_url, dict):
                                url = img_url.get("url", "")[:100]
                                detail = img_url.get("detail", "not_set")
                            else:
                                url = str(img_url)[:100]
                                detail = "not_set"
                            size = len(img_url.get("url", "")) if isinstance(img_url, dict) else len(str(img_url))
                            vllm_logger.info(
                                "    image_url[%d]: url_len=%d, detail=%s, url=%s...",
                                j, size, detail, url
                            )
                        elif part.get("type") == "text":
                            vllm_logger.info("    text[%d]: %s", j, part.get("text", "")[:300])
                elif isinstance(content, str):
                    vllm_logger.info("  msg[%d] role=%s: %s", i, msg_role, content[:300])

            # Log extra_body
            if hasattr(request, "extra_body") and request.extra_body:
                vllm_logger.info("Extra body: %s", request.extra_body)

            vllm_logger.info("=" * 60)

            try:
                result = await original_create(request, raw_request, **kwargs)
                vllm_logger.info("Request completed successfully")
                return result
            except Exception as e:
                vllm_logger.error("Request failed: %s", str(e))
                vllm_logger.error(traceback.format_exc())
                raise

        serving_chat.create_chat_completion = logged_create_chat_completion

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

            # Log incoming messages (truncate base64 for readability)
            vllm_logger.info("=" * 60)
            vllm_logger.info("/compress request received")
            vllm_logger.info("Message count: %d", len(messages))
            for i, msg in enumerate(messages):
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Multimodal content - log structure and image count
                    image_types = [c for c in content if c.get("type") == "image_url"]
                    text_parts = [c for c in content if c.get("type") == "text"]
                    vllm_logger.info(
                        "  msg[%d] role=%s: %d image_url items, %d text items",
                        i, msg.get("role"), len(image_types), len(text_parts)
                    )
                    for j, part in enumerate(content):
                        if part.get("type") == "image_url":
                            img_url = part.get("image_url", {})
                            url = img_url.get("url", "")[:100] if isinstance(img_url, dict) else str(img_url)[:100]
                            vllm_logger.info("    image_url[%d]: %s...", j, url)
                        elif part.get("type") == "text":
                            vllm_logger.info("    text[%d]: %s", j, part.get("text", "")[:200])
                elif isinstance(content, str):
                    vllm_logger.info("  msg[%d] role=%s: %s", i, msg.get("role"), content[:300])
            vllm_logger.info("=" * 60)

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

        # ─── Image processing endpoint ──────────────────────────────────────────
        @app.post("/v1/chat/image")
        async def analyze_image(request: Request):
            """
            Dedicated image analysis endpoint.

            Supports two input formats:
            1. JSON with base64 or URL:
               {"image_url": "data:image/png;base64,... or https://..."}

            2. Multipart form with raw binary:
               - field "image": binary file data (PNG, JPG, WEBP, etc.)
               - field "prompt": text question (optional)
               - field "thinking": "true"/"false" (optional, default false)

            Returns:
              {
                "description": "Model's response about the image",
                "thinking": "Model's thinking trace (if enabled)"
              }
            """
            content_type = request.headers.get("content-type", "")

            # Handle multipart form data (raw binary upload)
            if "multipart/form-data" in content_type:
                form = await request.form()
                image_data = None
                prompt = "Describe this image in detail."
                thinking = False

                for field_name, field_value in form.items():
                    if field_name == "image" and hasattr(field_value, "read"):
                        # Raw binary file upload
                        image_data = await field_value.read()
                    elif field_name == "prompt":
                        prompt = str(field_value)
                    elif field_name == "thinking":
                        thinking = str(field_value).lower() in ("true", "1", "yes")

                if image_data is None:
                    return JSONResponse(
                        content={"error": "No image data provided"},
                        status_code=400,
                    )

                # Encode to base64
                b64 = base64.b64encode(image_data).decode("utf-8")
                mime_type = "image/png"  # default, could be smarter
                image_url = f"data:{mime_type};base64,{b64}"

            else:
                # JSON body
                body = await request.json()
                image_url = body.get("image_url")
                prompt = body.get("prompt", "Describe this image in detail.")
                thinking = body.get("thinking", False)

            vllm_logger.info("=" * 60)
            vllm_logger.info("/v1/chat/image request received")
            vllm_logger.info("Prompt: %s", prompt)
            vllm_logger.info("Thinking: %s", thinking)
            if image_url:
                url_preview = image_url[:80] if isinstance(image_url, str) else str(image_url)[:80]
                vllm_logger.info("Image URL: %s...", url_preview)
            vllm_logger.info("=" * 60)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            extra_body = {}
            if thinking:
                extra_body["chat_template_kwargs"] = {"enable_thinking": True}

            chat_req = ChatCompletionRequest(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                stream=False,
                extra_body=extra_body if extra_body else None,
            )

            try:
                result = await serving_chat.create_chat_completion(chat_req)

                vllm_logger.info("Result type: %s", type(result))
                vllm_logger.info("Result keys: %s", result.keys() if isinstance(result, dict) else "N/A")

                if hasattr(result, "error"):
                    return JSONResponse(
                        content={"error": str(result.error)},
                        status_code=getattr(result.error, "code", 500),
                    )

                # Extract content from result (handle dict vs object)
                if isinstance(result, dict):
                    choices = result.get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        message = choices[0].get("message", {})
                        content = message.get("content", "") if isinstance(message, dict) else str(message)
                        reasoning = None
                    else:
                        content = str(choices[0]) if choices else ""
                        reasoning = None
                else:
                    if hasattr(result, "choices") and result.choices:
                        message = result.choices[0].message
                        content = getattr(message, "content", str(message))
                        reasoning = getattr(message, "reasoning", None)
                    else:
                        content = str(result)
                        reasoning = None

                response = {"description": content or ""}
                if reasoning:
                    response["thinking"] = reasoning

                vllm_logger.info("/v1/chat/image completed successfully")
                return response

            except Exception as e:
                vllm_logger.error("/v1/chat/image failed: %s", str(e))
                vllm_logger.error(traceback.format_exc())
                return JSONResponse(
                    content={"error": str(e)},
                    status_code=500,
                )

        @app.post("/v1/chat/image_base64")
        async def analyze_image_base64(request: Request):
            """
            Image analysis from local base64-encoded image.

            POST body:
              {
                "image_base64": "...base64 encoded image data...",
                "prompt": "What do you see in this image?",
                "thinking": false
              }
            """
            body = await request.json()
            image_base64 = body.get("image_base64")
            prompt = body.get("prompt", "Describe this image in detail.")
            thinking = body.get("thinking", False)

            if not image_base64:
                return JSONResponse(
                    content={"error": "image_base64 is required"},
                    status_code=400,
                )

            # Prepend data URI if not present
            if not image_base64.startswith("data:"):
                image_base64 = f"data:image/png;base64,{image_base64}"

            # Build messages for direct processing
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_base64}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            extra_body = {}
            if thinking:
                extra_body["chat_template_kwargs"] = {"enable_thinking": True}

            chat_req = ChatCompletionRequest(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                stream=False,
                extra_body=extra_body if extra_body else None,
            )

            try:
                result = await serving_chat.create_chat_completion(chat_req)

                if hasattr(result, "error"):
                    return JSONResponse(
                        content={"error": str(result.error)},
                        status_code=getattr(result.error, "code", 500),
                    )

                message = result.choices[0].message
                response = {
                    "description": message.content or "",
                }

                if hasattr(message, "reasoning") and message.reasoning:
                    response["thinking"] = message.reasoning

                return response

            except Exception as e:
                vllm_logger.error("/v1/chat/image_base64 failed: %s", str(e))
                vllm_logger.error(traceback.format_exc())
                return JSONResponse(
                    content={"error": str(e)},
                    status_code=500,
                )

        # ── Serve ───────────────────────────────────────────────────────────────
        listen_address, sock = setup_server(args)
<<<<<<< HEAD
        print(f"\n🚀 Blackwell NVFP4 Server @ 256K Context")
        print(f"📡 Chat API:    http://0.0.0.0:11112/v1/chat/completions")
        print(f"🖼️  Image API:  http://0.0.0.0:11112/v1/chat/image")
        print(f"📦 Compress:    http://0.0.0.0:11112/compress")
=======
        print(f"\n🚀 Blackwell NVFP4 Server @ 200K Context")
        print(f"📡 API:        http://0.0.0.0:{args.port}/v1")
        print(f"📦 Compress:   http://0.0.0.0:{args.port}/compress")
>>>>>>> main
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
