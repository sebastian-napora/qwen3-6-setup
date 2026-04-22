"""
Qwen3.6-35B-NVFP4 Auto-Compression Callback for LiteLLM.

Implements CustomLogger.async_pre_call_hook — intercepts requests before
they are forwarded to the LLM and auto-compresses long conversations.

Triggered when estimated input tokens exceed COMPRESS_THRESHOLD_TOKENS (default: 50 000).
Compresses older messages via /compress endpoint on the same vLLM backend,
then replaces them with a compressed summary.

Architecture:
    Copilot → LiteLLM (11111) → vLLM (11112)
                                         ↑
                                  /compress @ 11112
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
import traceback
import urllib.request
from typing import Any, Optional, Union

import litellm
from litellm.integrations.custom_logger import CustomLogger

# Log forwarding — send LiteLLM callback logs to the vLLM backend's /log endpoint
_COMPRESS_LOG_ENDPOINT = os.environ.get(
    "LITE_LLM_LOG_ENDPOINT", "http://localhost:11112/log"
)


def _log(
    level: int,
    req_id: str,
    event: str,
    **fields: Any,
) -> None:
    """Log to terminal (logger) AND to logs/vllm_requests.log."""
    from datetime import datetime, timezone
    from pathlib import Path

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "req_id": req_id,
        "event": event,
        **fields,
    }
    line = json.dumps(payload, ensure_ascii=False, default=str)

    # Terminal output
    logger.log(level, line)

    # Append to file directly
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "vllm_requests.log"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


logger = logging.getLogger("qwen36_compress")
logger.setLevel(logging.DEBUG)

COMPRESS_THRESHOLD_TOKENS: int = int(
    os.environ.get("LITE_LLM_COMPRESS_THRESHOLD_TOKENS", "150000")
)
COMPRESS_TARGET_TOKENS: int = int(
    os.environ.get("LITE_LLM_COMPRESS_TARGET_TOKENS", "16384")
)
COMPRESS_ENDPOINT: str = os.environ.get(
    "LITE_LLM_COMPRESS_ENDPOINT", "http://localhost:11112/compress"
)
COMPRESSED_MODELS: set[str] = set(
    m.strip()
    for m in os.environ.get("LITE_LLM_COMPRESS_MODELS", "qwen3.6-35b-nvfp4").split(",")
    if m.strip()
)
PRESERVE_RECENT_MESSAGES: int = int(
    os.environ.get("LITE_LLM_COMPRESS_PRESERVE_RECENT", "5")
)

# ── Token budget & proactive compression ────────────────────────────────────
MAX_CONTEXT_TOKENS: int = int(os.environ.get("LITE_LLM_MAX_CONTEXT_TOKENS", "226000"))
PROACTIVE_MARGIN: int = int(os.environ.get("LITE_LLM_PROACTIVE_MARGIN", "10000"))
LARGE_CHUNK_TOKEN_THRESHOLD: int = int(
    os.environ.get("LITE_LLM_LARGE_CHUNK_THRESHOLD", "3000")
)

_CACHED_TOKENIZER: Any = None


def _token_encode(text: str) -> list[int]:
    """Thread-safe token encoding via cached HuggingFace tokenizer."""
    global _CACHED_TOKENIZER
    if _CACHED_TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
            _CACHED_TOKENIZER = AutoTokenizer.from_pretrained(
                "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
                use_fast=True,
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.warning("Failed to load tokenizer for token counting: %s", exc)
            return []
    try:
        return _CACHED_TOKENIZER.encode(text, add_special_tokens=False)
    except Exception as exc:
        logger.warning("Token encoding failed: %s", exc)
        return []


def _count_message_tokens(msg: dict) -> int:
    """Estimate tokens for a single message dict."""
    content = msg.get("content", "")
    if isinstance(content, list):
        text_parts = [
            p.get("text", "") or p.get("inline_data", {}).get("data", "")
            for p in content if isinstance(p, dict)
        ]
        content = " ".join(text_parts)
    elif not isinstance(content, str):
        content = str(content)
    role = msg.get("role", "")
    return len(_token_encode(f"{role}: {content}"))


def count_messages_tokens(messages: list[dict]) -> int:
    """Count total tokens in a messages list."""
    return sum(_count_message_tokens(msg) for msg in messages)


def _sync_compress(messages: list[dict], target_tokens: int) -> dict | None:
    """Synchronous /compress call — used as fallback in thread pool."""
    payload = json.dumps({
        "messages": messages,
        "target_tokens": target_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        COMPRESS_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("compressed")
    except Exception as exc:
        logger.error("Compression call failed: %s", exc)
        return None


async def _async_compress(messages: list[dict], target_tokens: int) -> dict | None:
    """Async /compress call using aiohttp."""
    payload = json.dumps({
        "messages": messages,
        "target_tokens": target_tokens,
    })

    try:
        import aiohttp
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)
        ) as session:
            async with session.post(
                COMPRESS_ENDPOINT,
                data=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                result = await resp.json()
                return result.get("compressed")
    except ImportError:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_compress, messages, target_tokens)
    except Exception as exc:
        logger.error("Async compression call failed: %s", exc)
        return None


def _detect_large_chunks(messages: list[dict]) -> list[dict]:
    """Return messages that contain content blocks larger than LARGE_CHUNK_TOKEN_THRESHOLD."""
    large = []
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", "") or part.get("inline_data", {}).get("data", "")
                    tokens = len(_token_encode(text))
                    if tokens > LARGE_CHUNK_TOKEN_THRESHOLD:
                        large.append({
                            "msg_index": i,
                            "role": msg.get("role"),
                            "tokens": tokens,
                            "preview": text[:200],
                        })
                        break
        elif isinstance(content, str):
            tokens = len(_token_encode(content))
            if tokens > LARGE_CHUNK_TOKEN_THRESHOLD:
                large.append({
                    "msg_index": i,
                    "role": msg.get("role"),
                    "tokens": tokens,
                    "preview": content[:200],
                })
    return large


def _build_system_prompt(
    summary: dict,
    preserved: list[dict],
    token_budget: dict | None = None,
    large_chunks: list[dict] | None = None,
) -> str:
    """Build a system message from the compression result."""
    summary_text = summary.get("summary", "")
    preserved_msgs = preserved[-PRESERVE_RECENT_MESSAGES:] if preserved else []

    lines = [
        "[CONTEXT COMPRESSED — START]",
        f"Summary: {summary_text}",
        "",
        "Preserved messages (recent context):",
    ]
    for msg in preserved_msgs:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"\n--- {role.upper()} ---")
        lines.append(content)

    if token_budget:
        pct = token_budget.get("pct_used", 0)
        remaining = token_budget.get("remaining", 0)
        lines.append(
            f"\n[TOKEN BUDGET: {pct:.0%} used — ~{remaining:,} tokens remaining in context window]"
        )

    if large_chunks:
        total_chunk_tokens = sum(c.get("tokens", 0) for c in large_chunks)
        lines.append(
            f"\n[LARGE DATA CHUNKS DETECTED: {len(large_chunks)} block(s), "
            f"~{total_chunk_tokens:,} tokens total]"
        )
        for chunk in large_chunks[:3]:
            role = chunk.get("role", "?")
            tokens = chunk.get("tokens", 0)
            preview = chunk.get("preview", "")
            lines.append(
                f"  - [{role}] ~{tokens:,} tokens: \"{preview[:120]}...\""
            )
        if len(large_chunks) > 3:
            lines.append(f"  ... and {len(large_chunks) - 3} more")

    lines.append("\n[CONTEXT COMPRESSED — END]")
    return "\n".join(lines)


_SESSION_HISTORY: dict[str, list[dict]] = {}
_SESSION_LOCK = threading.Lock()
_COUNTER_LOCK = threading.Lock()
_CALL_COUNTER = 0


def _next_call_id() -> str:
    global _CALL_COUNTER
    with _COUNTER_LOCK:
        _CALL_COUNTER += 1
        return f"llm-{_CALL_COUNTER:06d}"

SUMMARY_SYSTEM_PROMPT = (
    "You are a session summarizer. Given a conversation, produce a concise "
    "(under 200 tokens) but informative summary capturing:\n"
    "1. Key facts, decisions, or outcomes reached\n"
    "2. Open questions or pending tasks\n"
    "3. Any important context the user shared\n"
    "Return ONLY the summary text, no preamble or markup."
)

SUMMARY_MODEL = "qwen3.6-35b-nvfp4"


def _summarize_sync(messages: list[dict], api_base: str) -> str | None:
    """Run a blocking /completions call to summarize messages."""
    payload = json.dumps({
        "prompt": SUMMARY_SYSTEM_PROMPT
                  + "\n\n---\n"
                  + "\n\n".join(
                      f"{m.get('role','')}: {m.get('content','')}"
                      for m in messages[-20:]
                  ),
        "max_tokens": 256,
        "temperature": 0.2,
        "skip_special_tokens": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        api_base,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("choices", [{}])[0].get("text", "").strip()
    except Exception as exc:
        logger.warning("Summarization call failed: %s", exc)
        return None


async def _async_summarize(messages: list[dict], api_base: str) -> str | None:
    """Async summarize via aiohttp."""
    try:
        import aiohttp
        payload = json.dumps({
            "prompt": SUMMARY_SYSTEM_PROMPT
                      + "\n\n---\n"
                      + "\n\n".join(
                          f"{m.get('role','')}: {m.get('content','')}"
                          for m in messages[-20:]
                      ),
            "max_tokens": 256,
            "temperature": 0.2,
            "skip_special_tokens": True,
        })
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            async with session.post(
                api_base,
                data=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                result = await resp.json()
                return result.get("choices", [{}])[0].get("text", "").strip()
    except ImportError:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _summarize_sync, messages, api_base,
        )
    except Exception as exc:
        logger.warning("Async summarization failed: %s", exc)
        return None


class Qwen36CompressCallback(CustomLogger):
    """
    LiteLLM CustomLogger that auto-compresses long conversations before inference.

    Conditions to trigger compression:
      1. Model name matches LITE_LLM_COMPRESS_MODELS (default: qwen3.6-35b-nvfp4)
      2. Estimated input tokens > LITE_LLM_COMPRESS_THRESHOLD_TOKENS (default: 50 000)
      3. At least 10 messages to compress (don't compress short chats)

    Returns a modified ``data`` dict with compressed messages if compression was
    triggered, otherwise returns ``None`` to pass through unchanged.
    """

    def __init__(
        self,
        threshold_tokens: int = COMPRESS_THRESHOLD_TOKENS,
        target_tokens: int = COMPRESS_TARGET_TOKENS,
        models: set[str] | None = None,
        preserve_recent: int = PRESERVE_RECENT_MESSAGES,
    ):
        super().__init__()
        self.threshold_tokens = threshold_tokens
        self.target_tokens = target_tokens
        self.models = models or COMPRESSED_MODELS
        self.preserve_recent = preserve_recent
        logger.info(
            "Qwen36CompressCallback init: threshold=%d tokens, target=%d tokens, models=%s",
            self.threshold_tokens, self.target_tokens, self.models,
        )

    def _session_id(self, data: dict) -> str:
        """Derive a stable session key from the request data."""
        user = data.get("user", "default")
        model = data.get("model", "unknown")
        raw = f"{user}::{model}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _should_compress(
        self,
        model: str | None,
        messages: list[dict],
        total_tokens: int | None = None,
    ) -> bool:
        if not model or not messages:
            return False
        model_lower = model.lower()
        if not any(
            m.lower() in model_lower or model_lower in m.lower()
            for m in self.models
        ):
            return False
        if len(messages) < 10:
            return False
        total = total_tokens if total_tokens is not None else count_messages_tokens(messages)
        # Primary: over threshold
        if total > self.threshold_tokens:
            return True
        # Proactive: within PROACTIVE_MARGIN of threshold (early compression)
        if total > self.threshold_tokens - PROACTIVE_MARGIN:
            return True
        return False

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Called by the LiteLLM proxy before each request is forwarded to the LLM.

        Returns:
          None        → pass through data unchanged
          dict        → replace data with returned dict
          str/Exception → rejection (not used here)
        """
        if call_type not in ("acompletion", "completion"):
            return None

        call_id = _next_call_id()
        try:
            messages = data.get("messages", [])
            if isinstance(messages, str):
                _log(logging.DEBUG, call_id, "llm_pre_call_skipped", reason="string_messages")
                return None

            model = data.get("model")
            total = count_messages_tokens(messages)

            # Log the incoming request with INPUT
            input_preview = ""
            try:
                msgs = data.get("messages", [])
                if isinstance(msgs, list):
                    total_len = sum(
                        len(str(m.get("content", ""))) for m in msgs
                    )
                    input_preview = (
                        f"[{len(msgs)} msgs, ~{total_len} chars]"
                        + f" | first: {str(msgs[0].get('content',''))[:100]}"
                        + f" | last: {str(msgs[-1].get('content',''))[:100]}"
                    )
            except Exception:
                input_preview = "unavailable"

            _log(logging.DEBUG, call_id, "llm_pre_call",
                 model=model, msg_count=len(messages), estimated_tokens=total,
                 call_type=call_type, input_preview=input_preview)

            # Store call_id so async_post_call_success_hook can correlate logs
            data["_call_id"] = call_id

            # Inject prior session summary if we have one, persist back into data
            messages = self.rewrite_messages(messages, data)
            data["messages"] = messages
            total = count_messages_tokens(messages)

            if not self._should_compress(model, messages, total_tokens=total):
                _log(logging.DEBUG, call_id, "llm_pre_call_skip_compress",
                             model=model, msg_count=len(messages),
                             estimated_tokens=total, reason="under_threshold")
                budget = {
                    "pct_used": min(total / MAX_CONTEXT_TOKENS, 1.0),
                    "remaining": max(MAX_CONTEXT_TOKENS - total, 0),
                    "total": MAX_CONTEXT_TOKENS,
                }
                large_chunks = _detect_large_chunks(messages)
                messages = self._inject_token_budget_notice(messages, budget, large_chunks)
                data["messages"] = messages
                return None

            is_proactive = total <= self.threshold_tokens
            _log(logging.INFO, call_id, "llm_compress_trigger",
                         model=model, msg_count=len(messages),
                         estimated_tokens=total, threshold=self.threshold_tokens,
                         proactive=is_proactive)

            logger.info(
                "Compressing %d messages (~%d tokens, threshold=%d, proactive_margin=%d) "
                "for model=%s — %s",
                len(messages), total, self.threshold_tokens, PROACTIVE_MARGIN,
                model, "proactive" if is_proactive else "over_threshold",
            )

            large_chunks = _detect_large_chunks(messages)
            compressed = await _async_compress(messages, self.target_tokens)

            if compressed is None:
                _log(logging.ERROR, call_id, "llm_compress_failed",
                             msg_count=len(messages), estimated_tokens=total)
                logger.warning("Compression failed — forwarding original messages")
                budget = {
                    "pct_used": min(total / MAX_CONTEXT_TOKENS, 1.0),
                    "remaining": max(MAX_CONTEXT_TOKENS - total, 0),
                    "total": MAX_CONTEXT_TOKENS,
                }
                messages = self._inject_token_budget_notice(messages, budget, large_chunks)
                data["messages"] = messages
                return None

            summary = compressed.get("summary", "")
            preserved = compressed.get("preserved_messages", [])
            budget_val = compressed.get("token_budget_used")

            logger.info(
                "Compression done: summary=%d chars, preserved=%d msgs, budget=%.2f",
                len(summary), len(preserved), budget_val if budget_val else -1,
            )
            _log(logging.INFO, call_id, "llm_compress_done",
                         summary_chars=len(summary), preserved_msgs=len(preserved),
                         token_budget=budget_val)

            budget = {
                "pct_used": min(budget_val or 0.0, 1.0),
                "remaining": max(int((1 - (budget_val or 0)) * MAX_CONTEXT_TOKENS), 0),
                "total": MAX_CONTEXT_TOKENS,
            }

            system_content = _build_system_prompt(
                {"summary": summary}, preserved,
                token_budget=budget,
                large_chunks=large_chunks,
            )

            compressed_messages = [{"role": "system", "content": system_content}]
            compressed_messages.extend(messages[-self.preserve_recent:])

            new_total = count_messages_tokens(compressed_messages)
            logger.info(
                "Compressed: %d msgs (~%d tokens) → %d msgs (~%d tokens)",
                len(messages), total, len(compressed_messages), new_total,
            )
            _log(logging.INFO, call_id, "llm_compress_result",
                         orig_msgs=len(messages), orig_tokens=total,
                         new_msgs=len(compressed_messages), new_tokens=new_total)

            data = dict(data)
            data["messages"] = compressed_messages
            return data

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("async_pre_call_hook FAILED: %s\n%s", exc, tb)
            _log(logging.ERROR, call_id, "llm_pre_call_error",
                         error=str(exc), error_type=type(exc).__name__, traceback=tb)
            return None  # Don't block the request on compression errors

    def _inject_token_budget_notice(
        self,
        messages: list[dict],
        budget: dict,
        large_chunks: list[dict],
    ) -> list[dict]:
        """
        Inject a budget/large-chunk notice into the first system message,
        or prepend a system message if none exists. Does not modify original.
        """
        total = budget.get("total", MAX_CONTEXT_TOKENS)
        pct = budget.get("pct_used", 0)
        remaining = budget.get("remaining", 0)

        lines = [
            "## TOKEN BUDGET",
            f"- Context window: {total:,} tokens",
            f"- Current usage: {pct:.0%} (~{total - remaining:,} tokens used)",
            f"- Remaining: ~{remaining:,} tokens",
        ]

        if pct >= 0.80:
            lines.append(
                "\n## ⚠️ WARNING: Context is >80% full. "
                "Prefer summarization over verbose responses. "
                "Compress or omit redundant information."
            )

        if pct >= 0.95:
            lines.append(
                "\n## 🚨 CRITICAL: Context is >95% full. "
                "Summarize aggressively. Drop boilerplate, logs, and repeated text. "
                "Keep only decisions, key facts, and current task."
            )

        if large_chunks:
            total_chunk = sum(c.get("tokens", 0) for c in large_chunks)
            lines.append(
                f"\n## 📦 LARGE DATA DETECTED: {len(large_chunks)} block(s) "
                f"({total_chunk:,} tokens). When responding, do not reproduce "
                "all of the data — summarize, extract key points, or point to "
                "the relevant portion. Prefer compression/summarization."
            )

        notice = "\n".join(lines)
        messages = list(messages)  # shallow copy

        # Inject into existing system message
        system_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_idx = i
                break

        if system_idx is not None:
            original = messages[system_idx].get("content", "")
            messages[system_idx] = {
                **messages[system_idx],
                "content": original + "\n\n" + notice,
            }
        else:
            messages.insert(0, {"role": "system", "content": notice})

        return messages

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        """After each LLM response, asynchronously summarize the session."""
        call_type = data.get("call_type", "")
        if call_type not in ("acompletion", "completion"):
            return None

        call_id = data.get("_call_id") or _next_call_id()
        try:
            messages = data.get("messages", [])
            if len(messages) < 2:
                return None

            sid = self._session_id(data)
            response_text = ""
            try:
                choices = getattr(response, "choices", []) or []
                if choices:
                    message = getattr(choices[0], "message", None) or getattr(choices[0], "content", "")
                    if hasattr(message, "content"):
                        response_text = (message.content or "")
                    elif isinstance(message, str):
                        response_text = message
            except Exception as exc:
                logger.warning("Failed to extract response text: %s", exc)

            response_tokens = len(response_text) // 4
            _log(logging.DEBUG, call_id, "llm_post_call",
                         sid=sid, response_chars=len(response_text),
                         response_tokens=response_tokens,
                         response_preview=response_text[:300] if response_text else "")

            with _SESSION_LOCK:
                if sid not in _SESSION_HISTORY:
                    _SESSION_HISTORY[sid] = []
                hist = _SESSION_HISTORY[sid]
                if messages and messages[-1].get("role") == "user":
                    hist.append(dict(messages[-1]))
                if response_text:
                    hist.append({"role": "assistant", "content": response_text})
                if len(hist) > 40:
                    hist[:] = hist[-40:]

            if len(hist) >= 8:
                api_base = os.environ.get(
                    "LITE_LLM_SUMMARIZE_API_BASE",
                    "http://localhost:11112/v1/completions",
                )
                summary = await _async_summarize(hist, api_base)
                if summary:
                    with _SESSION_LOCK:
                        if len(hist) > 14:
                            summary_msg = {
                                "role": "system",
                                "content": f"[Session summary: {summary}]",
                            }
                            hist[:] = [summary_msg] + hist[-12:]
                    logger.info("Session summarized (%d msgs → short), sid=%s", len(hist), sid)

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("async_post_call_success_hook FAILED: %s\n%s", exc, tb)
            _log(logging.ERROR, call_id, "llm_post_call_error",
                         error=str(exc), error_type=type(exc).__name__, traceback=tb)

    def rewrite_messages(self, messages: list[dict], data: dict) -> list[dict]:
        """
        Pre-call: if we have a stored summary for this session, inject it
        as the first user message so the model gets the gist without re-reading.
        """
        sid = self._session_id(data)
        with _SESSION_LOCK:
            if sid in _SESSION_HISTORY and len(_SESSION_HISTORY[sid]) >= 8:
                hist = _SESSION_HISTORY[sid]
                summary_msgs = [m for m in hist if m.get("role") == "system"]
                if summary_msgs:
                    prefix = summary_msgs[0].get("content", "")
                    messages = [
                        {"role": "user", "content": f"[Prior context: {prefix}]\n\n"}
                        + (list(messages) if isinstance(messages, list) else messages)
                    ]
        return messages


# ── Singleton for LiteLLM callback registration ────────────────────────────────
_callback_instance: Qwen36CompressCallback | None = None


def get_callback() -> Qwen36CompressCallback:
    global _callback_instance
    if _callback_instance is None:
        _callback_instance = Qwen36CompressCallback()
    return _callback_instance


def register():
    """Register the callback with LiteLLM's global callback system."""
    cb = get_callback()
    litellm.callbacks.append(cb)
    litellm.success_callback.append(cb)
    logger.info(
        "Qwen36CompressCallback registered to litellm.callbacks "
        "(threshold=%d tokens, models=%s)",
        cb.threshold_tokens, cb.models,
    )


# ── Todo/Approval/Summary Prompt Injection ────────────────────────────────────

TODO_APPROVAL_SYSTEM_PROMPT = """[SYSTEM INSTRUCTION]
You are lunch-model assistant. When the user asks you to plan, figure out a plan, or explain how you would approach a task — detect this intent naturally without relying on specific keywords.

**RESPONSE FORMAT (mandatory):**
1. ANALYSIS: Write 2-3 sentences max. Be specific to this exact request — NOT a generic framework.
2. TODO LIST: Numbered steps specific to this task.
3. END: "Awaiting your approval to proceed..." then stop. Do NOT execute yet.
4. AFTER APPROVAL: Execute, then "Done: X / Not done: Y" then <|done|> then stop.

**ANTI-REPETITION (strict):**
- NEVER start with "Sure", "I'll", "Let me", "Here's", or similar generic openings.
- NEVER use the same sentence structure across different tasks.
- NEVER rephrase a point you already made.
- Each response must be different from the previous one.
- Be concise — stop the moment you have finished answering.
- <|done|> = stop immediately, nothing after.
"""


# ── Rules injected into the FIRST user message of a new conversation ───────────
FIRST_MSG_RULES = """
[IMPORTANT — RESPONSE RULES]
You are lunch-model assistant. For every task request you MUST:

**RESPONSE FORMAT (mandatory):**
1. ANALYSIS: Write 2-3 sentences max. Be specific to this exact request — NOT a generic framework.
2. TODO LIST: Numbered steps specific to this task.
3. END: "Awaiting your approval to proceed..." then stop. Do NOT execute yet.
4. AFTER APPROVAL: Execute, then "Done: X / Not done: Y" then <|done|> then stop.

**ANTI-REPETITION (strict):**
- NEVER start with "Sure", "I'll", "Let me", "Here's", or similar generic openings.
- NEVER use the same sentence structure across different tasks.
- NEVER rephrase a point you already made.
- Each response must be different from the previous one.
- Be concise — stop the moment you have finished answering.
- <|done|> = stop immediately, nothing after.
"""


class TodoApprovalPromptCallback(CustomLogger):
    """
    Injects todo/approval/summary instructions into every request for lunch-model.

    - On first user message of a conversation: injects full plan rules as user prefix
    - On subsequent messages: injects lightweight system prompt
    - Configurable via env vars to enable/disable
    """

    def __init__(
        self,
        enabled: bool | None = None,
        models: set[str] | None = None,
    ):
        super().__init__()
        self.enabled = (
            enabled
            if enabled is not None
            else os.environ.get("LITE_LLM_TODO_APPROVAL", "true").lower()
            not in ("false", "0", "no", "off")
        )
        self.models = models or set(
            m.strip()
            for m in os.environ.get(
                "LITE_LLM_TODO_APPROVAL_MODELS",
                "qwen3.6-35b-nvfp4",
            ).split(",")
            if m.strip()
        )
        self._seen_sessions: set[str] = set()
        self._seen_lock = threading.Lock()
        logger.info(
            "TodoApprovalPromptCallback init: enabled=%s, models=%s",
            self.enabled, self.models,
        )

    def _session_id(self, data: dict) -> str:
        import hashlib
        user = data.get("user", "default")
        model = data.get("model", "unknown")
        raw = f"{user}::{model}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _is_first_user_message(self, messages: list[dict]) -> bool:
        """Check if this is the first user message in this conversation."""
        user_msgs = [m for m in messages if m.get("role") == "user"]
        # If there's exactly 1 user message (the current one we're processing),
        # or user messages appear before any assistant messages, it's first
        if len(user_msgs) <= 1:
            return True
        # Check if there's any assistant message before the last user message
        for m in messages:
            if m.get("role") == "assistant":
                return False
            if m.get("role") == "user" and m is not messages[-1]:
                return False
        return True

    def _should_apply(self, model: str | None) -> bool:
        if not self.enabled:
            return False
        if not model:
            return False
        model_lower = model.lower()
        return any(
            m.lower() in model_lower or model_lower in m.lower()
            for m in self.models
        )

    def _extract_last_user_message(
        self, messages: list[dict]
    ) -> tuple[list[dict], str]:
        """Pop the last user message from the list, return (remaining, content)."""
        messages = list(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                content = messages[i].get("content", "")
                # Normalize list content (e.g. multimodal) to plain text
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") or str(p)
                        for p in content if isinstance(p, dict)
                    )
                elif not isinstance(content, str):
                    content = str(content)
                del messages[i]
                return messages, content
        return messages, ""

    def _count_tokens(self, messages: list[dict]) -> int:
        """Estimate token count for a messages list."""
        global _CACHED_TOKENIZER
        if _CACHED_TOKENIZER is None:
            try:
                from transformers import AutoTokenizer
                _CACHED_TOKENIZER = AutoTokenizer.from_pretrained(
                    "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
                    use_fast=True,
                    trust_remote_code=True,
                )
            except Exception:
                return 0
        total = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") or str(p) for p in content if isinstance(p, dict))
            elif not isinstance(content, str):
                content = str(content)
            try:
                total += len(_CACHED_TOKENIZER.encode(content, add_special_tokens=False))
            except Exception:
                pass
        return total

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        if call_type not in ("acompletion", "completion"):
            return None

        model = data.get("model")
        if not self._should_apply(model):
            return None

        messages = data.get("messages", [])
        if not messages:
            return None

        sid = self._session_id(data)

        with self._seen_lock:
            is_new = sid not in self._seen_sessions
            if is_new:
                self._seen_sessions.add(sid)

        # Token count BEFORE injection
        tokens_before = self._count_tokens(messages)

        # Extract last user message for wrapping
        messages_copy = list(messages)
        last_user_idx = None
        for i in range(len(messages_copy) - 1, -1, -1):
            if messages_copy[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return None

        content = messages_copy[last_user_idx].get("content", "")
        # Normalize list content (e.g. multimodal) to plain text
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") or str(p)
                for p in content if isinstance(p, dict)
            )
        elif not isinstance(content, str):
            content = str(content)

        # Build new messages
        new_messages = []

        # System prompt (always lightweight — intent-based plan trigger)
        new_messages.append({"role": "system", "content": TODO_APPROVAL_SYSTEM_PROMPT})

        # Existing messages except last user
        for i, m in enumerate(messages_copy):
            if i == last_user_idx:
                continue
            new_messages.append(m)

        # Last user message: wrap with rules on first turn, plain on rest
        if is_new:
            new_messages.append({"role": "user", "content": FIRST_MSG_RULES + content})
        else:
            new_messages.append({"role": "user", "content": content})

        # Token count AFTER injection
        tokens_after = self._count_tokens(new_messages)

        # Log token usage
        _log(
            logging.INFO, sid or "unknown", "todo_tokens",
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            injection_overhead=tokens_after - tokens_before,
            is_new_session=is_new,
            model=model,
            msg_count=len(messages),
        )
        logger.info(
            "TOKEN USAGE  model=%s  msgs=%d  tokens=%d/%d  overhead=%d (%s)",
            model, len(messages), tokens_before, tokens_after,
            tokens_after - tokens_before, "new" if is_new else "continue",
        )

        # Inject token info so LLM outputs it visibly in its response
        token_info = (
            f"\n\n[CONTEXT INFO — include in your response footer]\n"
            f"Tokens used: ~{tokens_before:,} / {tokens_after:,} | "
            f"Context window: 262,144 | "
            f"Remaining: ~{max(262144 - tokens_after, 0):,}"
        )
        system_with_tokens = TODO_APPROVAL_SYSTEM_PROMPT + token_info

        # Rebuild with token info in system prompt, keep all other messages
        final_messages = []
        final_messages.append({"role": "system", "content": system_with_tokens})
        # Re-add existing messages except the original last user (we'll re-add it)
        for i, m in enumerate(messages_copy):
            if i == last_user_idx:
                continue
            final_messages.append(m)
        # Re-add user message with or without rules wrapper
        if is_new:
            final_messages.append({"role": "user", "content": FIRST_MSG_RULES + content})
        else:
            final_messages.append({"role": "user", "content": content})

        data = dict(data)
        data["messages"] = final_messages
        logger.debug(
            "TodoApprovalPrompt injected (new=%s) for model=%s (%d → %d msgs)",
            is_new, model, len(messages), len(final_messages),
        )
        return data


# ── Singleton for TodoApprovalPromptCallback ───────────────────────────────────

_todo_callback_instance: TodoApprovalPromptCallback | None = None


def get_todo_callback() -> TodoApprovalPromptCallback:
    global _todo_callback_instance
    if _todo_callback_instance is None:
        _todo_callback_instance = TodoApprovalPromptCallback()
    return _todo_callback_instance


def register_todo_callback():
    """Register the todo/approval prompt callback with LiteLLM."""
    cb = get_todo_callback()
    litellm.callbacks.append(cb)
    litellm.success_callback.append(cb)
    logger.info(
        "TodoApprovalPromptCallback registered (enabled=%s, models=%s)",
        cb.enabled, cb.models,
    )
