"""
Qwen3.6-35B-A3B history sanitization callback for LiteLLM.

Intercepts requests before they are forwarded to the LLM and:
  1. Strips stored thinking token blocks (<think>...) from assistant
     history — prevents the model re-entering thinking mode on every turn.
  2. Compresses old tool result messages to reduce context accumulation.
  3. Trims oldest conversation exchanges when the estimated token count
     would exceed the model's context window.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

import litellm
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("qwen_compress")

# Matches Qwen's thinking token blocks.
_THINKING_RE = re.compile(
    r'(<\|begin_of_thought\|>.*?<\|end_of_thought\|>|<think>.*?)',
    re.DOTALL | re.IGNORECASE,
)

TOOL_RESULT_MAX_CHARS = 400
_COMPRESSED_MARKER_RE = re.compile(r'\[… \d+ chars omitted\]')

TOOL_DESC_MAX_CHARS = 280
TOOL_PARAM_DESC_MAX_CHARS = 120
_TOOL_SCHEMA_DROP_FIELDS = frozenset({"examples", "x-ms-docs", "deprecated", "additionalProperties"})

CONTEXT_LIMIT_TOKENS = 256_000
CONTEXT_TRIM_BUFFER = 3_000
_CHARS_PER_TOKEN = 2.5
_DEFAULT_MAX_OUTPUT_TOKENS = 4_096
_OVERFLOW_SESSION_PATH = Path(__file__).parent / "logs" / "last_session.json"


def _compress_param_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    out: dict = {}
    for k, v in schema.items():
        if k in _TOOL_SCHEMA_DROP_FIELDS:
            continue
        if k == "description" and isinstance(v, str) and len(v) > TOOL_PARAM_DESC_MAX_CHARS:
            v = v[:TOOL_PARAM_DESC_MAX_CHARS] + "…"
        elif k == "properties" and isinstance(v, dict):
            v = {pk: _compress_param_schema(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            v = _compress_param_schema(v)
        out[k] = v
    return out


def _compress_tool_schemas(tools: list) -> tuple[list, bool]:
    if not tools:
        return tools, False
    compressed = []
    changed = False
    for tool in tools:
        if not isinstance(tool, dict):
            compressed.append(tool)
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            compressed.append(tool)
            continue
        new_fn = dict(fn)
        did_change = False
        desc = fn.get("description", "")
        if isinstance(desc, str) and len(desc) > TOOL_DESC_MAX_CHARS:
            new_fn["description"] = desc[:TOOL_DESC_MAX_CHARS] + "…"
            did_change = True
        params = fn.get("parameters")
        if isinstance(params, dict):
            new_params = _compress_param_schema(params)
            if new_params != params:
                new_fn["parameters"] = new_params
                did_change = True
        if did_change:
            compressed.append({**tool, "function": new_fn})
            changed = True
        else:
            compressed.append(tool)
    return compressed, changed


def _strip_thinking_tokens(text: str) -> str:
    return _THINKING_RE.sub('', text).strip()


def _compress_tool_result_text(text: str) -> str:
    if len(text) <= TOOL_RESULT_MAX_CHARS:
        return text
    if _COMPRESSED_MARKER_RE.search(text):
        return text
    original_len = len(text)
    stripped = text.strip()
    if stripped.startswith(('{', '[')):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return (
                    f"[tool result omitted: JSON object, keys={list(obj.keys())}, "
                    f"{original_len} chars]"
                )
            if isinstance(obj, list):
                return (
                    f"[tool result omitted: JSON array, {len(obj)} items, "
                    f"{original_len} chars]"
                )
        except (json.JSONDecodeError, ValueError):
            pass
    return text[:TOOL_RESULT_MAX_CHARS] + f"\n[… {original_len - TOOL_RESULT_MAX_CHARS} chars omitted]"


def _compress_content(content: Any) -> tuple[Any, bool]:
    if isinstance(content, str):
        compressed = _compress_tool_result_text(content)
        return compressed, compressed != content
    if isinstance(content, list):
        if not all(isinstance(p, dict) and p.get("type") == "text" for p in content):
            return content, False
        new_parts: list[dict] = []
        changed = False
        for part in content:
            text = part.get("text", "")
            compressed = _compress_tool_result_text(text)
            if compressed != text:
                new_parts.append({**part, "text": compressed})
                changed = True
            else:
                new_parts.append(part)
        return new_parts, changed
    return content, False


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _message_token_estimate(msg: dict) -> int:
    content = msg.get("content") or ""
    if isinstance(content, str):
        base = _estimate_tokens(content)
    elif isinstance(content, list):
        base = sum(
            _estimate_tokens(p.get("text", "") if isinstance(p, dict) else str(p))
            for p in content
        )
    else:
        base = 4
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        try:
            base += _estimate_tokens(json.dumps(tool_calls))
        except Exception:
            base += 50
    return base + 8


def _messages_token_estimate(messages: list) -> int:
    return sum(_message_token_estimate(m) for m in messages if isinstance(m, dict))


def _tools_token_estimate(tools: Any) -> int:
    if not tools:
        return 0
    try:
        return _estimate_tokens(json.dumps(tools, ensure_ascii=False))
    except Exception:
        return 0


def _trim_to_context(messages: list, token_budget: int) -> tuple[list, bool]:
    total = _messages_token_estimate(messages)
    if total <= token_budget:
        return messages, False

    head: list = []
    idx = 0
    while (
        idx < len(messages)
        and isinstance(messages[idx], dict)
        and messages[idx].get("role") == "system"
    ):
        head.append(messages[idx])
        idx += 1

    last_user_idx = -1
    for j in range(len(messages) - 1, idx - 1, -1):
        if isinstance(messages[j], dict) and messages[j].get("role") == "user":
            last_user_idx = j
            break

    if last_user_idx < idx:
        return messages, False

    current_turn = list(messages[last_user_idx:])
    middle = list(messages[idx:last_user_idx])

    head_tokens = _messages_token_estimate(head)
    current_tokens = _messages_token_estimate(current_turn)
    middle_tokens = total - head_tokens - current_tokens

    was_changed = False

    while middle and (head_tokens + middle_tokens + current_tokens) > token_budget:
        first_user = next(
            (k for k, m in enumerate(middle) if isinstance(m, dict) and m.get("role") == "user"),
            -1,
        )

        if first_user == -1:
            middle_tokens = 0
            middle = []
            was_changed = True
            break

        if first_user > 0:
            chunk = middle[:first_user]
            middle_tokens -= _message_token_estimate(chunk)
            middle = middle[first_user:]
            was_changed = True
            continue

        drop_until = len(middle)
        for k in range(1, len(middle)):
            if isinstance(middle[k], dict) and middle[k].get("role") == "user":
                drop_until = k
                break

        chunk = middle[:drop_until]
        middle = middle[drop_until:]
        middle_tokens -= _message_token_estimate(chunk)
        was_changed = True

        logger.warning(
            "Context trim: dropped oldest exchange. "
            "Remaining: ~%d tokens (budget=%d)",
            head_tokens + middle_tokens + current_tokens,
            token_budget,
        )

    if (head_tokens + middle_tokens + current_tokens) > token_budget:
        logger.warning(
            "Context trim: history fully cleared but system+current_turn "
            "(~%d tokens) still exceeds budget (~%d tokens).",
            head_tokens + current_tokens,
            token_budget,
        )

    return head + middle + current_turn, was_changed


def _extract_msg_text(msg: dict) -> str:
    content = msg.get("content") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _save_overflow_session(messages: list, tokens: int) -> None:
    try:
        _OVERFLOW_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        user_turns = [
            _extract_msg_text(m)[:400]
            for m in messages if isinstance(m, dict) and m.get("role") == "user"
        ]
        asst_turns = [
            _extract_msg_text(m)[:400]
            for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
        ]
        _OVERFLOW_SESSION_PATH.write_text(json.dumps({
            "timestamp": "",
            "message_count": len(messages),
            "estimated_tokens": tokens,
            "user_turns": user_turns,
            "assistant_turns_last3": asst_turns[-3:],
        }, ensure_ascii=False, indent=2))
        logger.info("Overflow session saved → %s", _OVERFLOW_SESSION_PATH)
    except Exception as exc:
        logger.warning("Failed to save overflow session: %s", exc)


def _make_overflow_response(messages: list, tokens: int) -> str:
    user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    first = _extract_msg_text(user_msgs[0])[:280].strip() if user_msgs else ""
    recent = [
        _extract_msg_text(m)[:130].strip()
        for m in user_msgs[-4:-1]
        if isinstance(m, dict)
    ]
    recent = [r for r in recent if r]

    lines = [
        f"⚠️ **Context window limit reached** (~{tokens:,} / {CONTEXT_LIMIT_TOKENS:,} tokens)\n\n",
        "All available history has been trimmed but the session was still too large. "
        f"A snapshot has been saved to `logs/last_session.json`.\n",
    ]
    if first:
        lines.append(f"\n**Session started with:**\n> {first}…\n")
    if recent:
        lines.append("\n**Recent topics discussed:**")
        for r in recent:
            lines.append(f"\n> {r}…")
        lines.append("\n")
    lines.append(
        "\n---\n"
        "**Fresh session started automatically.** "
        "Just continue — describe what you need and I'll pick up from here."
    )
    return "".join(lines)


class QwenHistorySanitizer(CustomLogger):
    """Strip thinking tokens, compress tool results, and trim context window."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        if call_type not in ("acompletion", "completion"):
            return None

        messages = data.get("messages", [])
        if isinstance(messages, str):
            return None

        last_user_idx = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user_idx = i

        sanitized: list[Any] = []
        changed = False

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                sanitized.append(msg)
                continue

            role = msg.get("role")

            if role == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    stripped = _strip_thinking_tokens(content)
                    if stripped != content:
                        msg = {**msg, "content": stripped}
                        changed = True

            elif role == "tool" and i < last_user_idx:
                content = msg.get("content")
                new_content, did_change = _compress_content(content)
                if did_change:
                    msg = {**msg, "content": new_content}
                    changed = True

            sanitized.append(msg)

        if changed:
            data = {**data, "messages": sanitized}

        tools = data.get("tools") or []
        if tools:
            compressed_tools, tools_changed = _compress_tool_schemas(tools)
            if tools_changed:
                data = {**data, "tools": compressed_tools}
                tools = compressed_tools

        tools_tokens = _tools_token_estimate(tools)
        max_output = int(data.get("max_tokens") or _DEFAULT_MAX_OUTPUT_TOKENS)
        output_reserve = min(max_output, CONTEXT_LIMIT_TOKENS // 2)
        token_budget = CONTEXT_LIMIT_TOKENS - CONTEXT_TRIM_BUFFER - tools_tokens - output_reserve

        before_tokens = _messages_token_estimate(sanitized)
        if before_tokens > token_budget:
            trimmed, did_trim = _trim_to_context(sanitized, token_budget)
            if did_trim:
                after_tokens = _messages_token_estimate(trimmed)
                logger.warning(
                    "Context window trim: %d→%d messages, ~%d→~%d estimated tokens",
                    len(sanitized), len(trimmed),
                    before_tokens, after_tokens,
                )
                sanitized = trimmed
                changed = True

                if after_tokens > token_budget:
                    _save_overflow_session(data.get("messages", []), before_tokens)

        if changed:
            return data
        return None


_callback_instance: QwenHistorySanitizer | None = None


def get_callback() -> QwenHistorySanitizer:
    global _callback_instance
    if _callback_instance is None:
        _callback_instance = QwenHistorySanitizer()
    return _callback_instance


def register():
    cb = get_callback()
    litellm.callbacks.append(cb)
    litellm.success_callback.append(cb)
    logger.info("QwenHistorySanitizer registered to litellm.callbacks")