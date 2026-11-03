"""按消息正文的字符位置搜索、读取历史，不改变上下文操作。"""

from collections.abc import Iterator, Sequence
import re
import json

from praxis_agent.context.models import ConversationHistory
from praxis_agent.context.budgets import DEFAULT_BUDGETS
from praxis_agent.context.operations import ContextOperation, item_payload


_HISTORY_TOOLS = {"search_history", "read_history"}


def _only_history_calls(output: object) -> bool:
    """A run_code result that only retrieves history is another copy of that history."""
    try:
        result = json.loads(json.loads(output)["output"])
    except (ValueError, TypeError, KeyError):
        return False
    calls = result.get("calls") if isinstance(result, dict) else None
    return isinstance(calls, list) and bool(calls) and all(
        isinstance(call, dict) and call.get("tool") in _HISTORY_TOOLS for call in calls
    )


def _body(payload: dict[str, object]) -> str | None:
    if payload.get("type") == "function_call":
        return payload.get("arguments")
    if payload.get("type") == "function_call_output":
        content = payload.get("output")
    elif payload.get("role") in {"user", "assistant"}:
        content = payload.get("content")
    else:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 多文本块直接连接，不改写空白；搜索和读取共用这份正文。
        return "".join(
            block["text"] for block in content
            if isinstance(block.get("text"), str)
        )
    return None


def _payloads(
    history: ConversationHistory, operations: Sequence[ContextOperation],
) -> Iterator[tuple[str, dict[str, object]]]:
    # 先查替代消息，再查原始历史；两组内部都从新到旧。
    for operation in reversed(operations):
        replacement = operation.replacement
        if replacement is not None:
            yield replacement["message_id"], replacement["payload"]
    items = history.snapshot()
    for index in range(len(items) - 1, -1, -1):
        yield f"m{index + 1}", item_payload(items[index])


def _messages(history, operations) -> Iterator[tuple[str, str]]:
    for message_id, payload in _payloads(history, operations):
        text = _body(payload)
        if text is not None:
            yield message_id, text


def read_history_media(history, operations, message_id: str) -> list[dict[str, object]]:
    """Restore original media blocks without serializing their binary data as text."""
    for current_id, payload in _payloads(history, operations):
        if current_id == message_id:
            content = payload.get("output" if payload.get("type") == "function_call_output" else "content")
            return [dict(block) for block in content
                    if block.get("type") in {"input_image", "input_file"}] if isinstance(content, list) else []
    raise ValueError(f"message not found: {message_id}")


def search_history(
    history: ConversationHistory, operations: Sequence[ContextOperation],
    query: str, max_results: int = 10,
) -> dict[str, object]:
    """每条来源返回首个命中，跳过历史回查本身；offset 是片段起点。"""
    if not isinstance(query, str) or not 1 <= len(query) <= 200:
        raise ValueError("query must contain 1-200 characters")
    if not isinstance(max_results, int) or not 1 <= max_results <= 20:
        raise ValueError("max_results must be between 1 and 20")
    # 不先 lower/casefold 正文，避免 Unicode 转换改变字符位置。
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    payloads = list(_payloads(history, operations))
    calls = {payload["call_id"]: payload.get("name") for _, payload in payloads
             if payload.get("type") == "function_call" and payload.get("call_id")}
    completed = {payload.get("call_id") for _, payload in payloads
                 if payload.get("type") == "function_call_output"}
    retrieval_calls = {
        call_id for call_id, name in calls.items() if name in _HISTORY_TOOLS or call_id not in completed
    }
    retrieval_calls.update(
        payload["call_id"] for _, payload in payloads
        if payload.get("type") == "function_call_output"
        and calls.get(payload.get("call_id")) == "run_code"
        and _only_history_calls(payload.get("output"))
    )
    matches = []
    for message_id, payload in payloads:
        if payload.get("call_id") in retrieval_calls:
            continue
        text = _body(payload)
        match = pattern.search(text) if text is not None else None
        if match is None:
            continue
        if len(matches) == max_results:
            return {"matches": matches, "truncated": True}
        offset = max(0, match.start() - 80)
        end = min(len(text), match.end() + 80)
        matches.append({
            "message_id": message_id,
            "offset": offset,
            "text": text[offset:end],
        })
    return {"matches": matches, "truncated": False}


def read_history(
    history: ConversationHistory, operations: Sequence[ContextOperation],
    message_id: str, start_offset: int = 0, end_offset: int | None = None,
    *, character_limit: int | None = DEFAULT_BUDGETS["max_history_read_chars"],
) -> dict[str, object]:
    """Read a half-open character range, budgeting the complete model-facing JSON."""
    if isinstance(start_offset, bool) or not isinstance(start_offset, int) or start_offset < 0:
        raise ValueError("start_offset must be a non-negative integer")
    if end_offset is not None and (
        isinstance(end_offset, bool) or not isinstance(end_offset, int) or end_offset < start_offset
    ):
        raise ValueError("end_offset must be null or an integer at least start_offset")
    for current_id, text in _messages(history, operations):
        if current_id != message_id:
            continue
        end = len(text) if end_offset is None else min(end_offset, len(text))
        if start_offset > len(text):
            raise ValueError(f"range exceeds message length {len(text)}")

        def render(stop):
            truncated = stop < end
            return {
                "text": text[start_offset:stop],
                "total_chars": len(text),
                "truncated": truncated,
                "next_offset": stop if truncated else None,
                "note": ("Only part of the requested range is shown. Continue with start_offset=next_offset "
                         "and the same end_offset; omitted end means end of message.") if truncated else "",
            }

        if character_limit is None:
            return render(end)
        if character_limit < 1:
            raise ValueError("character_limit must be positive")
        candidate = render(min(end, start_offset + character_limit))
        if len(json.dumps(candidate, ensure_ascii=False)) <= character_limit:
            return candidate
        low, high = start_offset, min(end - 1, start_offset + character_limit)
        while low < high:
            middle = (low + high + 1) // 2
            if len(json.dumps(render(middle), ensure_ascii=False)) <= character_limit:
                low = middle
            else:
                high = middle - 1
        result = render(low)
        if (low == start_offset and start_offset < end) or len(json.dumps(result, ensure_ascii=False)) > character_limit:
            raise ValueError("Read budget cannot fit metadata and text; increase the output budget")
        return result
    raise ValueError(f"text message not found: {message_id}")
