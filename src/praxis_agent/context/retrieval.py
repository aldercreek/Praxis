"""按消息正文的字符位置搜索、读取历史，不改变上下文操作。"""

from collections.abc import Iterator, Sequence
import re
import json

from praxis_agent.context.models import ConversationHistory
from praxis_agent.context.budgets import DEFAULT_BUDGETS
from praxis_agent.context.operations import ContextOperation, item_payload


def _body(payload: dict[str, object]) -> str | None:
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


def _messages(
    history: ConversationHistory, operations: Sequence[ContextOperation],
) -> Iterator[tuple[str, str]]:
    # 先查替代消息，再查原始历史；两组内部都从新到旧。
    for operation in reversed(operations):
        replacement = operation.replacement
        if replacement is not None:
            text = _body(replacement["payload"])
            if text is not None:
                yield replacement["message_id"], text
    items = history.snapshot()
    for index in range(len(items) - 1, -1, -1):
        text = _body(item_payload(items[index]))
        if text is not None:
            yield f"m{index + 1}", text


def search_history(
    history: ConversationHistory, operations: Sequence[ContextOperation],
    query: str, max_results: int = 10,
) -> dict[str, object]:
    """字面匹配返回附近正文；offset 是片段起点，不是命中词起点。"""
    if not isinstance(query, str) or not 1 <= len(query) <= 200:
        raise ValueError("query must contain 1-200 characters")
    if not isinstance(max_results, int) or not 1 <= max_results <= 20:
        raise ValueError("max_results must be between 1 and 20")
    # 不先 lower/casefold 正文，避免 Unicode 转换改变字符位置。
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    matches = []
    for message_id, text in _messages(history, operations):
        for match in pattern.finditer(text):
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
        end = len(text) if end_offset is None else end_offset
        if start_offset > len(text) or end > len(text):
            raise ValueError(f"range exceeds message length {len(text)}")

        def render(stop):
            truncated = stop < end
            return {
                "text": text[start_offset:stop],
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
