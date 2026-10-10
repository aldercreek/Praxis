"""按消息正文的字符位置搜索、读取历史，不改变上下文操作。"""

from collections.abc import Iterator, Sequence
import re

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
    message_id: str, offset: int = 0, length: int = 8000,
    *, max_length: int = DEFAULT_BUDGETS["max_history_read_chars"],
) -> dict[str, object]:
    """读取 Python 字符串切片；next_offset 可直接用于下一次调用。"""
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if not isinstance(length, int) or not 1 <= length <= max_length:
        raise ValueError(f"length must be between 1 and {max_length}")
    for current_id, text in _messages(history, operations):
        if current_id != message_id:
            continue
        if offset > len(text):
            raise ValueError(f"offset exceeds total_length {len(text)}")
        end = min(len(text), offset + length)
        return {
            "message_id": message_id,
            "offset": offset,
            "text": text[offset:end],
            "next_offset": end,
            "total_length": len(text),
            "eof": end == len(text),
        }
    raise ValueError(f"text message not found: {message_id}")
