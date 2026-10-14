"""上下文工具的终端摘要；只读取事件，不参与上下文管理。"""

import json
from collections.abc import Mapping

from rich.console import Group, RenderableType
from rich.text import Text

from praxis_agent.context.capacity import ContextStatus
from praxis_agent.terminal.cells import tool_activity_cell, tool_completed_cell


CONTEXT_TOOLS = {"delete", "compress", "search_history", "read_history"}
_ACTIONS = {
    "delete": "Hiding messages",
    "compress": "Summarizing context",
    "search_history": "Searching history",
    "read_history": "Reading history",
}


def _short(value: object, limit: int = 72) -> str:
    text = " ".join(str(value).split())
    text = "".join(char for char in text if char.isprintable())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _ids(arguments: Mapping[str, object]) -> list[str]:
    values = arguments.get("message_ids", [])
    return list(dict.fromkeys(value for value in values if isinstance(value, str))) if isinstance(values, list) else []


def _json_object(output: str) -> dict[str, object]:
    try:
        value = json.loads(output)
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def context_activity(name: str, arguments: Mapping[str, object]) -> RenderableType:
    if name in {"delete", "compress"}:
        detail = f"{len(_ids(arguments))} messages"
    elif name == "search_history":
        detail = f'“{_short(arguments.get("query", ""))}”'
    else:
        detail = _short(arguments.get("message_id", ""), 24)
    return tool_activity_cell(_ACTIONS[name], detail)


def context_completed(
    name: str, arguments: Mapping[str, object], output: str,
) -> RenderableType:
    result = _json_object(output)
    if not result:
        return tool_completed_cell(f"Completed {name.replace('_', ' ')}")
    details: list[str] = []
    if name in {"delete", "compress"}:
        ids = _ids(arguments)
        verb = "Summarized" if name == "compress" else "Hidden"
        title = f"{verb} {len(ids)} {'message' if len(ids) == 1 else 'messages'}"
        sources = ", ".join(_short(mid, 18) for mid in ids[:3])
        if len(ids) > 3:
            sources += f" · +{len(ids) - 3} more"
        details.append(f"{sources} → {_short(result.get('message_id', ''), 18)}")
        details.append("Original history retained · applies next request")
    elif name == "search_history":
        matches = result.get("matches", [])
        matches = matches if isinstance(matches, list) else []
        count = len(matches)
        title = f"Searched history · {count}{'+' if result.get('truncated') else ''} {'match' if count == 1 else 'matches'}"
        details.append(f'“{_short(arguments.get("query", ""))}”')
        if not matches:
            details.append("No matches · try a different keyword")
        for match in matches[:3]:
            if isinstance(match, dict):
                details.append(f"{_short(match.get('message_id', ''), 18)} · offset {_short(match.get('offset', '?'), 12)}")
        if count > 3:
            details.append(f"+{count - 3} more matches returned to the model")
    else:
        title = "Read history"
        start, end = result.get("offset"), result.get("next_offset")
        total = result.get("total_length")
        details.append(_short(result.get("message_id", ""), 24))
        if all(isinstance(value, int) for value in (start, end, total)):
            details.append(f"{end - start:,} characters · [{start:,}, {end:,}) of {total:,}")
        details.append("End of message" if result.get("eof") else f"Continue at offset {_short(end, 12)}")
    lines = []
    for index, detail in enumerate(details):
        line = Text("  └ " if index == 0 else "    ", style="dim")
        line.append(detail, style="cyan" if index == 0 else "dim")
        lines.append(line)
    return Group(tool_completed_cell(title), *lines)


def capacity_footer(status: ContextStatus | None) -> Text | None:
    """展示本次请求已有的容量估算，不解析模型提示词或重复计算。"""
    if status is None:
        return None
    line = Text("  上下文占用 ≈ ", style="dim", no_wrap=True, overflow="ellipsis")
    line.append(f"{status.usage_percent:g}%", style="cyan")
    line.append(" · 最近请求")
    return line
