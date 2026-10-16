"""上下文工具的终端摘要；只读取事件，不参与上下文管理。"""

from collections.abc import Mapping

from rich.console import RenderableType
from rich.text import Text

from praxis_agent.context.capacity import ContextStatus
from praxis_agent.terminal.cells import tool_activity_cell


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


def context_activity(name: str, arguments: Mapping[str, object]) -> RenderableType:
    if name in {"delete", "compress"}:
        detail = f"{len(_ids(arguments))} messages"
    elif name == "search_history":
        detail = f'“{_short(arguments.get("query", ""))}”'
    else:
        detail = _short(arguments.get("message_id", ""), 24)
    return tool_activity_cell(_ACTIONS[name], detail)


def capacity_footer(status: ContextStatus | None) -> Text | None:
    """展示本次请求已有的容量估算，不解析模型提示词或重复计算。"""
    if status is None:
        return None
    line = Text("  上下文占用 ≈ ", style="dim", no_wrap=True, overflow="ellipsis")
    line.append(f"{status.usage_percent:g}%", style="cyan")
    line.append(" · 最近请求")
    return line
