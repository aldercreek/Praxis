"""上下文工具的终端摘要；只读取事件，不参与上下文管理。"""

from rich.text import Text

from praxis_agent.context.capacity import ContextStatus


def capacity_footer(status: ContextStatus | None) -> Text | None:
    """展示本次请求已有的容量估算，不解析模型提示词或重复计算。"""
    if status is None:
        return None
    line = Text("Context ≈ ", no_wrap=True, overflow="ellipsis")
    line.append(f"{status.usage_percent:g}%", style="cyan")
    return line
