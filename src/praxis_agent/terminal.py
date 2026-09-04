import json
from collections.abc import Mapping

from praxis_agent.events import AgentEvent, ToolCallPrepared, ToolFinished


_MAX_ARGUMENT_DISPLAY_CHARS = 500
_MAX_RESULT_DISPLAY_CHARS = 4_000


def _truncate_for_display(text: str, max_chars: int) -> str:
    """限制终端展示长度，同时保留输出的开头和结尾。"""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + "\n... terminal output truncated ...\n"
        + text[-half:]
    )


class ConsoleRenderer:
    """把 Agent 事件转换成当前版本的终端文本。"""

    def emit(self, event: AgentEvent) -> None:
        """只展示工具事件，其他事件先作为扩展点保留。"""
        if isinstance(event, ToolCallPrepared):
            encoded_arguments = json.dumps(
                dict(event.arguments),
                ensure_ascii=False,
            )
            visible_arguments = _truncate_for_display(
                encoded_arguments,
                _MAX_ARGUMENT_DISPLAY_CHARS,
            )
            print(f"[tool] {event.tool_name} {visible_arguments}")
            return

        if isinstance(event, ToolFinished):
            status = "ok" if event.result.success else "error"
            print(f"[tool:{status}] {event.tool_name}")
            if event.tool_name == "run_command":
                print(
                    _truncate_for_display(
                        event.result.output,
                        _MAX_RESULT_DISPLAY_CHARS,
                    )
                )


def approve_tool_call(
    name: str,
    arguments: Mapping[str, object],
) -> bool:
    """询问用户是否允许执行需要审批的工具。"""
    del arguments
    try:
        answer = input(f"Allow {name}? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}
