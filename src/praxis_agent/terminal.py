import json
from collections.abc import Mapping

from praxis_agent.tools import ToolResult


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


def show_tool_call(
    name: str,
    arguments: Mapping[str, object],
) -> None:
    """在终端展示模型准备调用的工具和参数。"""
    encoded_arguments = json.dumps(dict(arguments), ensure_ascii=False)
    visible_arguments = _truncate_for_display(
        encoded_arguments,
        _MAX_ARGUMENT_DISPLAY_CHARS,
    )
    print(f"[tool] {name} {visible_arguments}")


def show_tool_result(name: str, result: ToolResult) -> None:
    """展示工具状态；命令工具还会展示受限长度的运行输出。"""
    status = "ok" if result.success else "error"
    print(f"[tool:{status}] {name}")
    if name == "run_command":
        print(
            _truncate_for_display(
                result.output,
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
