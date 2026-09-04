import json
from collections.abc import Mapping

from praxis_agent.events import (
    AgentEvent,
    AssistantTextCompleted,
    AssistantTextDelta,
    ModelRequestStarted,
    ToolCallPrepared,
    ToolFinished,
    TurnFailed,
)


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

    def __init__(self) -> None:
        self._request_has_text = False
        self._line_open = False

    def emit(self, event: AgentEvent) -> None:
        """按照事件类型更新当前终端输出。"""
        if isinstance(event, ModelRequestStarted):
            self._request_has_text = False
            return

        if isinstance(event, AssistantTextDelta):
            self._write_text_delta(event.text)
            return

        if isinstance(event, AssistantTextCompleted):
            if self._request_has_text:
                self._finish_open_line()
            else:
                print(event.text)
            return

        if isinstance(event, ToolCallPrepared):
            self._finish_open_line()
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
            return

        if isinstance(event, TurnFailed):
            self._finish_open_line()

    def _write_text_delta(self, text: str) -> None:
        """立即展示一段模型文本，并记录当前光标是否仍在行中。"""
        if not text:
            return
        print(text, end="", flush=True)
        self._request_has_text = True
        self._line_open = not text.endswith(("\n", "\r"))

    def _finish_open_line(self) -> None:
        """必要时补充换行，避免后续状态文字粘在模型文本后面。"""
        if self._line_open:
            print()
            self._line_open = False


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
