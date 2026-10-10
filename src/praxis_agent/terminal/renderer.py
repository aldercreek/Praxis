from __future__ import annotations

import os
import shlex
import subprocess
from time import monotonic
from collections.abc import Mapping
from dataclasses import dataclass

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.text import Text

from praxis_agent.events import (
    AgentEvent,
    ChildAgentEvent,
    ChildAgentStatus,
    ChildAgentMessage,
    ContextCompaction,
    ApprovalRequested,
    ApprovalResolved,
    AssistantTextCompleted,
    AssistantTextDelta,
    ModelRequestCompleted,
    ModelRequestStarted,
    PlanUpdated,
    ToolCallPrepared,
    ToolFinished,
    ToolStarted,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
)
from praxis_agent.terminal.cells import (
    ExplorationItem,
    assistant_cell,
    command_cell,
    error_cell,
    exploration_cell,
    file_change_cell,
    plan_cell,
    tool_activity_cell,
    tool_completed_cell,
    working_cell,
)
from praxis_agent.terminal.context import (
    CONTEXT_TOOLS,
    capacity_footer,
    context_activity,
    context_completed,
)


_EXPLORATION_TOOLS = frozenset(
    {"read_file", "list_files", "find_files", "search_text"}
)


@dataclass(frozen=True, slots=True)
class _PreparedCall:
    name: str
    arguments: dict[str, object]


class ConsoleRenderer:
    """Render agent events as a Codex-like terminal transcript."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._interactive = self._console.is_terminal
        self._live: Live | None = None
        self._prepared_calls: dict[str, _PreparedCall] = {}
        self._exploration_items: list[ExplorationItem] = []
        self._assistant_buffer = ""
        self._assistant_committed = False
        self._request_has_text = False
        self._line_open = False
        self._capacity: RenderableType | None = None
        self._turn_started = monotonic()
        self._phase_started = monotonic()
        self._phase = "等待模型响应"
        self._active: RenderableType = ""
        self._tool_started: dict[str, float] = {}
        self._plan_seen = False
        self._streamed_length = 0
        self._fence: str | None = None

    def emit(self, event: AgentEvent) -> None:
        """Update the transcript and active area for one agent event."""
        if isinstance(event, ChildAgentEvent):
            # Detailed worker events belong to tracing, never to the root text buffer.
            return
        if isinstance(event, (ChildAgentStatus, ChildAgentMessage)):
            self._finish_plain_line()
            self._flush_exploration()
            label = event.agent_id[:14]
            if isinstance(event, ChildAgentStatus):
                statuses = {"queued": "排队中", "running": "运行中", "idle": "空闲",
                            "closing": "正在关闭", "closed": "已关闭"}
                text = f"[{label}] {statuses.get(event.status, event.status)}"
            else:
                preview = event.content[:500]
                if len(event.content) > 500:
                    preview += "…"
                text = f"[{label} → 主 Agent] {preview}"
            self._commit(Text(text, style="dim"))
            return
        if isinstance(event, TurnStarted):
            self._begin_turn()
            return

        if isinstance(event, ContextCompaction):
            self._finish_plain_line()
            self._flush_exploration()
            if event.stage == "started":
                self._commit(Text(f"· 上下文超出预算，正在整理历史（约 {event.before_tokens:,} tokens）", style="dim"))
                self._set_phase("正在压缩上下文")
                self._show_working()
            elif event.stage == "chunk_started":
                phase = "摘要" if event.round_number == 1 else f"收短第 {event.round_number - 1} 轮"
                progress = f"{event.chunk_number}/{event.chunk_count}" if event.chunk_count else f"第 {event.chunk_number} 块"
                label = f"压缩上下文 · {phase} {progress}"
                self._set_phase(label)
                if not self._interactive:
                    self._commit(Text(label, style="dim"))
                self._show_working()
            elif event.stage == "completed":
                self._commit(Text(
                    f"✓ 上下文已压缩 · 约 {event.before_tokens:,} → {event.after_tokens:,} tokens"
                    f" · {event.request_count} 次摘要请求 · {event.elapsed_seconds:.1f}s", style="green",
                ))
                self._set_phase("等待模型响应")
                self._show_working()
            elif event.stage == "failed":
                self._commit(Text(
                    f"× 上下文压缩失败 · {event.elapsed_seconds:.1f}s · {event.error}", style="red",
                ))
                self._stop_live()
            return

        if isinstance(event, ModelRequestStarted):
            self._set_phase("等待模型响应")
            self._streamed_length = 0
            self._fence = None
            self._capacity = capacity_footer(event.input_items)
            self._assistant_buffer = ""
            self._assistant_committed = False
            self._request_has_text = False
            if not self._exploration_items:
                self._show_working()
            else:
                self._restore_active_state()
            return

        if isinstance(event, AssistantTextDelta):
            self._on_text_delta(event.text)
            return

        if isinstance(event, ModelRequestCompleted):
            self._on_model_request_completed(event)
            return

        if isinstance(event, AssistantTextCompleted):
            self._on_text_completed(event.text)
            return

        if isinstance(event, ToolCallPrepared):
            self._finish_plain_line()
            self._prepared_calls[event.call_id] = _PreparedCall(
                event.tool_name,
                dict(event.arguments),
            )
            return

        if isinstance(event, ApprovalRequested):
            self._flush_exploration()
            self._stop_live()
            return

        if isinstance(event, ApprovalResolved):
            if self._interactive:
                self._ensure_live()
                self._restore_active_state()
            return

        if isinstance(event, ToolStarted):
            self._on_tool_started(event)
            return

        if isinstance(event, ToolFinished):
            self._on_tool_finished(event)
            return

        if isinstance(event, PlanUpdated):
            self._finish_plain_line()
            self._flush_exploration()
            complete = all(step.status.value == "completed" for step in event.plan)
            self._commit(plan_cell(event.plan, compact=self._plan_seen and not complete))
            self._plan_seen = True
            self._show_working()
            return

        if isinstance(event, TurnFailed):
            self._flush_answer()
            self._finish_plain_line()
            self._flush_exploration()
            self._stop_live()
            if self._capacity is not None:
                self._commit(self._capacity)
            return

        if isinstance(event, TurnCompleted):
            self._finish_plain_line()
            self._flush_exploration()
            self._stop_live()
            if self._capacity is not None:
                self._commit(self._capacity)

    def _begin_turn(self) -> None:
        self._turn_started = monotonic()
        self._plan_seen = False
        self._tool_started.clear()
        self._capacity = None
        self._prepared_calls.clear()
        self._exploration_items.clear()
        self._assistant_buffer = ""
        self._assistant_committed = False
        self._request_has_text = False
        self._line_open = False
        self._show_working()

    def _on_text_delta(self, text: str) -> None:
        if not text:
            return
        first_text = not self._request_has_text
        self._request_has_text = True
        if not self._interactive:
            print(text, end="", file=self._console.file, flush=True)
            self._line_open = not text.endswith(("\n", "\r"))
            return

        if first_text:
            self._flush_exploration()
            self._set_phase("正在生成回答")
        self._assistant_buffer += text
        # 完成的 Markdown 段落写入历史，Live 只显示有限的未完成尾部。
        while "\n" in self._assistant_buffer or len(self._assistant_buffer) > 2000:
            lines = self._assistant_buffer.splitlines(keepends=True)
            end = 0
            fence = self._fence
            boundary = 0
            for line in lines:
                if not line.endswith("\n"):
                    break
                end += len(line)
                stripped = line.lstrip()
                if stripped.startswith(("```", "~~~")):
                    marker = stripped[:3]
                    fence = None if fence == marker else marker if fence is None else fence
                if not line.strip() and fence is None:
                    boundary = end
                    break
            if boundary:
                chunk = self._assistant_buffer[:boundary]
                self._commit(Text(chunk.rstrip("\n")) if self._fence else assistant_cell(chunk))
                self._fence = fence
                self._assistant_buffer = self._assistant_buffer[boundary:]
                self._streamed_length += boundary
            elif len(self._assistant_buffer) > 2000:
                # 超长代码块或段落按原文分批输出，避免超屏裁剪和反复重排。
                boundary = self._assistant_buffer.rfind("\n", 0, 2000) + 1 or 2000
                chunk = self._assistant_buffer[:boundary]
                for line in chunk.splitlines():
                    if line.lstrip().startswith(("```", "~~~")):
                        marker = line.lstrip()[:3]
                        self._fence = None if self._fence == marker else marker if self._fence is None else self._fence
                self._commit(Text(chunk.rstrip("\n")))
                self._assistant_buffer = self._assistant_buffer[boundary:]
                self._streamed_length += boundary
            else:
                break
        # 原文预览避免不完整表格、代码块的 Markdown 布局不断变化。
        self._set_active(Text(self._assistant_buffer[-1200:]))

    def _flush_answer(self) -> None:
        if self._interactive and self._assistant_buffer:
            self._commit(Text(self._assistant_buffer) if self._fence else assistant_cell(self._assistant_buffer))
            self._assistant_buffer = ""

    def _on_model_request_completed(
        self,
        event: ModelRequestCompleted,
    ) -> None:
        if not self._interactive or not self._request_has_text:
            return

        authoritative_text = (
            event.response.output_text
            if event.response is not None and event.response.output_text
            else self._assistant_buffer
        )
        if authoritative_text:
            self._flush_answer()
            self._assistant_committed = True

        has_tool_calls = bool(
            event.response is not None and event.response.function_calls
        )
        if has_tool_calls:
            self._show_working()
        else:
            self._clear_active()

    def _on_text_completed(self, text: str) -> None:
        if not self._interactive:
            if self._request_has_text:
                self._finish_plain_line()
            else:
                self._console.print(text, markup=False, highlight=False)
            return

        if not self._assistant_committed and text:
            self._flush_exploration()
            self._commit(assistant_cell(text))
            self._assistant_committed = True
        self._clear_active()

    def _on_tool_started(self, event: ToolStarted) -> None:
        self._tool_started[event.call_id] = monotonic()
        self._set_phase(f"正在执行 {event.tool_name}")
        call = self._prepared_call(event.call_id, event.tool_name)
        if call.name in _EXPLORATION_TOOLS:
            self._restore_active_state()
            return

        self._flush_exploration()
        if call.name in CONTEXT_TOOLS:
            self._set_active(context_activity(call.name, call.arguments))
        elif call.name == "update_plan":
            self._show_working()
        elif call.name == "run_command":
            self._set_active(
                command_cell(
                    _display_command(call.arguments),
                    active=True,
                )
            )
        elif call.name == "edit_file":
            self._set_active(
                tool_activity_cell(
                    "Editing",
                    _string_argument(call.arguments, "relative_path"),
                )
            )
        elif call.name == "create_file":
            self._set_active(
                tool_activity_cell(
                    "Creating",
                    _string_argument(call.arguments, "relative_path"),
                )
            )
        else:
            self._set_active(tool_activity_cell(f"Running {call.name}"))

    def _on_tool_finished(self, event: ToolFinished) -> None:
        call = self._prepared_calls.pop(
            event.call_id,
            _PreparedCall(event.tool_name, {}),
        )
        result = event.result
        started = self._tool_started.pop(event.call_id, None)
        elapsed = monotonic() - started if started is not None else None

        if call.name in _EXPLORATION_TOOLS:
            if result.success:
                self._exploration_items.append(_exploration_item(call.name, call.arguments))
                self._restore_active_state()
                return
            self._flush_exploration()
            self._commit(error_cell(f"{call.name} 失败", result.output))
            self._restore_active_state()
            return

        self._flush_exploration()
        if call.name == "update_plan" and result.success:
            return
        if not result.success:
            title = (
                "Command not run"
                if call.name == "run_command"
                and "declined by the user" in result.output
                else f"{_display_tool_name(call.name)} failed"
            )
            self._commit(error_cell(title, result.output))
            self._restore_active_state()
            return

        if call.name in CONTEXT_TOOLS:
            self._commit(context_completed(call.name, call.arguments, result.output))
        elif call.name == "run_command":
            exit_code, stdout, stderr = _parse_command_result(result.output)
            self._commit(
                command_cell(
                    _display_command(call.arguments),
                    active=False,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    elapsed=elapsed,
                )
            )
        elif call.name in {"edit_file", "create_file"}:
            self._commit(
                file_change_cell(
                    _string_argument(call.arguments, "relative_path"),
                    result.output,
                    created=call.name == "create_file",
                )
            )
        else:
            self._commit(
                tool_completed_cell(
                    _completed_tool_title(call.name),
                    _tool_detail(call.name, call.arguments),
                )
            )
        self._restore_active_state()

    def _prepared_call(self, call_id: str, name: str) -> _PreparedCall:
        return self._prepared_calls.get(call_id, _PreparedCall(name, {}))

    def _flush_exploration(self) -> None:
        if not self._exploration_items:
            return
        self._commit(exploration_cell(self._exploration_items, active=False))
        self._exploration_items.clear()

    def _restore_active_state(self) -> None:
        pending = [
            _exploration_item(call.name, call.arguments)
            for call_id in self._tool_started
            if (call := self._prepared_calls.get(call_id)) is not None
            and call.name in _EXPLORATION_TOOLS
        ]
        if self._exploration_items or pending:
            self._set_active(
                exploration_cell([*self._exploration_items, *pending], active=bool(pending))
            )
        else:
            self._show_working()

    def _show_working(self) -> None:
        if self._interactive:
            self._set_active(working_cell())

    def _set_active(self, renderable: RenderableType) -> None:
        if not self._interactive:
            return
        self._active = renderable
        self._ensure_live()
        assert self._live is not None
        self._live.update(self._live_content(), refresh=False)

    def _set_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_started = monotonic()

    def _live_content(self) -> RenderableType:
        now = monotonic()
        status = Text(f"{self._phase} · 本次 {now - self._phase_started:.0f}s · 总耗时 {now - self._turn_started:.0f}s", style="dim")
        return Group(self._active, status, *([self._capacity] if self._capacity is not None else []))

    def _clear_active(self) -> None:
        self._active = ""
        if self._live is not None:
            self._live.update("", refresh=True)

    def _commit(self, renderable: RenderableType) -> None:
        if self._live is not None:
            self._active = ""
            self._live.console.print(renderable)
        else:
            self._console.print(renderable)

    def _ensure_live(self) -> None:
        if not self._interactive or self._live is not None:
            return
        self._live = Live(
            "",
            console=self._console,
            refresh_per_second=10,
            transient=True,
            get_renderable=self._live_content,
        )
        self._live.start(refresh=True)

    def _stop_live(self) -> None:
        if self._live is None:
            return
        self._live.stop()
        self._live = None

    def _finish_plain_line(self) -> None:
        if self._line_open:
            print(file=self._console.file)
            self._line_open = False


def approve_tool_call(
    name: str,
    arguments: Mapping[str, object],
) -> bool:
    """Ask whether a tool that requires approval may run."""
    action = (
        f"Run {_display_command(arguments)}"
        if name == "run_command"
        else f"Allow {_display_tool_name(name)}"
    )
    try:
        print(f"\n执行目录：{arguments.get('relative_path', '.')}\n{action}")
        answer = input("是否执行？[y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _exploration_item(
    name: str,
    arguments: Mapping[str, object],
) -> ExplorationItem:
    path = _string_argument(arguments, "relative_path") or "."
    if name == "read_file":
        start = arguments.get("start_line", 1)
        count = arguments.get("max_lines", 400)
        return ExplorationItem("Read", f"{path} · 起始行 {start}，最多 {count} 行")
    if name == "list_files":
        return ExplorationItem("List", path)
    if name == "find_files":
        pattern = _string_argument(arguments, "file_pattern") or "*"
        return ExplorationItem("Find", f"{pattern} in {path}")
    query = _string_argument(arguments, "query")
    return ExplorationItem("Search", f"{query} in {path}".strip())


def _display_command(arguments: Mapping[str, object]) -> str:
    argv = arguments.get("argv")
    if not isinstance(argv, list) or not all(
        isinstance(part, str) for part in argv
    ):
        return "command"
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _parse_command_result(output: str) -> tuple[int, str, str]:
    first_line, separator, remainder = output.partition("\n")
    if not separator or not first_line.startswith("exit_code: "):
        return 0, output, ""
    try:
        exit_code = int(first_line.removeprefix("exit_code: "))
    except ValueError:
        return 0, output, ""

    stdout_prefix = "stdout:\n"
    if not remainder.startswith(stdout_prefix):
        return exit_code, remainder, ""
    stdout_and_stderr = remainder.removeprefix(stdout_prefix)
    stdout, separator, stderr = stdout_and_stderr.rpartition("\nstderr:\n")
    if not separator:
        return exit_code, stdout_and_stderr, ""
    return (
        exit_code,
        "" if stdout == "<empty>" else stdout,
        "" if stderr == "<empty>" else stderr,
    )


def _string_argument(
    arguments: Mapping[str, object],
    name: str,
) -> str:
    value = arguments.get(name)
    return value if isinstance(value, str) else ""


def _tool_detail(name: str, arguments: Mapping[str, object]) -> str:
    if name in {"read_file", "edit_file", "create_file"}:
        return _string_argument(arguments, "relative_path")
    return ""


def _display_tool_name(name: str) -> str:
    return name.replace("_", " ")


def _completed_tool_title(name: str) -> str:
    return f"Completed {_display_tool_name(name)}"
