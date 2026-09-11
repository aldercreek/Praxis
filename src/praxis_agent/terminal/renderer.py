from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass

from rich.console import Console, RenderableType
from rich.live import Live

from praxis_agent.events import (
    AgentEvent,
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

    def emit(self, event: AgentEvent) -> None:
        """Update the transcript and active area for one agent event."""
        if isinstance(event, TurnStarted):
            self._begin_turn()
            return

        if isinstance(event, ModelRequestStarted):
            self._assistant_buffer = ""
            self._assistant_committed = False
            self._request_has_text = False
            if not self._exploration_items:
                self._show_working()
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
            self._commit(plan_cell(event.plan))
            self._show_working()
            return

        if isinstance(event, TurnFailed):
            self._finish_plain_line()
            self._flush_exploration()
            self._stop_live()
            return

        if isinstance(event, TurnCompleted):
            self._finish_plain_line()
            self._flush_exploration()
            self._stop_live()

    def _begin_turn(self) -> None:
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
        self._request_has_text = True
        if not self._interactive:
            print(text, end="", file=self._console.file, flush=True)
            self._line_open = not text.endswith(("\n", "\r"))
            return

        if not self._assistant_buffer:
            self._flush_exploration()
        self._assistant_buffer += text
        self._set_active(assistant_cell(self._assistant_buffer))

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
            self._commit(assistant_cell(authoritative_text))
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
        call = self._prepared_call(event.call_id, event.tool_name)
        if call.name in _EXPLORATION_TOOLS:
            item = _exploration_item(call.name, call.arguments)
            self._exploration_items.append(item)
            self._set_active(
                exploration_cell(self._exploration_items, active=True)
            )
            return

        self._flush_exploration()
        if call.name == "update_plan":
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

        if call.name in _EXPLORATION_TOOLS:
            if result.success:
                self._set_active(
                    exploration_cell(self._exploration_items, active=False)
                )
                return
            if self._exploration_items:
                self._exploration_items.pop()
            self._flush_exploration()
            self._commit(error_cell("Exploration failed", result.output))
            self._show_working()
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
            self._show_working()
            return

        if call.name == "run_command":
            exit_code, stdout, stderr = _parse_command_result(result.output)
            self._commit(
                command_cell(
                    _display_command(call.arguments),
                    active=False,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
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
        self._show_working()

    def _prepared_call(self, call_id: str, name: str) -> _PreparedCall:
        return self._prepared_calls.get(call_id, _PreparedCall(name, {}))

    def _flush_exploration(self) -> None:
        if not self._exploration_items:
            return
        self._commit(exploration_cell(self._exploration_items, active=False))
        self._exploration_items.clear()

    def _restore_active_state(self) -> None:
        if self._exploration_items:
            self._set_active(
                exploration_cell(self._exploration_items, active=False)
            )
        else:
            self._show_working()

    def _show_working(self) -> None:
        if self._interactive:
            self._set_active(working_cell())

    def _set_active(self, renderable: RenderableType) -> None:
        if not self._interactive:
            return
        self._ensure_live()
        assert self._live is not None
        self._live.update(renderable, refresh=True)

    def _clear_active(self) -> None:
        if self._live is not None:
            self._live.update("", refresh=True)

    def _commit(self, renderable: RenderableType) -> None:
        if self._live is not None:
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
        answer = input(f"{action}? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _exploration_item(
    name: str,
    arguments: Mapping[str, object],
) -> ExplorationItem:
    path = _string_argument(arguments, "relative_path") or "."
    if name == "read_file":
        return ExplorationItem("Read", path)
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
