"""Rich renderables used by the Textual agent pages."""
from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from praxis_agent.planning import PlanStep
    from praxis_agent.terminal.views import ToolBlock


@dataclass
class MessageBlock:
    """One source-backed message, shared by streaming and completed rendering."""

    text: str
    user: bool = False
    revision: int = field(default=0, init=False)

    def update(self, text: str) -> None:
        if text != self.text:
            self.text = text
            self.revision += 1

    def __rich_console__(self, console, options):
        grid = Table.grid(padding=0, expand=True)
        grid.add_column(width=2, no_wrap=True)
        grid.add_column(ratio=1)
        grid.add_row(
            Text("›" if self.user else "•", style="dim"),
            Text(self.text) if self.user else Markdown(self.text, code_theme="ansi_dark"),
        )
        # A message may be taller than the viewport; scrolling must retain every line.
        yield from console.render(grid, options.update(height=None))


def working_cell() -> RenderableType:
    return _block("处理中", active=True)


def plan_cell(plan: Iterable[PlanStep], *, compact: bool = False) -> RenderableType:
    plan = list(plan)
    if compact:
        completed = sum(step.status.value == "completed" for step in plan)
        current = next((step.description for step in plan if step.status.value == "in_progress"), "等待下一步")
        return _block(f"计划 · {completed}/{len(plan)} 已完成", [_detail_line(current)])
    lines = [_completed_header("Updated Plan")]
    for index, step in enumerate(plan):
        prefix = "  └ " if index == 0 else "    "
        line = Text(prefix, style="dim")
        if step.status.value == "completed":
            line.append("✔ ", style="dim")
            line.append(step.description, style="dim strike")
        elif step.status.value == "in_progress":
            line.append("› ", style="bold cyan")
            line.append(step.description, style="bold cyan")
        else:
            line.append("○ ", style="dim")
            line.append(step.description, style="dim")
        lines.append(line)
    return Group(*lines)


def tool_activity_cell(title: str, detail: str = "") -> RenderableType:
    details = [_detail_line(detail)] if detail else []
    return _block(title, details, active=True)


def _block(
    title: str | Text,
    details: Iterable[RenderableType] = (),
    *,
    active: bool = False,
) -> RenderableType:
    header = Table.grid(padding=0)
    header.add_column(width=2, no_wrap=True)
    header.add_column()
    indicator: RenderableType
    if active:
        indicator = Spinner("dots", style="dim")
    else:
        indicator = Text("•", style="dim")
    header.add_row(
        indicator,
        Text(title, style="bold") if isinstance(title, str) else title,
    )
    return Group(header, *details)


def _completed_header(title: str) -> Text:
    line = Text("• ", style="dim")
    line.append(title, style="bold")
    return line


def _detail_line(detail: str) -> Text:
    line = Text("  └ ", style="dim")
    line.append(detail)
    return line


def display_command(arguments: Mapping[str, object]) -> str:
    argv = arguments.get("argv")
    if not isinstance(argv, list) or not all(
        isinstance(part, str) for part in argv
    ):
        return "command"
    if not argv:
        return "command"
    executable = argv[0].replace("\\", "/").rsplit("/", 1)[-1].lower().removesuffix(".exe")
    # Only simplify the display of known shell wrappers; execution keeps the original argv.
    flags = {"powershell": "-command", "pwsh": "-command", "cmd": "/c"}
    if executable in flags and len(argv) >= 3 and argv[-2].lower() == flags[executable]:
        return argv[-1]
    if executable in {"bash", "sh", "zsh"} and len(argv) == 3 and argv[1] in {"-c", "-lc"}:
        return argv[2]
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


_TOOL_LABELS = {
    "read_file": "读取", "list_files": "列出", "find_files": "查找",
    "search_text": "搜索", "edit_file": "修改", "write_file": "写入",
    "run_command": "执行", "update_plan": "更新计划",
}


def tool_description(name: str, arguments: Mapping[str, object]) -> tuple[str, str]:
    label = _TOOL_LABELS.get(name, name)
    if name == "run_command":
        return label, display_command(arguments)
    path = str(arguments.get("relative_path", ""))
    query = arguments.get("query") or arguments.get("file_pattern")
    detail = f"{query} · {path}" if query and path else str(query or path)
    return label, detail


def tool_title(calls: tuple[ToolBlock, ...]) -> Text:
    if len(calls) > 1:
        return Text(f"已查看代码 · {len(calls)} 次操作", style="bold")
    call = calls[0]
    title = Text("失败 · " if call.failed else "已", style="bold red" if call.failed else "bold")
    title.append(call.title)
    if call.detail:
        title.append(" " + " ".join(call.detail.split()), style="not bold")
    return title


def _preview_lines(output: str, *, command: bool) -> list[Text]:
    if command:
        # The tool's transport envelope is retained in details, not in the preview.
        _, separator, streams = output.partition("\nstdout:\n")
        if separator:
            stdout, stderr_separator, stderr = streams.rpartition("\nstderr:\n")
            if stderr_separator:
                parts = [part for part in (stdout, stderr) if part.strip() != "<empty>"]
                output = "\n".join(parts)
    lines = [Text.from_ansi(line) for line in output.splitlines() if line.strip()]
    return lines[-3:]


def tool_preview(calls: tuple[ToolBlock, ...]) -> RenderableType:
    if len(calls) > 1:
        details = [Text(f"{call.title} {call.detail}".rstrip()) for call in calls[-3:]]
        if len(calls) > 3:
            details.insert(0, Text(f"… 另有 {len(calls) - 3} 次操作 · Ctrl+T 查看"))
    else:
        call = calls[0]
        if not call.failed and call.name != "run_command":
            return ""
        details = _preview_lines(call.output, command=call.name == "run_command")
    lines = []
    for index, detail in enumerate(details):
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append("  └ " if index == 0 else "    ", style="dim")
        line.append_text(detail)
        lines.append(line)
    return Group(*lines)


def tool_details(calls: tuple[ToolBlock, ...]) -> RenderableType:
    lines = []
    for index, call in enumerate(calls):
        if index:
            lines.append(Text(""))
        if len(calls) > 1:
            lines.append(Text(f"{call.title} {call.detail}".rstrip(), style="bold"))
        lines.extend((Text("参数", style="dim"), Text(call.arguments),
                      Text("输出", style="dim"), Text.from_ansi(call.output)))
    return Group(*lines)


def approval_content(name: str, arguments: Mapping[str, object]) -> RenderableType:
    """Readable operation details; the separate parameter viewer retains the exact payload."""
    lines = []
    if name == "run_command":
        lines.append(Text("$ " + display_command(arguments)))
    else:
        title, detail = tool_description(name, arguments)
        lines.append(Text(f"{title} {detail}".rstrip()))
    labels = {"relative_path": "目录", "workspace": "工作区", "timeout_seconds": "超时（秒）",
              "old_string": "替换前", "new_string": "替换后", "content": "写入内容"}
    for key, value in arguments.items():
        if key in {"argv", "agent_id"}:
            continue
        line = Text(f"{labels.get(key, key)}：", style="dim")
        line.append(str(value), style="not dim")
        lines.append(line)
    return Group(*lines)
