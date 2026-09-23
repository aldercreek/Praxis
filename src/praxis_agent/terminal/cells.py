from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Iterable

    from praxis_agent.planning import PlanStep


@dataclass(frozen=True, slots=True)
class ExplorationItem:
    action: str
    detail: str


def working_cell() -> RenderableType:
    return _block("Working", active=True)


def assistant_cell(markdown: str) -> RenderableType:
    grid = Table.grid(padding=0, expand=True)
    grid.add_column(width=2, no_wrap=True)
    grid.add_column(ratio=1)
    grid.add_row(
        Text("•", style="dim"),
        Markdown(markdown, code_theme="ansi_dark"),
    )
    return grid


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


def exploration_cell(
    items: Iterable[ExplorationItem],
    *,
    active: bool,
) -> RenderableType:
    details: list[Text] = []
    items = list(items)
    paths = [item.detail for item in items if item.action in {"Read", "List"}]
    prefix = ""
    if len(paths) == len(items) and len(paths) > 1:
        import posixpath
        prefix = posixpath.commonpath([posixpath.dirname(path) for path in paths])
        if prefix:
            details.append(_detail_line(f"目录：{prefix}/"))
    for index, item in enumerate(items):
        line = Text("  └ " if index == 0 else "    ", style="dim")
        line.append(item.action, style="cyan")
        if item.detail:
            line.append(" ")
            line.append(item.detail.removeprefix(prefix + "/") if prefix else item.detail)
        details.append(line)
    return _block(
        "Exploring" if active else "Explored",
        details,
        active=active,
    )


def command_cell(
    command: str,
    *,
    active: bool,
    exit_code: int | None = None,
    stdout: str = "",
    stderr: str = "",
    elapsed: float | None = None,
) -> RenderableType:
    title = Text("Running " if active else "Ran ", style="bold")
    title.append(command if len(command) <= 120 else command[:117] + "…")
    if exit_code is not None:
        title.append(f" · exit {exit_code}")
    if elapsed is not None:
        title.append(f" · {elapsed:.1f}s")

    details: list[Text] = []
    if not active:
        visible_lines = _command_output_lines(stdout, stderr)
        if not visible_lines:
            visible_lines = [("(no output)", "dim")]
        for index, (content, style) in enumerate(visible_lines):
            line = Text("  └ " if index == 0 else "    ", style="dim")
            output = Text.from_ansi(content)
            if style:
                output.stylize(style)
            else:
                output.stylize("dim")
            line.append_text(output)
            details.append(line)

    marker_style = None
    if exit_code is not None:
        marker_style = "bold green" if exit_code == 0 else "bold red"
    return _block(
        title,
        details,
        active=active,
        marker_style=marker_style,
    )


def file_change_cell(
    path: str,
    diff: str,
    *,
    created: bool,
) -> RenderableType:
    added, removed = _diff_counts(diff)
    title = Text("Added " if created else "Edited ", style="bold")
    title.append(path)
    title.append(" (")
    title.append(f"+{added}", style="green")
    title.append(" ")
    title.append(f"-{removed}", style="red")
    title.append(")")

    details: list[Text] = []
    preview = _middle_truncate(diff.splitlines(), max_lines=7)
    for index, content in enumerate(preview):
        line = Text("  └ " if index == 0 else "    ", style="dim")
        style = "dim"
        if content.startswith("+") and not content.startswith("+++"):
            style = "green"
        elif content.startswith("-") and not content.startswith("---"):
            style = "red"
        elif content.startswith("@@"):
            style = "cyan dim"
        line.append(content, style=style)
        details.append(line)

    return _block(title, details)


def tool_activity_cell(title: str, detail: str = "") -> RenderableType:
    details = [_detail_line(detail)] if detail else []
    return _block(title, details, active=True)


def tool_completed_cell(title: str, detail: str = "") -> RenderableType:
    details = [_detail_line(detail)] if detail else []
    return _block(title, details)


def error_cell(title: str, detail: str = "") -> RenderableType:
    details: list[Text] = []
    if detail:
        for index, content in enumerate(
            _middle_truncate(detail.split("\nCheck the arguments for", 1)[0].splitlines(), max_lines=3)
        ):
            line = Text("  └ " if index == 0 else "    ", style="dim")
            line.append(content[:240] + ("…" if len(content) > 240 else ""), style="red")
            details.append(line)
        if "\nCheck the arguments for" in detail:
            details.append(_detail_line("参数修正说明已返回模型；需按工具定义修正后重试。"))
    return _block(title, details, marker="✗", marker_style="bold red")


def _block(
    title: str | Text,
    details: Iterable[RenderableType] = (),
    *,
    active: bool = False,
    marker: str = "•",
    marker_style: str | None = "dim",
) -> RenderableType:
    header = Table.grid(padding=0)
    header.add_column(width=2, no_wrap=True)
    header.add_column()
    indicator: RenderableType
    if active:
        indicator = Spinner("dots", style="dim")
    else:
        indicator = Text(marker, style=marker_style)
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


def _command_output_lines(
    stdout: str,
    stderr: str,
) -> list[tuple[str, str | None]]:
    lines: list[tuple[str, str | None]] = []
    lines.extend((line, None) for line in stdout.splitlines())
    lines.extend((line, "red dim") for line in stderr.splitlines())
    return _middle_truncate(lines, max_lines=5)


def _middle_truncate[T](items: list[T], max_lines: int) -> list[T]:
    if len(items) <= max_lines:
        return items
    head_count = max_lines // 2
    tail_count = max_lines - head_count - 1
    omitted = len(items) - head_count - tail_count
    marker = f"… +{omitted} lines"
    if items and isinstance(items[0], tuple):
        ellipsis = (marker, "dim")
    else:
        ellipsis = marker
    return [*items[:head_count], ellipsis, *items[-tail_count:]]  # type: ignore[list-item]


def _diff_counts(diff: str) -> tuple[int, int]:
    added = 0
    removed = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed
