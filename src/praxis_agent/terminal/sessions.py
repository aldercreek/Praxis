from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from praxis_agent.planning.state import RunStatus
from praxis_agent.sessions.models import SessionSummary


def print_session_list(
    sessions: list[SessionSummary],
    workspace: Path,
    *,
    console: Console | None = None,
) -> None:
    """使用紧凑表格展示当前工作区的持久化会话。"""
    output = console or Console()
    heading = Text("Saved sessions for ", style="bold")
    heading.append(str(workspace))
    output.print(heading)

    if not sessions:
        output.print("No saved sessions.", style="dim")
        return

    table = Table(
        box=None,
        padding=(0, 2),
        pad_edge=False,
        show_edge=False,
        header_style="dim",
    )
    table.add_column("ID", no_wrap=True)
    table.add_column("Updated", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Title", ratio=1)

    for session in sessions:
        status, status_style = _display_status(session.latest_run_status)
        table.add_row(
            Text(session.id[:8], style="bold cyan"),
            session.updated_at.astimezone().strftime("%Y-%m-%d %H:%M"),
            Text(status, style=status_style),
            session.title or "(untitled)",
        )
    output.print(table)


def _display_status(status: RunStatus | None) -> tuple[str, str]:
    if status is RunStatus.COMPLETED:
        return status.value, "green"
    if status is RunStatus.FAILED:
        return status.value, "red"
    if status is RunStatus.RUNNING:
        return status.value, "yellow"
    return "new", "dim"
