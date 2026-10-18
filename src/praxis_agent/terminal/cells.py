"""Rich renderables used by the Textual agent pages."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pygments.token import Comment, Keyword
from rich.console import Group, RenderableType
from rich.markdown import Heading, Markdown
from rich.style import Style
from rich.syntax import ANSISyntaxTheme
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from praxis_agent import __version__
from praxis_agent.terminal.preview import OutputPreview, output_lines

if TYPE_CHECKING:
    from praxis_agent.planning import PlanStep
    from praxis_agent.terminal.views import ToolBlock


ACCENT = "#8ab8c5"


@dataclass(frozen=True)
class WelcomeHeader:
    workspace: Path | None
    model: str | None
    compact: bool = False

    def __rich_console__(self, console, options):
        content = Text(no_wrap=True, overflow="ellipsis")
        if self.compact or options.max_width < 29:
            content.append("Praxis\n", style=f"bold {ACCENT}")
        else:
            content.append(
                "█▀▀█ █▀▀█ ▄▀▀▄ ▀▄ ▄▀ ▀█▀ ▄▀▀▀\n"
                "█▄▄█ █▄▄▀ █▄▄█   █    █  ▀▀▀▄\n"
                "█    █  █ █  █ ▄▀ ▀▄ ▄█▄ ▄▄▄▀\n\n",
                style=ACCENT,
            )
        content.append(f"v{__version__}", style="#a0a0a0")
        if self.model:
            content.append(" · 模型 ", style="#a0a0a0")
            content.append(self.model)
        if self.workspace is not None:
            workspace = self.workspace
            path = Text(str(workspace))
            path_width = max(1, options.max_width - 5)
            if path.cell_len > path_width:
                for index in range(1, len(workspace.parts)):
                    path = Text(str(Path(workspace.anchor, "…", *workspace.parts[index:])))
                    if path.cell_len <= path_width:
                        break
                if path.cell_len > path_width:
                    path = Text("…" + workspace.name)
                    while path.cell_len > path_width:
                        path = Text("…") + path[2:]
            content.append("\n目录 ", style="#a0a0a0")
            content.append_text(path)
        yield content


class ConversationHeading(Heading):
    def __rich_console__(self, console, options):
        yield Text("#" * int(self.tag[1]) + " ", style="bold") + self.text


class ConversationMarkdown(Markdown):
    elements: ClassVar[dict] = {**Markdown.elements, "heading_open": ConversationHeading}


# Keep conversation code in the same foreground as prose; weight carries emphasis.
CODE_THEME = ANSISyntaxTheme({
    Keyword: Style(bold=True), Comment: Style(dim=True),
})
MARKDOWN_THEME = Theme({
    **{f"markdown.h{level}": "bold" for level in range(1, 7)},
    "markdown.code": "bold", "markdown.code_block": "none",
})


_MESSAGE_ID = r"m(?:[0-9]+|[0-9a-f]{32})"
_MESSAGE_LABEL = re.compile(
    rf"\[message_id={_MESSAGE_ID}(?:, operation_id=op[0-9]+)?\][ \t]*(?:\r?\n|$)")


def _pending_message_label(text: str) -> bool:
    """Hold only a possible leading metadata line while its chunks are arriving."""
    if text.endswith("\r") and _MESSAGE_LABEL.fullmatch(text[:-1]):
        return True
    start = "[message_id=m"
    if start.startswith(text):
        return True
    if not text.startswith(start):
        return False
    identifier, separator, suffix = text[len(start):].partition(",")
    if not separator:
        return re.fullmatch(r"[0-9]+|[0-9a-f]{0,32}", identifier) is not None
    if re.fullmatch(_MESSAGE_ID, "m" + identifier) is None:
        return False
    operation = " operation_id=op"
    return operation.startswith(suffix) or (
        suffix.startswith(operation) and re.fullmatch(r"[0-9]*", suffix[len(operation):]) is not None
    )


@dataclass
class MessageBlock:
    """One source-backed message, shared by streaming and completed rendering."""

    text: str
    user: bool = False
    streaming: bool = False
    revision: int = field(default=0, init=False)

    def update(self, text: str, *, streaming: bool = False) -> None:
        if text != self.text or streaming != self.streaming:
            self.text = text
            self.streaming = streaming
            self.revision += 1

    @property
    def display_text(self) -> str:
        if self.user:
            return self.text
        text = self.text
        while match := _MESSAGE_LABEL.match(text):
            text = text[match.end():]
        return "" if self.streaming and _pending_message_label(text) else text

    def __rich_console__(self, console, options):
        text = self.display_text
        if not text:
            return
        grid = Table.grid(padding=0, expand=True)
        grid.add_column(width=2, no_wrap=True)
        grid.add_column(ratio=1)
        grid.add_row(
            Text("›" if self.user else "•", style="dim"),
            Text(text) if self.user else ConversationMarkdown(text, code_theme=CODE_THEME),
        )
        # A message may be taller than the viewport; scrolling must retain every line.
        with console.use_theme(MARKDOWN_THEME):
            yield from console.render(grid, options.update(height=None))


@dataclass(frozen=True)
class CollaborationBlock:
    label: str
    text: str

    def __rich_console__(self, console, options):
        yield Text(f"• {self.label} → 主 Agent：收到新消息", style="dim")


def collaboration_messages(text: str) -> list[dict] | None:
    prefix = "[Agent collaboration messages; source-labelled task data]\n"
    if text.startswith(prefix):
        try:
            messages = json.loads(text[len(prefix):])
        except ValueError:
            return None
        if isinstance(messages, list) and all(
            isinstance(message, dict) and isinstance(message.get("content"), str)
            for message in messages
        ):
            return messages
    return None


def plan_cell(plan: Iterable[PlanStep], *, compact: bool = False) -> RenderableType:
    plan = list(plan)
    if compact:
        completed = sum(step.status.value == "completed" for step in plan)
        current = next((step.description for step in plan if step.status.value == "in_progress"), "等待下一步")
        return Group(_completed_header(f"计划 · {completed}/{len(plan)} 已完成"), _detail_line(current))
    lines = [_completed_header("更新计划")]
    for index, step in enumerate(plan):
        prefix = "  └ " if index == 0 else "    "
        line = Text(prefix, style="dim")
        if step.status.value == "completed":
            line.append("✔ ", style="dim")
            line.append(step.description, style="dim strike")
        elif step.status.value == "in_progress":
            line.append("› ", style="bold")
            line.append(step.description, style="bold")
        else:
            line.append("○ ", style="dim")
            line.append(step.description, style="dim")
        lines.append(line)
    return Group(*lines)


def _completed_header(title: str) -> Text:
    line = Text.assemble(("• ", "bold green"), title)
    return line


def _detail_line(detail: str) -> Text:
    line = Text("  └ ")
    line.append(detail)
    return line


def display_command(arguments: Mapping[str, object]) -> str:
    if isinstance(arguments.get("command"), str):
        return arguments["command"]
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
    "delete": "整理消息", "compress": "压缩上下文",
    "search_history": "搜索历史", "read_history": "读取历史",
    "spawn_agent": "创建子 Agent", "send_messages": "发送协作消息",
    "wait_agents": "等待子 Agent", "close_agent": "关闭子 Agent", "send_message": "汇报结果",
}


def tool_description(name: str, arguments: Mapping[str, object]) -> tuple[str, str]:
    label = _TOOL_LABELS.get(name, name)
    if name == "run_command":
        return label, display_command(arguments)
    if name == "spawn_agent":
        return label, str(arguments.get("task_summary") or arguments.get("task", ""))
    if name in {"wait_agents", "send_messages"}:
        items = arguments.get("agent_ids" if name == "wait_agents" else "messages", [])
        return label, f"{len(items)} 个 Agent" if name == "wait_agents" else f"{len(items)} 条消息"
    if name in {"delete", "compress"}:
        return label, f"{len(arguments.get('message_ids', []))} 条消息"
    path = str(arguments.get("relative_path", ""))
    query = arguments.get("query") or arguments.get("file_pattern")
    detail = f"{query} · {path}" if query and path else str(query or path)
    return label, detail


def tool_title(calls: tuple[ToolBlock, ...]) -> Text:
    if len(calls) > 1:
        running = any(call.state == "running" for call in calls)
        title = Text(f"{'正在' if running else '已'}查看代码 · {len(calls)} 次操作")
        failed = sum(call.failed for call in calls)
        cancelled = sum(call.state == "cancelled" for call in calls)
        if failed:
            title.append(f" · {failed} 次失败", style="red")
        if cancelled:
            title.append(f" · {cancelled} 次已停止", style="dim")
        return title
    call = calls[0]
    title = Text({"running": "正在", "cancelled": "已停止 · "}.get(call.state, "已"))
    if call.failed:
        title = Text("失败 · ")
    if call.structured_output is not None:
        result = call.structured_output
        label = {"spawn_failed": "启动失败", "timed_out": "已超时", "cancelled": "已停止"}.get(result["status"])
        if result["status"] == "exited":
            label = "已完成" if result["exit_code"] == 0 else f"已结束 · 退出码 {result['exit_code']}"
        title = Text(f"{label} · ")
    title.append(call.title)
    if call.detail:
        title.append(" ")
        title.append(call.detail if call.name == "run_command" else " ".join(call.detail.split()))
    if call.file_change is not None:
        title.append(" (")
        title.append(f"+{call.file_change.added}", style="green")
        title.append(f" -{call.file_change.removed}", style="red")
        title.append(")")
    return title


def change_lines(change) -> list[Text]:
    rows = []
    old_line = new_line = 0
    for line in change.diff.splitlines()[2:]:
        hunk = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if hunk:
            old_line, new_line = int(hunk[1]), int(hunk[2])
            rows.append(Text(line, style="dim"))
        elif line.startswith(("+", "-", " ")):
            before = str(old_line) if not line.startswith("+") else ""
            after = str(new_line) if not line.startswith("-") else ""
            row = Text(f"{before:>4} {after:>4} ", style="dim")
            row.append(line, style="not dim green" if line.startswith("+") else
                       "not dim red" if line.startswith("-") else "not dim")
            rows.append(row)
            old_line += not line.startswith("+")
            new_line += not line.startswith("-")
        else:
            rows.append(Text(line, style="dim"))
    return rows or [Text("内容未变化")]


def _operation_summary(call: ToolBlock) -> list[Text] | None:
    if call.failed and call.name != "send_messages":
        return None
    if call.name in {"delete", "compress"}:
        return []  # IDs remain in details; the title already describes the change.
    if call.name not in {"spawn_agent", "send_messages", "wait_agents", "close_agent", "send_message"}:
        return None
    try:
        result = json.loads(call.output)
    except (ValueError, TypeError):
        return None
    if not isinstance(result, dict) or (call.failed and "results" not in result):
        return None
    statuses = {"queued": "排队中", "running": "执行中", "idle": "本轮结束",
                "closing": "正在关闭", "closed": "已关闭"}
    if call.name == "wait_agents":
        lines = [Text(f"{statuses.get(agent['status'], agent['status'])} · {agent.get('task') or agent['agent_id']}")
                 for agent in result.get("agents", [])]
        count = len(result.get("messages", []))
        if count:
            lines.append(Text(f"收到 {count} 条协作消息；Ctrl+T 查看全文"))
        if result.get("timed_out"):
            lines.append(Text("本次等待结束，子 Agent 仍在执行"))
        return lines
    if call.name == "send_messages":
        return [Text(str(item.get("error")), style="red") for item in result.get("results", []) if item.get("error")]
    return []


def tool_preview(calls: tuple[ToolBlock, ...]) -> RenderableType:
    if len(calls) > 1:
        details = [tool_title((call,)) for call in calls]
    else:
        call = calls[0]
        if call.exploration and not call.failed:
            return ""
        if call.state == "running" and not call.output:
            return ""
        summary = _operation_summary(call) if call.state == "completed" or call.failed else None
        if summary is not None:
            return OutputPreview(summary) if summary else ""
        details = change_lines(call.file_change) if call.file_change is not None else output_lines(
            call.structured_output if call.structured_output is not None else call.output)
    return OutputPreview(details)


def tool_details(calls: tuple[ToolBlock, ...]) -> RenderableType:
    lines = []
    for index, call in enumerate(calls):
        if index:
            lines.append(Text(""))
        if len(calls) > 1:
            lines.append(Text(f"{call.title} {call.detail}".rstrip(), style="bold"))
        if call.file_change is not None:
            lines.extend(change_lines(call.file_change))
        lines.extend((Text("参数", style="dim"), Text(call.arguments), Text("输出", style="dim")))
        if call.structured_output is not None:
            for field, value in call.structured_output.items():
                lines.extend((Text(field, style="dim"), Text.from_ansi(str(value) if value is not None else "null")))
        else:
            lines.append(Text.from_ansi(call.output))
    return Group(*lines)


def approval_content(name: str, arguments: Mapping[str, object]) -> RenderableType:
    """Readable operation details; the separate parameter viewer retains the exact payload."""
    lines = []
    if name == "run_command":
        lines.append(Text("$ " + display_command(arguments)))
    else:
        title, detail = tool_description(name, arguments)
        lines.append(Text(f"{title} {detail}".rstrip()))
    labels = {"relative_path": "目录", "workdir": "目录", "workspace": "工作区", "timeout_seconds": "超时（秒）", "reason": "原因",
              "old_string": "替换前", "new_string": "替换后", "content": "写入内容"}
    for key, value in arguments.items():
        if key in {"argv", "command", "agent_id"}:
            continue
        line = Text(f"{labels.get(key, key)}：", style="dim")
        line.append(str(value), style="not dim")
        lines.append(line)
    return Group(*lines)
