"""One Textual screen with memory-only, independently scrollable agent pages."""

from __future__ import annotations

import json
from concurrent.futures import CancelledError, Future, TimeoutError
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from threading import Thread
from time import monotonic
from typing import TYPE_CHECKING

from rich.console import Group
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static

from praxis_agent.terminal.cells import MessageBlock, approval_content, tool_preview, tool_title
from praxis_agent.terminal.views import ToolBlock, ViewEventSink, ViewStore

if TYPE_CHECKING:
    from praxis_agent.agent import Agent


@dataclass
class Approval:
    tool: str
    arguments: dict
    future: Future


class ApprovalPanel(Vertical):
    """An inline choice in the bottom pane; decisions still resolve the existing Future."""

    DEFAULT_CSS = """
    ApprovalPanel { display: none; height: auto; max-height: 80%; padding: 1 1 0 1;
                    border-top: solid $foreground 15%; }
    #approval-heading { height: auto; text-style: bold; margin-bottom: 1; }
    #approval-details { height: auto; max-height: 20vh; scrollbar-size: 0 0; }
    #approval-choices { height: 2; min-height: 2; border: none; padding: 0; margin-top: 1;
                        background: transparent; scrollbar-size: 0 0; }
    #approval-choices > .option-list--option { padding: 0; }
    #approval-choices > .option-list--option-highlighted {
        background: transparent; color: $accent; text-style: bold;
    }
    #approval-choices > .option-list--option-hover { background: transparent; }
    #approval-hint { height: auto; color: $text-muted; margin-top: 1; }
    """
    BINDINGS = [
        Binding("y,1", "allow", "允许本次", show=False),
        Binding("n,2,escape", "deny", "拒绝", show=False),
    ]

    class Decided(Message):
        def __init__(self, request: Approval, allowed: bool) -> None:
            super().__init__()
            self.request = request
            self.allowed = allowed

    def __init__(self) -> None:
        super().__init__(id="approval")
        self.request: Approval | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="approval-heading")
        with VerticalScroll(id="approval-details"):
            yield Static("", id="approval-content")
        yield OptionList("1. 允许本次 (y)", "2. 拒绝 (n)", id="approval-choices")
        yield Static("↑/↓ 选择 · Enter 确认 · Esc 拒绝 · Ctrl+T 查看完整参数", id="approval-hint")

    def show_request(self, request: Approval, label: str) -> None:
        self.request = request
        question = "是否执行以下命令？" if request.tool == "run_command" else "是否允许以下操作？"
        self.query_one("#approval-heading", Static).update(Text(f"{label} · {question}"))
        self.query_one("#approval-content", Static).update(approval_content(request.tool, request.arguments))
        self.query_one("#approval-details", VerticalScroll).scroll_home(animate=False)
        self.display = True
        choices = self.query_one(OptionList)
        choices.highlighted = 0
        choices.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self._decide(event.option_index == 0)

    def _decide(self, allowed: bool) -> None:
        if self.request is not None:
            self.post_message(self.Decided(self.request, allowed))
            self.request = None

    def action_allow(self) -> None:
        self._decide(True)

    def action_deny(self) -> None:
        self._decide(False)


class DetailsScreen(ModalScreen[None]):
    """A full-screen transcript/parameter viewer, independent of the main scroll position."""

    DEFAULT_CSS = """
    DetailsScreen { background: $background; padding: 0 1; }
    #details-heading { height: auto; padding: 1 0; text-style: bold; }
    #details-scroll { height: 1fr; scrollbar-size: 0 0; }
    #details-hint { height: 1; color: $text-muted; }
    """
    BINDINGS = [Binding("escape,q", "close", "返回", show=False)]

    def __init__(self, title: str, content, *, agent_id: str | None = None,
                 request: Approval | None = None) -> None:
        super().__init__()
        self.title = title
        self.content = content
        self.agent_id = agent_id
        self.request = request

    def compose(self) -> ComposeResult:
        yield Static(Text(self.title), id="details-heading")
        with VerticalScroll(id="details-scroll"):
            yield Static(self.content, id="details-content")
        yield Static("↑/↓ 或 PgUp/PgDn 滚动 · Esc / Ctrl+T 返回", id="details-hint")

    def on_mount(self) -> None:
        self.query_one("#details-scroll", VerticalScroll).focus()

    def update_content(self, content) -> None:
        pane = self.query_one("#details-scroll", VerticalScroll)
        position = pane.scroll_y
        self.query_one("#details-content", Static).update(content)
        self.call_after_refresh(pane.scroll_to, y=position, animate=False, immediate=True)

    def action_close(self) -> None:
        self.dismiss()


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02d}"


class ToolRecord(Vertical):
    """A non-interactive activity summary; full results live in the transcript viewer."""

    def __init__(self, calls: tuple[ToolBlock, ...], index: int) -> None:
        super().__init__(id=f"record-{index}", classes="record")
        self.calls = calls

    def compose(self) -> ComposeResult:
        yield Static("", classes="tool-title")
        yield Static("", classes="tool-preview")

    def on_mount(self) -> None:
        self.update_calls(self.calls)

    def update_calls(self, calls: tuple[ToolBlock, ...]) -> None:
        self.calls = calls
        self._update_title()
        preview = tool_preview(calls)
        widget = self.query_one(".tool-preview", Static)
        widget.update(preview)
        widget.display = bool(preview)

    def _update_title(self) -> None:
        title = Text("• ", style="red" if self.calls[0].failed else "dim")
        title.append_text(tool_title(self.calls))
        title.truncate(max(1, self.content_size.width), overflow="ellipsis")
        self.query_one(".tool-title", Static).update(title)

    def on_resize(self) -> None:
        self._update_title()

class AgentRow(Static, can_focus=True):
    """A single-line agent selector with persistent, monochrome selection."""

    BINDINGS = [
        Binding("up", "app.previous_agent", show=False),
        Binding("down", "app.next_agent", show=False),
    ]

    class Selected(Message):
        def __init__(self, agent_id: str) -> None:
            super().__init__()
            self.agent_id = agent_id

    def __init__(self, agent_id: str, *, id: str) -> None:
        super().__init__("", id=id)
        self.agent_id = agent_id

    def on_click(self) -> None:
        self.focus()

    def on_focus(self) -> None:
        self.post_message(self.Selected(self.agent_id))


class PraxisApp(App[int]):
    TITLE = "Praxis"
    CSS = """
    Screen { layout: vertical; overflow: hidden; }
    #heading { height: auto; max-height: 3; padding: 0 1; color: $text-muted; }
    #transcript { height: 1fr; padding: 0 1; scrollbar-size: 0 0; }
    #blocks { height: auto; }
    .record { height: auto; margin-bottom: 1; }
    .user-message { background: $foreground 6%; padding: 1 1; margin-top: 1; }
    .tool-title { height: 1; }
    .tool-preview { height: auto; color: $text-muted; }
    #active { height: auto; margin-bottom: 1; }
    #status { height: auto; max-height: 2; color: $text-muted; padding: 0 1; }
    #agents { height: auto; max-height: 6; padding: 0 1; scrollbar-size: 0 0; }
    #agents AgentRow { height: 1; background: transparent; color: $foreground; }
    #agents AgentRow:hover, #agents AgentRow:focus { background: transparent; }
    #composer { height: 1; margin: 1 1 0 1; }
    #prompt { width: 2; height: 1; text-style: bold; }
    #task {
        width: 1fr; height: 1; min-height: 1;
        border: none; padding: 0; margin: 0;
        background: transparent; background-tint: transparent;
    }
    #hint { height: 1; color: $text-muted; padding: 0 1; }
    """
    BINDINGS = [
        Binding("alt+left", "previous_agent", "上一个 agent", priority=True),
        Binding("alt+right", "next_agent", "下一个 agent", priority=True),
        Binding("ctrl+t", "toggle_details", "完整记录", priority=True),
        Binding("ctrl+q", "request_exit", "退出", priority=True),
        Binding("ctrl+c", "request_exit", "退出", priority=True),
    ]

    def __init__(self, sink: ViewEventSink, *, initial_task: str = "") -> None:
        super().__init__()
        self.sink = sink
        self.agent: Agent | None = None
        self.store = ViewStore()
        self.selected_agent_id = "main"
        self.initial_task = initial_task
        self.busy = False
        self.closing = False
        self._runner: Thread | None = None
        self._results: SimpleQueue[tuple[str, str]] = SimpleQueue()
        self.user_input: SimpleQueue[str] = SimpleQueue()
        self._approvals: SimpleQueue[Approval] = SimpleQueue()
        self._approval: Approval | None = None
        self._row_agents: dict[str, str] = {"agent-0": "main"}
        self._mounted_blocks: list = []
        self._mounted_revisions: list[int] = []
        self._mounted_agent = "main"

    def compose(self) -> ComposeResult:
        yield Static("Praxis / 主 agent", id="heading")
        with VerticalScroll(id="transcript"):
            yield Vertical(id="blocks")
            yield Static("", id="active")
        yield Static("", id="status")
        with VerticalScroll(id="agents"):
            yield AgentRow("main", id="agent-0")
        yield ApprovalPanel()
        with Horizontal(id="composer"):
            yield Static("›", id="prompt")
            yield Input(id="task")
        yield Static("Enter 发送 · Ctrl+T 完整记录 · Alt+←/→ 切换 · Ctrl+Q 退出", id="hint")

    async def on_mount(self) -> None:
        self._main_screen = self.screen
        self._main_screen.query_one("#agents", VerticalScroll).display = False
        self._refresh_timer = self.set_interval(0.1, self._tick)
        await self._draw(switching=True)
        self._main_screen.query_one("#task", Input).focus()
        if self.initial_task:
            self._start(self.initial_task)

    def approve_tool_call(self, name: str, arguments: dict) -> bool:
        """Called by execution threads; the UI owns all user input."""
        request = Approval(name, dict(arguments), Future())
        self._approvals.put(request)
        while not self.closing:
            agent_id = arguments.get("agent_id")
            runtime = getattr(self.agent, "collaboration", None)
            if agent_id and runtime is not None:
                handle = runtime.handles.get(agent_id)
                if handle is None or handle.control.cancelled.is_set():
                    break
            try:
                return request.future.result(timeout=0.1)
            except TimeoutError:
                continue
            except CancelledError:
                return False
        request.future.cancel()
        return False

    async def _tick(self) -> None:
        if not self.is_running:
            return
        changed = self.sink.drain(self.store)
        while True:
            try:
                operation, error = self._results.get_nowait()
            except Empty:
                break
            if operation == "closed":
                self.exit(1 if error else 0)
                return
            self.busy = False
            if operation == "clear" and not error:
                self.sink.discard_pending()
                self.store = ViewStore()
                self.selected_agent_id = "main"
                self._mounted_blocks = []
                await self._main_screen.query_one("#blocks", Vertical).remove_children()
                agents = self._main_screen.query_one("#agents", VerticalScroll)
                await agents.remove_children()
                await agents.mount(AgentRow("main", id="agent-0"))
                self._row_agents = {"agent-0": "main"}
            if error:
                view = self.store.views["main"]
                if view.state != "failed":
                    view._commit(Text(error, style="red"))
                    view.state = "failed"
                    view.finished_at = monotonic()
                    view._set_phase("执行出错")
            changed.add("main")
            field = self._main_screen.query_one("#task", Input)
            field.disabled = self.closing
            field.placeholder = ""
            if self.selected_agent_id == "main" and not self.closing and self._approval is None:
                field.focus()
        # 最后一次队列检查之后到达的输入，在本轮退出后继续执行。
        if not self.busy and not self.closing:
            try:
                task = self.user_input.get_nowait()
            except Empty:
                pass
            else:
                self._start(task)
        await self._update_agents()
        if self.selected_agent_id in changed:
            await self._draw()
        if isinstance(self.screen, DetailsScreen) and self.screen.agent_id in changed:
            self.screen.update_content(self.store.get(self.screen.agent_id).transcript_content())
        # Refresh the clock and bounded command previews without rebuilding history.
        view = self.store.get(self.selected_agent_id)
        self._main_screen.query_one("#active", Static).update(view.active_content())
        status = Text(f"{view.phase} · 本轮 {_duration(view.elapsed())}")
        status_widget = self._main_screen.query_one("#status", Static)
        status_widget.display = view.state != "idle"
        status_widget.update(Group(
            status, *([view._capacity] if view._capacity is not None else []),
        ))
        if self.closing:
            self._main_screen.query_one("#hint", Static).update("正在关闭 agent 和命令，请稍候……")
        elif not self.user_input.empty():
            self._main_screen.query_one("#hint", Static).update(
                f"{self.user_input.qsize()} 条补充已排队，当前操作结束后处理 · Ctrl+Q 退出"
            )
        else:
            self._main_screen.query_one("#hint", Static).update(
                "Enter 发送 · Ctrl+T 完整记录 · Alt+←/→ 切换 · Ctrl+Q 退出"
            )
        self._show_approval()

    async def _update_agents(self) -> None:
        agents = self._main_screen.query_one("#agents", VerticalScroll)
        known = set(self._row_agents.values())
        for agent_id in self.store.views:
            if agent_id not in known:
                row_id = f"agent-{len(self._row_agents)}"
                self._row_agents[row_id] = agent_id
                await agents.mount(AgentRow(agent_id, id=row_id))
        agents.display = len(self.store.views) > 1
        agents.styles.height = min(6, len(self.store.views))
        for row_id, agent_id in self._row_agents.items():
            view = self.store.views[agent_id]
            row = agents.query_one(f"#{row_id}", AgentRow)
            selected = agent_id == self.selected_agent_id
            width = max(1, row.content_size.width)
            duration = Text(_duration(view.elapsed()), style="dim")
            name = Text(view.label, style="bold" if selected else "")
            name.truncate(min(12, max(1, width // 3)), overflow="ellipsis", pad=True)
            label = Text("● " if selected else "  ")
            label.append_text(name)
            label.append("  ")
            phase = Text(view.phase, style="dim")
            phase.truncate(max(0, width - label.cell_len - duration.cell_len - 1),
                           overflow="ellipsis", pad=True)
            label.append_text(phase)
            label.append(" ")
            label.append_text(duration)
            label.truncate(width, overflow="ellipsis")
            row.update(label)

    def _block_widget(self, index: int, block):
        if isinstance(block, tuple):
            return ToolRecord(block, index)
        classes = "record user-message" if isinstance(block, MessageBlock) and block.user else "record"
        return Static(block, id=f"record-{index}", classes=classes)

    async def _draw(self, *, switching: bool = False) -> None:
        view = self.store.get(self.selected_agent_id)
        pane = self._main_screen.query_one("#transcript", VerticalScroll)
        if not switching:
            view.scroll_position = pane.scroll_y
            view.follow_output = pane.is_vertical_scroll_end
        blocks = self._main_screen.query_one("#blocks", Vertical)
        old = self._mounted_blocks
        new = view.display_blocks()
        if self._mounted_agent != self.selected_agent_id or len(new) < len(old):
            await blocks.remove_children()
            old = []
        for index, block in enumerate(new[:len(old)]):
            if isinstance(block, MessageBlock):
                if block.revision != self._mounted_revisions[index]:
                    blocks.query_one(f"#record-{index}", Static).update(block)
            elif block != old[index]:
                blocks.query_one(f"#record-{index}", ToolRecord).update_calls(block)
        if len(new) > len(old):
            await blocks.mount(*(self._block_widget(i, new[i]) for i in range(len(old), len(new))))
        self._mounted_blocks = list(new)
        self._mounted_revisions = [block.revision if isinstance(block, MessageBlock) else 0 for block in new]
        self._mounted_agent = self.selected_agent_id
        title = f"Praxis / {view.label}"
        if view.task:
            title += f" · {view.task}"
        self._main_screen.query_one("#heading", Static).update(Text(title))
        self._main_screen.query_one("#active", Static).update(view.active_content())
        self.call_after_refresh(self._restore_scroll, self.selected_agent_id,
                                view.follow_output, view.scroll_position,
                                None if switching else view.scroll_position)

    def _restore_scroll(self, agent_id: str, follow: bool, position: float,
                        expected_position: float | None = None) -> None:
        if self.selected_agent_id != agent_id:
            return
        pane = self._main_screen.query_one("#transcript", VerticalScroll)
        # A deferred refresh must not override scrolling since this draw began.
        if expected_position is not None and pane.scroll_y != expected_position:
            return
        if follow:
            pane.scroll_end(animate=False, immediate=True)
        else:
            pane.scroll_to(y=position, animate=False, immediate=True)

    @on(AgentRow.Selected)
    async def on_agent_selected(self, event: AgentRow.Selected) -> None:
        agent_id = event.agent_id
        if agent_id not in self.store.views or agent_id == self.selected_agent_id:
            return
        pane = self._main_screen.query_one("#transcript", VerticalScroll)
        previous = self.store.get(self.selected_agent_id)
        previous.scroll_position = pane.scroll_y
        previous.follow_output = pane.is_vertical_scroll_end
        self.selected_agent_id = agent_id
        await self._update_agents()
        await self._draw(switching=True)

    def action_toggle_details(self) -> None:
        if isinstance(self.screen, DetailsScreen):
            self.screen.dismiss()
        elif self._approval is not None:
            request = self._approval
            view = self.store.get(request.arguments.get("agent_id", "main"))
            self.push_screen(DetailsScreen(
                f"{view.label} · {request.tool} · 完整参数",
                Text(json.dumps(request.arguments, ensure_ascii=False, indent=2)), request=request,
            ))
        else:
            view = self.store.get(self.selected_agent_id)
            self.push_screen(DetailsScreen(
                f"{view.label} · 完整执行记录", view.transcript_content(), agent_id=view.agent_id,
            ))

    def _select_relative(self, direction: int) -> None:
        ids = list(self._row_agents.values())
        agent_id = ids[(ids.index(self.selected_agent_id) + direction) % len(ids)]
        row_id = next(key for key, value in self._row_agents.items() if value == agent_id)
        row = self._main_screen.query_one(f"#{row_id}", AgentRow)
        if isinstance(self.focused, AgentRow):
            row.focus()
        row.scroll_visible(animate=False)
        self.post_message(AgentRow.Selected(agent_id))

    def action_previous_agent(self) -> None:
        self._select_relative(-1)

    def action_next_agent(self) -> None:
        self._select_relative(1)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        task = event.value.strip()
        if not task or self.closing or self.agent is None:
            return
        if task.lower() == "/clear" and self.busy:
            self.notify("请等待当前任务结束后再清空历史。", severity="warning")
            return
        event.input.value = ""
        if task.lower() == "/exit":
            self.action_request_exit()
        elif self.busy:
            self.user_input.put(task)
        else:
            self._start(task)

    def _start(self, task: str) -> None:
        if self.agent is None or self.busy or self.closing:
            return
        self.busy = True
        operation = "clear" if task.lower() == "/clear" else "run"
        self.agent.user_input = self.user_input
        field = self._main_screen.query_one("#task", Input)
        field.disabled = operation == "clear"
        field.placeholder = "补充主任务要求（当前操作结束后处理）" if operation == "run" else ""

        def execute() -> None:
            error = ""
            try:
                if operation == "clear":
                    self.agent.clear_history()
                else:
                    self.agent.run(task)
            except Exception as failure:
                error = f"{type(failure).__name__}: {failure}"
            finally:
                self._results.put((operation, error))

        self._runner = Thread(target=execute, name="praxis-main", daemon=True)
        self._runner.start()

    def _show_approval(self) -> None:
        if self._approval is not None:
            if not self._approval.future.done():
                return
            if isinstance(self.screen, DetailsScreen) and self.screen.request is self._approval:
                self.screen.dismiss()
            self._hide_approval()
        if self.closing:
            return
        if self.screen is not self._main_screen:
            return
        while True:
            try:
                request = self._approvals.get_nowait()
            except Empty:
                return
            if not request.future.done():
                break
        self._approval = request
        view = self.store.get(request.arguments.get("agent_id", "main"))

        self._main_screen.query_one("#composer").display = False
        self._main_screen.query_one("#hint").display = False
        self._main_screen.query_one(ApprovalPanel).show_request(request, view.label)

    @on(ApprovalPanel.Decided)
    def on_approval_decided(self, event: ApprovalPanel.Decided) -> None:
        if event.request is self._approval:
            if not event.request.future.done():
                event.request.future.set_result(event.allowed)
            self._hide_approval()

    def _hide_approval(self) -> None:
        self._approval = None
        panel = self._main_screen.query_one(ApprovalPanel)
        panel.request = None
        panel.display = False
        self._main_screen.query_one("#composer").display = True
        self._main_screen.query_one("#hint").display = True
        if not self.closing:
            self._main_screen.query_one("#task", Input).focus()

    def action_request_exit(self) -> None:
        if self.closing:
            return
        self.closing = True
        if self._approval is not None and not self._approval.future.done():
            self._approval.future.cancel()

        def close() -> None:
            error = ""
            try:
                if self.agent is not None:
                    self.agent.close()
                if self._runner is not None:
                    self._runner.join()
            except Exception as failure:
                error = str(failure)
            self._results.put(("closed", error))

        Thread(target=close, name="praxis-close", daemon=True).start()

    def on_unmount(self) -> None:
        self._refresh_timer.stop()
        self.closing = True
        if self._approval is not None:
            self._approval.future.cancel()
