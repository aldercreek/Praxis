"""One Textual screen with memory-only, independently scrollable agent pages."""

from __future__ import annotations

import json
import sys
from concurrent.futures import CancelledError, Future, InvalidStateError
from dataclasses import dataclass
from queue import Empty, Queue, SimpleQueue
from threading import Lock, Thread
from time import monotonic
from typing import TYPE_CHECKING, ClassVar

import pyperclip
from rich.console import Group, RenderableType
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, HorizontalScroll, Vertical, VerticalScroll
from textual.geometry import Region, Size
from textual.layouts.vertical import VerticalLayout
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.theme import Theme
from textual.widgets import OptionList, Static

from praxis_agent.execution import ExecutionStopped, Run, current_handle, run_operation
from praxis_agent.terminal.cells import (
    ACCENT,
    MessageBlock,
    WelcomeHeader,
    approval_content,
    tool_preview,
    tool_title,
)
from praxis_agent.terminal.composer import Composer
from praxis_agent.terminal.preview import OutputPreview
from praxis_agent.terminal.screen import ConversationScreen
from praxis_agent.terminal.selection import SelectableText
from praxis_agent.terminal.views import ToolBlock, ViewEventSink, ViewStore

if TYPE_CHECKING:
    from praxis_agent.agent import Agent


@dataclass
class Approval:
    tool: str
    arguments: dict
    future: Future


class FooterHint(Static):
    """Temporarily replace a footer hint without a popup or a layout change."""

    def __init__(self, *, id: str) -> None:
        super().__init__("", id=id)
        self._hint = Text()
        self._flash_timer = None

    def update_hint(self, hint: Text) -> None:
        self._hint = hint
        if self._flash_timer is None and self.renderable != hint:
            self.update(hint)

    def flash(self, text: str, *, error: bool = False) -> None:
        self.clear_flash()
        self.update(Text(text, style="red" if error else "", no_wrap=True, overflow="ellipsis"))
        self._flash_timer = self.set_timer(3, self.clear_flash)

    @on(events.Hide)
    def clear_flash(self) -> None:
        if self._flash_timer is not None:
            self._flash_timer.stop()
            self._flash_timer = None
        self.update(self._hint)


class ApprovalPanel(Vertical):
    """An inline choice in the bottom pane; decisions still resolve the existing Future."""

    DEFAULT_CSS = """
    ApprovalPanel { display: none; height: auto; padding: 0 1; }
    #approval-heading { height: 1; text-style: bold; }
    #approval-content { height: auto; }
    #approval-overflow { height: 1; color: $text-muted; }
    #approval-choices { height: 2; min-height: 2; border: none; padding: 0;
                        background: transparent; scrollbar-size: 0 0; }
    #approval-choices > .option-list--option { padding: 0; }
    #approval-choices > .option-list--option-highlighted {
        background: transparent; color: $foreground; text-style: bold;
    }
    #approval-choices > .option-list--option-hover { background: transparent; }
    #approval-hint { height: 1; color: $text-muted; }
    """
    BINDINGS: ClassVar[list[Binding]] = [
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
        self.label = ""

    def compose(self) -> ComposeResult:
        yield Static("", id="approval-heading")
        yield SelectableText("", id="approval-content")
        yield Static("", id="approval-overflow")
        yield OptionList("1. 允许本次 (y)", "2. 拒绝 (n)", id="approval-choices")
        yield FooterHint(id="approval-hint")

    def show_request(self, request: Approval, label: str) -> None:
        self.request = request
        self.label = label
        self.display = True
        self.fit_content()
        choices = self.query_one(OptionList)
        choices.highlighted = 0
        choices.focus()

    def on_resize(self) -> None:
        self.fit_content()

    def fit_content(self) -> None:
        if self.request is None:
            return
        width = max(1, self.app.size.width - 2)
        question = "是否执行命令？" if self.request.tool == "run_command" else "是否允许操作？"
        heading = Text(f"{self.label} · {question}", no_wrap=True, overflow="ellipsis")
        self.query_one("#approval-heading", Static).update(heading)
        content = approval_content(self.request.tool, self.request.arguments)
        rows = self.app.console.render_lines(content, self.app.console.options.update(width=width), pad=False)
        # Reserve one history row plus question, two choices and confirmation hints.
        budget = max(0, min(10, self.app.size.height - 5))
        hidden = len(rows) > budget
        shown = max(0, budget - int(hidden))
        self.query_one("#approval-content", Static).update(
            Group(*(Text.assemble(*((s.text, s.style) for s in row)) for row in rows[:shown]))
        )
        self.query_one("#approval-content").display = shown > 0
        overflow = self.query_one("#approval-overflow", Static)
        overflow.display = hidden and budget > 0
        overflow.update(Text(f"… 省略 {len(rows) - shown} 行 · Ctrl+T 查看全文",
                             no_wrap=True, overflow="ellipsis"))
        self.query_one("#approval-hint", FooterHint).update_hint(shortcut_line([
            ("Enter", "确认"), ("Esc", "停止"), ("↑/↓", "选择"), ("Ctrl+T", "详情"),
        ], width))

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
    BINDINGS: ClassVar[list[Binding]] = [Binding("escape,q", "close", "返回", show=False)]

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
            yield SelectableText(self.content, id="details-content")
        yield FooterHint(id="details-hint")

    def on_mount(self) -> None:
        self.query_one("#details-scroll", VerticalScroll).focus()
        self.on_resize()

    def on_resize(self) -> None:
        if self.is_mounted:
            self.query_one("#details-hint", FooterHint).update_hint(shortcut_line([
                ("Ctrl+T", "返回"), ("Esc", "停止" if self.app.busy else "返回"),
                ("↑/↓", "滚动"), ("Ctrl+C", "复制选中文字"),
            ], max(1, self.app.size.width - 2)))

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


def shortcut_line(actions: list[tuple[str, str]], width: int) -> Text:
    """Keep whole hints in priority order; never clip a key or leave a separator."""
    line = Text(no_wrap=True)
    for key, description in actions:
        item = Text(key, style="bold") + Text((" " if key else "") + description)
        separator = " · " if line else ""
        if line.cell_len + len(separator) + item.cell_len <= width:
            line.append(separator)
            line.append_text(item)
    return line


class ToolRecord(Vertical):
    """A selectable activity summary; full results live in the transcript viewer."""

    def __init__(self, calls: tuple[ToolBlock, ...], index: int) -> None:
        super().__init__(id=f"record-{index}", classes="record")
        self.calls = calls

    def compose(self) -> ComposeResult:
        yield SelectableText(self._title(), classes="tool-title")
        preview = SelectableText(tool_preview(self.calls), classes="tool-preview")
        preview.display = bool(preview.renderable)
        yield preview

    def update_calls(self, calls: tuple[ToolBlock, ...]) -> None:
        self.calls = calls
        title = self.query_one(".tool-title", Static)
        content = self._title()
        if title.renderable != content:
            title.update(content)
        preview = tool_preview(calls)
        widget = self.query_one(".tool-preview", Static)
        if widget.renderable != preview:
            widget.update(preview)
        widget.display = bool(preview)

    def _title(self) -> OutputPreview:
        style = "bold red" if any(call.failed for call in self.calls) else (
            "dim" if any(call.state in {"running", "cancelled"} for call in self.calls) else "bold green")
        # Wrap during rendering, when Rich has the actual layout width, including the first frame.
        return OutputPreview([tool_title(self.calls)], max_rows=2, prefix="• ", style="", prefix_style=style)

class AgentStrip(HorizontalScroll):
    """Keep agent navigation on one line, including ordinary mouse-wheel input."""

    def move(self, direction: int) -> None:
        self.scroll_relative(x=direction * max(4, self.size.width // 3),
                             animate=False, immediate=True)

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.stop()
        event.prevent_default()
        self.move(1)

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.stop()
        event.prevent_default()
        self.move(-1)

    def watch_scroll_x(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_x(old_value, new_value)
        self.call_after_refresh(self.refresh_edges)

    def on_resize(self) -> None:
        self.call_after_refresh(self.refresh_edges)
        self.call_after_refresh(self.reveal_selected)

    def reveal_selected(self) -> None:
        if self.parent.display:
            for tab in self.query(".selected"):
                tab.scroll_visible(animate=False, immediate=True)

    def refresh_edges(self) -> None:
        self.parent.query_one("#agents-prev").styles.visibility = "visible" if self.scroll_x > 0 else "hidden"
        self.parent.query_one("#agents-next").styles.visibility = (
            "visible" if self.scroll_x < self.max_scroll_x else "hidden")


class AgentTab(Static, can_focus=True):
    """An agent selector whose selected appearance is independent of keyboard focus."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("up", "app.previous_agent", show=False),
        Binding("down", "app.next_agent", show=False),
        Binding("left", "app.previous_agent", show=False),
        Binding("right", "app.next_agent", show=False),
    ]

    class Selected(Message):
        def __init__(self, agent_id: str) -> None:
            super().__init__()
            self.agent_id = agent_id

    def __init__(self, agent_id: str, *, id: str) -> None:
        super().__init__("", id=id)
        self.agent_id = agent_id

    def on_focus(self) -> None:
        self.post_message(self.Selected(self.agent_id))


class ConversationLayout(VerticalLayout):
    """Keep the welcome page at the top; anchor conversation controls at the bottom."""

    def arrange(self, parent, children, size: Size):
        placements = super().arrange(parent, children, size)
        if not parent.app.empty_session:
            return placements
        for index, placement in enumerate(placements):
            if placement.widget.id != "transcript":
                continue
            pane = placement.widget
            region = placement.region
            gutter = pane.styles.gutter
            content_height = pane.get_content_height(
                region.size, parent.app.size, max(0, region.width - gutter.width))
            unused = max(0, region.height - content_height - gutter.height)
            placements[index] = placement._replace(
                region=Region(region.x, region.y, region.width, region.height - unused))
            for following in range(index + 1, len(placements)):
                item = placements[following]
                placements[following] = item._replace(region=item.region.translate((0, -unused)))
            break
        return placements


class PraxisApp(App[int]):
    TITLE = "Praxis"
    CSS = """
    Screen, Screen:ansi {
        layout: vertical; overflow: hidden; background: $background; color: $foreground;
    }
    #transcript { width: 100%; height: 1fr; min-height: 0; padding: 0 1; scrollbar-size: 0 0; }
    #welcome { display: none; height: auto; padding: 1 0 0 0; }
    #blocks { height: auto; }
    .record { height: auto; margin-bottom: 1; }
    .tool-title { height: auto; text-style: bold; }
    .tool-preview { height: auto; color: $text-muted; }
    #jump-latest { display: none; height: 1; text-align: center; color: $text-muted; }
    #jump-latest:hover { color: $foreground; }
    #controls { width: 100%; height: auto; padding: 1 0 0 0; }
    #controls.approving { padding-top: 0; }
    #status { height: auto; max-height: 2; color: $text-muted; padding: 0 1; }
    #pending { height: auto; max-height: 3; color: $text-muted; padding: 0 1; }
    #agent-bar { height: 1; }
    #agents-prev, #agents-next { width: 1; height: 1; color: $text-muted; visibility: hidden; }
    #agents-prev:hover, #agents-next:hover { color: $foreground; }
    #agents { width: 1fr; height: 1; scrollbar-size: 0 0; }
    #agents AgentTab { width: auto; height: 1; margin-right: 2;
                       background: transparent; color: $text-muted; }
    #agents AgentTab:hover, #agents AgentTab:focus { color: $foreground; }
    #agents AgentTab.selected { color: $primary; text-style: bold; }
    #composer { height: auto; padding: 1; }
    #prompt { width: 2; height: 1; color: $text-muted; }
    #task {
        width: 1fr; height: 1; min-height: 1;
        border: none; padding: 0; margin: 0;
        background: transparent; background-tint: transparent; scrollbar-size: 0 0;
    }
    #task:focus { border: none; }
    #shortcuts { height: auto; max-height: 12; padding: 0 3; color: $text-muted; scrollbar-size: 0 0; }
    #hint { height: 1; color: $text-muted; padding: 0 1; }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("alt+left", "previous_agent", "上一个 agent", priority=True),
        Binding("alt+right", "next_agent", "下一个 agent", priority=True),
        Binding("ctrl+t", "toggle_details", "完整记录", priority=True),
        Binding("ctrl+q", "request_exit", "退出", priority=True),
        Binding("ctrl+c", "interrupt", "停止或退出", priority=True),
        Binding("ctrl+shift+c", "copy_selection", "复制选中文字", priority=True),
        Binding("escape", "stop_run", "停止本轮", priority=True),
        Binding("ctrl+r", "resume_pending", "发送保留的补充", priority=True),
        Binding("f1", "show_shortcuts", "快捷键", priority=True),
        Binding("shift+left", "edit_pending", "编辑待发送消息", priority=True),
        Binding("pageup", "scroll_history(-1)", "上翻", priority=True),
        Binding("pagedown", "scroll_history(1)", "下翻", priority=True),
        Binding("ctrl+home", "history_edge(False)", "最早记录", priority=True),
        Binding("ctrl+end", "history_edge(True)", "回到最新", priority=True),
        Binding("alt+up", "jump_message(-1)", "上一条输入", priority=True),
        Binding("alt+down", "jump_message(1)", "下一条输入", priority=True),
    ]

    def __init__(self, sink: ViewEventSink, *, initial_task: str = "") -> None:
        super().__init__()
        self.register_theme(Theme(
            name="praxis", primary=ACCENT, foreground="#f2f2f2",
            background="#0c0c0c", surface="#0c0c0c", panel="#0c0c0c",
            success="ansi_green", error="ansi_red", variables={"text-muted": "#a0a0a0"},
        ))
        self.theme = "praxis"
        self.ansi_color = True
        self.sink = sink
        self.agent: Agent | None = None
        self.store = ViewStore()
        self.selected_agent_id = "main"
        self.initial_task = initial_task
        self.busy = False
        self.closing = False
        self.stopping = False
        self.current_run: Run | None = None
        self.paused_inputs: list[str] = []
        self._runner: Thread | None = None
        self._results: SimpleQueue[tuple[str | None, str, str]] = SimpleQueue()
        self.user_input: Queue[str] = Queue()
        self._approvals: SimpleQueue[Approval] = SimpleQueue()
        self._approval: Approval | None = None
        self._approval_lock = Lock()
        self._approval_handles = set()
        self._tab_agents: dict[str, str] = {"agent-0": "main"}
        self._mounted_blocks: list = []
        self._mounted_revisions: list[int] = []
        self._mounted_agent = "main"
        self._switching_page = False
        self._selection_widget: SelectableText | None = None
        self._shortcuts_open = False
        self._input_history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""
        self._cursor_visible = False
        self._cursor_style_set = False
        self._show_terminal_cursor = False

    def get_default_screen(self) -> Screen:
        return ConversationScreen()

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="transcript"):
            yield Static("", id="welcome")
            yield Vertical(id="blocks")
        yield Static(Text("↓ 回到最新 · Ctrl+End", no_wrap=True, overflow="ellipsis"), id="jump-latest")
        with Vertical(id="controls"):
            with Horizontal(id="agent-bar"):
                yield Static("‹", id="agents-prev")
                with AgentStrip(id="agents"):
                    yield AgentTab("main", id="agent-0")
                yield Static("›", id="agents-next")
            yield Static("", id="status")
            yield Static("", id="pending")
            yield ApprovalPanel()
            with Horizontal(id="composer"):
                yield Static("›", id="prompt")
                yield Composer()
            with VerticalScroll(id="shortcuts"):
                yield Static("", id="shortcut-content")
            yield FooterHint(id="hint")

    @property
    def empty_session(self) -> bool:
        main = self.store.views["main"]
        return (len(self.store.views) == 1 and main.state == "idle" and not main.committed_blocks
                and not (self.busy or self.closing or self._approval))

    async def on_mount(self) -> None:
        self._main_screen = self.screen
        self._main_screen.styles.layout = ConversationLayout()
        self._input_history = [block.text for block in self.store.views["main"].committed_blocks
                               if isinstance(block, MessageBlock) and block.user]
        self._main_screen.query_one("#agent-bar").display = False
        self.set_interval(0.1, self._tick)
        if sys.platform == "win32" and not self.is_headless and self._driver is not None:
            # Textual 1's Windows driver does not probe synchronized output itself.
            self._driver.write("\x1b[?2026$p")
            self._driver.flush()
        await self._draw(switching=True)
        self._main_screen.query_one("#task", Composer).focus()
        if self.initial_task:
            self._input_history.append(self.initial_task)
            self._start(self.initial_task)
        self._refresh_bottom()

    def _display(self, screen: Screen, renderable: RenderableType | None) -> None:
        focused = self.focused
        show_cursor = (
            screen is self.screen and isinstance(focused, Composer)
            and focused.is_on_screen and not focused.read_only and not focused.disabled
            and focused.content_region.contains_point(focused.cursor_screen_offset)
        )
        if show_cursor:
            # Layout can move the editor without moving its text selection.
            self.cursor_position = focused.cursor_screen_offset
        self._show_terminal_cursor = show_cursor
        super()._display(screen, renderable)

    def _begin_update(self) -> None:
        super()._begin_update()
        if self._driver is not None and self._cursor_visible:
            self._driver.write("\x1b[?25l")
            self._cursor_visible = False

    def _end_update(self) -> None:
        if self._driver is not None and self._show_terminal_cursor:
            if not self._cursor_style_set:
                self._driver.write("\x1b[6 q")
                self._cursor_style_set = True
            self._driver.write("\x1b[?25h")
            self._cursor_visible = True
        super()._end_update()

    def approve_tool_call(self, name: str, arguments: dict) -> bool:
        """Called by execution threads; the UI owns all user input."""
        if current_handle() is None:
            return run_operation("approval", lambda handle: self.approve_tool_call(name, arguments))
        request = Approval(name, dict(arguments), Future())
        handle = current_handle()
        with self._approval_lock:
            if self.closing:
                return False
            self._approval_handles.add(handle)
        unsubscribe = handle.on_cancel(request.future.cancel)
        self._approvals.put(request)
        try:
            allowed = request.future.result()
            handle.check()
            return allowed
        except CancelledError:
            if self.closing:
                return False
            raise ExecutionStopped("approval cancelled")
        finally:
            unsubscribe()
            request.future.cancel()
            with self._approval_lock:
                self._approval_handles.discard(handle)

    def _cancel_approvals(self):
        with self._approval_lock:
            handles = tuple(self._approval_handles)
        for handle in handles:
            handle.cancel()

    async def _tick(self) -> None:
        with self.batch_update():
            await self._update_views()

    async def _update_views(self) -> None:
        if not self.is_running:
            return
        self._remember_scroll()
        changed = self.sink.drain(self.store)
        while True:
            try:
                run_id, operation, error = self._results.get_nowait()
            except Empty:
                break
            if operation == "closed":
                self.exit(1 if error else 0)
                return
            if run_id is not None and (self.current_run is None or run_id != self.current_run.id):
                continue
            self._show_approval()
            self.busy = False
            self.stopping = False
            changed.update(self.sink.drain(self.store))
            if operation == "cancelled":
                view = self.store.views["main"]
                view.state = "cancelled"
                view.finished_at = monotonic()
                view._phase = "已停止"
            if operation == "clear" and not error:
                self.sink.discard_pending()
                self.store = ViewStore()
                self.selected_agent_id = "main"
                self._mounted_blocks = []
                await self._main_screen.query_one("#blocks", Vertical).remove_children()
                agents = self._main_screen.query_one(AgentStrip)
                await agents.remove_children()
                await agents.mount(AgentTab("main", id="agent-0"))
                self._tab_agents = {"agent-0": "main"}
            if error:
                view = self.store.views["main"]
                if view.state != "failed":
                    view.committed_blocks.append(Text(error, style="red"))
                    view.state = "failed"
                    view.finished_at = monotonic()
                    view._phase = "执行出错"
            changed.add("main")
            field = self._main_screen.query_one("#task", Composer)
            field.disabled = self.closing
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
        self._show_approval()
        self._refresh_bottom()

    def _refresh_bottom(self) -> None:
        view = self.store.get(self.selected_agent_id)
        screen = self._main_screen
        width = max(1, self.size.width - 2)
        approval = self._approval is not None
        is_main = self.selected_agent_id == "main"
        controls = screen.query_one("#controls")
        controls.set_class(approval, "approving")
        agents = screen.query_one("#agent-bar")
        agents.display = len(self.store.views) > 1 and not approval
        screen.query_one("#composer").display = is_main and not approval
        status = screen.query_one("#status", Static)
        running = view.state in {"running", "closing"} or (
            self.busy and is_main and view.state not in {"ended", "failed", "cancelled"})
        status.display = not approval and not self.empty_session
        if status.display:
            settling = view.state == "closing" or running and (self.stopping or self.closing)
            if settling:
                phase = "正在关闭" if self.closing or view.state == "closing" else "正在停止 · 等待操作清理"
            else:
                phase = "处理中" if view.has_active_tools else view.phase
            text = Text(f"• {phase}")
            if view.started_at is not None:
                text.append(f" · {'已运行' if running else '耗时'} {_duration(view.elapsed())}")
            text.truncate(width, overflow="ellipsis")
            if view._capacity is not None:
                capacity = view._capacity.copy()
                capacity.truncate(width, overflow="ellipsis")
                text.append(" · " if text.cell_len + capacity.cell_len + 3 <= width else "\n")
                text.append_text(capacity)
            if status.renderable != text:
                status.update(text)
        editor = screen.query_one(Composer)
        editor.read_only = not is_main
        with self.user_input.mutex:
            queued = list(self.user_input.queue)
        retained = self.paused_inputs
        preview = screen.query_one("#pending", Static)
        preview.display = is_main and bool(retained or queued) and not approval
        if preview.display:
            pending = [("已保留", item) for item in retained] + [("已排队", item) for item in queued]
            lines = [Text(f"› {' '.join(item.split())}  ({label})", no_wrap=True, overflow="ellipsis")
                     for label, item in pending[:2]]
            if len(pending) > 2:
                lines.append(Text(f"  还有 {len(pending) - 2} 条待发送"))
            content = Text("\n").join(lines)
            if preview.renderable != content:
                preview.update(content)
        self._refresh_shortcuts(editor, approval)
        shortcuts = screen.query_one("#shortcuts")
        welcome_space = (self.size.height - int(editor.styles.height.value)
                         - screen.query_one("#composer").styles.padding.height - 2)
        if shortcuts.display:
            welcome_space -= int(shortcuts.styles.max_height.value)
        if preview.display:
            welcome_space -= min(3, len(retained) + len(queued))
        welcome = screen.query_one("#welcome", Static)
        welcome.display = self.empty_session and welcome_space >= 4
        if welcome.display:
            content = WelcomeHeader(
                getattr(self.agent, "workspace", None), getattr(self.agent, "session_model", None),
                compact=welcome_space < 7,
            )
            if welcome.renderable != content:
                welcome.update(content)
        hint = screen.query_one("#hint", FooterHint)
        hint.display = not approval
        if editor.read_only:
            actions = [("Alt+←/→", "切换 Agent")]
        elif self.stopping or self.closing:
            actions = [("", "可继续编辑，停止后发送")]
        elif retained and not running:
            actions = [("Ctrl+R", "发送保留输入"), ("Shift+←", "取回编辑")]
        elif queued and not editor.text:
            actions = [("Shift+←", "取回编辑"), ("?", "快捷键")]
        elif editor.text:
            actions = [("Enter", "发送补充" if running else "发送"), ("Shift+Enter", "换行")]
        elif running:
            actions = [("Enter", "发送补充")]
        else:
            actions = [("/", "命令"), ("?", "快捷键")]
        if self.busy and not (self.stopping or self.closing):
            actions.insert(1, ("Esc", "停止本轮"))
        hint.update_hint(shortcut_line(actions, width))

    def on_resize(self) -> None:
        if hasattr(self, "_main_screen"):
            self._refresh_bottom()

    @on(Composer.Changed)
    def refresh_composer_hint(self) -> None:
        if hasattr(self, "_main_screen"):
            editor = self._main_screen.query_one(Composer)
            if self._history_index is not None and editor.text != self._input_history[self._history_index]:
                self._history_index = None
            self._refresh_bottom()

    def action_show_shortcuts(self) -> None:
        if self.screen is not self._main_screen:
            return
        self._shortcuts_open = not self._shortcuts_open
        self._refresh_bottom()

    def _refresh_shortcuts(self, editor: Composer, approval: bool) -> None:
        widget = self._main_screen.query_one("#shortcuts", VerticalScroll)
        commands = self._matching_commands(editor.text)
        widget.display = not approval and not editor.read_only and bool(self._shortcuts_open or commands)
        if not widget.display:
            return
        actions = [
            ("Enter", "发送消息；执行中发送补充"),
            ("Shift+Enter", "换行；粘贴多行会保留换行"),
            ("Esc", "停止当前轮次，等待资源清理后继续输入"),
            ("Ctrl+R", "发送停止后保留的补充"),
            ("Shift+←", "取回下一条待发送消息，输入框为空时可用"),
            ("↑/↓", "在首行/末行回看输入历史"),
            ("PgUp/PgDn · Ctrl+Home/End", "滚动记录 · 最早/最新"),
            ("Alt+↑/↓", "跳到上一条/下一条用户消息"),
            ("Ctrl+T", "完整记录；审批时查看完整参数"),
            ("Alt+←/→", "切换 Agent"),
            ("Ctrl+C / Ctrl+Shift+C", "拖选后复制文字"),
            ("Ctrl+C", "无选区时停止或退出"), ("Ctrl+Q", "退出 Praxis"),
            ("F1 / 空输入时 ?", "快捷键帮助"),
        ]
        # History, controls, editor padding and footer all need their own rows.
        reserved = 3 + int(editor.styles.height.value) + editor.parent.styles.padding.height
        for selector in ("#agent-bar", "#status", "#pending"):
            row = self._main_screen.query_one(selector)
            if row.display:
                height = row.get_content_height(self.size, self.size, max(1, self.size.width - 2))
                if selector == "#agent-bar":
                    height = 1
                reserved += height
        widget.styles.max_height = max(1, min(12, self.size.height // 2, self.size.height - reserved))
        content = Text("\n").join(
            Text.assemble((key, "bold"), "  " + description, no_wrap=True, overflow="ellipsis")
            for key, description in (commands or actions))
        help_text = widget.query_one(Static)
        if help_text.renderable != content:
            help_text.update(content)

    @staticmethod
    def _matching_commands(text: str) -> list[tuple[str, str]]:
        return [(name, description) for name, description in (
            ("/clear", "清空当前会话"), ("/help", "快捷键帮助"), ("/exit", "退出 Praxis"),
        ) if text.startswith("/") and name.startswith(text)]

    def action_complete_command(self) -> None:
        editor = self._main_screen.query_one(Composer)
        matches = self._matching_commands(editor.text)
        if len(matches) == 1:
            editor.load_draft(matches[0][0])

    def recall_input(self, direction: int) -> bool:
        if not self._input_history:
            return False
        editor = self._main_screen.query_one(Composer)
        if self._history_index is None:
            if direction > 0:
                return False
            self._history_draft = editor.text
            index = len(self._input_history) - 1
        else:
            index = max(0, self._history_index + direction)
        self._history_index = index if index < len(self._input_history) else None
        editor.load_draft(self._input_history[index] if self._history_index is not None else self._history_draft)
        return True

    def action_edit_pending(self) -> None:
        editor = self._main_screen.query_one(Composer)
        if self.screen is not self._main_screen or editor.read_only or editor.text or self.closing:
            return
        if self.paused_inputs:
            task = self.paused_inputs.pop(0)
        else:
            try:
                task = self.user_input.get_nowait()
            except Empty:
                return  # The agent may already have accepted this message.
        editor.load_draft(task)
        self._refresh_bottom()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "edit_pending":
            editor = self.focused
            return (isinstance(editor, Composer) and not editor.read_only and not editor.text
                    and bool(self.paused_inputs or not self.user_input.empty()))
        return super().check_action(action, parameters)

    @on(events.Click, "#agents-prev")
    def scroll_agents_left(self) -> None:
        self._main_screen.query_one(AgentStrip).move(-1)

    @on(events.Click, "#agents-next")
    def scroll_agents_right(self) -> None:
        self._main_screen.query_one(AgentStrip).move(1)

    async def _update_agents(self) -> None:
        agents = self._main_screen.query_one(AgentStrip)
        known = set(self._tab_agents.values())
        new_tabs = []
        for agent_id in self.store.views:
            if agent_id not in known:
                tab_id = f"agent-{len(self._tab_agents)}"
                self._tab_agents[tab_id] = agent_id
                new_tabs.append(AgentTab(agent_id, id=tab_id))
        # Register the whole batch before yielding; switching pages may also refresh tabs.
        if new_tabs:
            await agents.mount(*new_tabs)
        for tab_id, agent_id in self._tab_agents.items():
            view = self.store.views[agent_id]
            tab = agents.query_one(f"#{tab_id}", AgentTab)
            selected = agent_id == self.selected_agent_id
            symbol = {"running": "•", "closing": "•", "ended": "✓", "failed": "×",
                      "cancelled": "■"}.get(view.state, "○")
            style = {"ended": "bold green", "failed": "bold red"}.get(view.state, "not bold #a0a0a0")
            label = Text.assemble((symbol, style), " " + view.label, no_wrap=True)
            tab.set_class(selected, "selected")
            if tab.renderable != label:
                tab.update(label)
        self.call_after_refresh(agents.refresh_edges)

    def _block_widget(self, index: int, block):
        if isinstance(block, tuple):
            return ToolRecord(block, index)
        return SelectableText(block, id=f"record-{index}", classes="record")

    async def _draw(self, *, switching: bool = False) -> None:
        view = self.store.get(self.selected_agent_id)
        if switching:
            self._switching_page = True
        if not switching:
            self._remember_scroll()
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

    def _refresh_jump_latest(self) -> None:
        pane = self._main_screen.query_one("#transcript", VerticalScroll)
        button = self._main_screen.query_one("#jump-latest")
        # Discount the row the button itself takes away from the transcript.
        unseen = pane.max_scroll_y - pane.scroll_y - (button.size.height if button.display else 0)
        button.display = (unseen > 0 and not self.store.get(self.selected_agent_id).follow_output
                          and self._approval is None)

    def _remember_scroll(self) -> None:
        if self._switching_page:
            return
        view = self.store.get(self.selected_agent_id)
        pane = self._main_screen.query_one("#transcript", VerticalScroll)
        # Resizing the viewport changes max_scroll_y without a user scroll.
        # Only a changed position may suspend following new output.
        if pane.scroll_y != view.scroll_position:
            view.follow_output = pane.is_vertical_scroll_end and self._selection_widget is None
            view.scroll_position = pane.scroll_y
        elif pane.is_vertical_scroll_end and self._selection_widget is None:
            view.follow_output = True
        self._refresh_jump_latest()

    @on(events.Click, "#jump-latest")
    def return_to_latest(self) -> None:
        self.action_history_edge(True)

    def action_history_edge(self, latest: bool) -> None:
        self.clear_text_selection()
        if isinstance(self.screen, DetailsScreen):
            pane = self.screen.query_one("#details-scroll", VerticalScroll)
        else:
            pane = self._main_screen.query_one("#transcript", VerticalScroll)
            self.store.get(self.selected_agent_id).follow_output = latest
        pane.scroll_to(y=pane.max_scroll_y if latest else 0, animate=False, immediate=True)
        if self.screen is self._main_screen:
            self.store.get(self.selected_agent_id).scroll_position = pane.scroll_y
            self._refresh_bottom()

    def action_scroll_history(self, direction: int) -> None:
        pane = self.screen.query_one(
            "#details-scroll" if isinstance(self.screen, DetailsScreen) else "#transcript", VerticalScroll)
        pane.scroll_relative(y=direction * max(1, pane.size.height - 1), animate=False, immediate=True)
        if self.screen is self._main_screen:
            self._remember_scroll()
            self._refresh_bottom()

    def action_jump_message(self, direction: int) -> None:
        if self.screen is not self._main_screen:
            return
        pane = self._main_screen.query_one("#transcript", VerticalScroll)
        positions = [self._main_screen.query_one(f"#record-{index}").virtual_region.y
                     for index, block in enumerate(self._mounted_blocks)
                     if isinstance(block, MessageBlock) and block.user]
        targets = [y for y in positions if (y - pane.scroll_y) * direction > 0]
        if targets:
            pane.scroll_to(y=min(targets) if direction > 0 else max(targets), animate=False, immediate=True)
            self._remember_scroll()
            self._refresh_bottom()

    @on(AgentTab.Selected)
    async def on_agent_selected(self, event: AgentTab.Selected) -> None:
        agent_id = event.agent_id
        if agent_id not in self.store.views or agent_id == self.selected_agent_id:
            return
        self.clear_text_selection()
        self._main_screen.query_one("#hint", FooterHint).clear_flash()
        self._remember_scroll()
        with self.batch_update():
            self._switching_page = True
            self.selected_agent_id = agent_id
            await self._update_agents()
            await self._draw(switching=True)
            self._refresh_bottom()
        self.call_after_refresh(self._main_screen.query_one(AgentStrip).reveal_selected)

    def action_toggle_details(self) -> None:
        self.clear_text_selection()
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
        ids = list(self._tab_agents.values())
        agent_id = ids[(ids.index(self.selected_agent_id) + direction) % len(ids)]
        tab_id = next(key for key, value in self._tab_agents.items() if value == agent_id)
        tab = self._main_screen.query_one(f"#{tab_id}", AgentTab)
        if isinstance(self.focused, AgentTab):
            tab.focus()
        self.post_message(AgentTab.Selected(agent_id))

    def action_previous_agent(self) -> None:
        self._select_relative(-1)

    def action_next_agent(self) -> None:
        self._select_relative(1)

    @on(Composer.Submitted)
    def on_message_submitted(self, event: Composer.Submitted) -> None:
        task = event.text.strip()
        if not task or self.closing or self.agent is None or self.selected_agent_id != "main":
            return
        if self.stopping and task.lower() != "/exit":
            self.notify("正在停止；输入已保留，结束后按 Enter 发送。")
            return
        if task.lower() == "/clear" and self.busy:
            self.notify("请等待当前任务结束后再清空历史。", severity="warning")
            return
        event.editor.text = ""
        self._shortcuts_open = False
        self._history_index = None
        if not self._input_history or self._input_history[-1] != task:
            self._input_history.append(task)
        self.action_history_edge(True)
        if task.lower() == "/exit":
            self.action_request_exit()
        elif task.lower() == "/help":
            self.action_show_shortcuts()
        elif self.busy:
            self.user_input.put(task)
        else:
            self._start(task)
        self._refresh_bottom()

    def _start(self, task: str) -> None:
        if self.agent is None or self.busy or self.closing:
            return
        self.busy = True
        self.stopping = False
        operation = "clear" if task.lower() == "/clear" else "run"
        scope = Run("main") if operation == "run" else None
        self.current_run = scope
        self.agent.user_input = self.user_input
        field = self._main_screen.query_one("#task", Composer)
        field.disabled = operation == "clear"

        def execute() -> None:
            error = ""
            completed_operation = operation
            try:
                if operation == "clear":
                    self.agent.clear_history()
                else:
                    scope.execute(lambda: self.agent.run(task))
            except ExecutionStopped:
                completed_operation = "cancelled"
            except Exception as failure:
                error = f"{type(failure).__name__}: {failure}"
            finally:
                self._results.put((scope.id if scope is not None else None, completed_operation, error))

        self._runner = Thread(target=execute, name="praxis-main", daemon=True)
        self._runner.start()

    def action_stop_run(self) -> None:
        if self.busy and self.current_run is not None:
            if self.current_run.finished.done() or self.stopping:
                return
            self.stopping = True
            while True:
                try:
                    self.paused_inputs.append(self.user_input.get_nowait())
                except Empty:
                    break
            self.current_run.cancel()
            if isinstance(self.screen, DetailsScreen):
                self.screen.dismiss()
            self._show_approval()
            self._refresh_bottom()
        elif isinstance(self.screen, DetailsScreen):
            self.screen.dismiss()
        elif self._approval is not None:
            self._main_screen.query_one(ApprovalPanel).action_deny()
        elif self._shortcuts_open:
            self._shortcuts_open = False
            self._refresh_bottom()

    def action_resume_pending(self) -> None:
        if not self.busy and not self.closing and self.paused_inputs:
            task = "\n\n".join(self.paused_inputs)
            self.paused_inputs.clear()
            self._start(task)

    def action_interrupt(self) -> None:
        if self._copy_selection():
            return
        if self.busy:
            self.action_stop_run()
        else:
            self.action_request_exit()

    def set_text_selection(self, widget: SelectableText) -> None:
        self.clear_text_selection()
        self._selection_widget = widget
        if self.screen is self._main_screen and self._approval is None:
            self.store.get(self.selected_agent_id).follow_output = False

    def clear_text_selection(self) -> None:
        widget, self._selection_widget = self._selection_widget, None
        if widget is not None:
            widget.clear_selection()

    def on_mouse_down(self) -> None:
        self.clear_text_selection()

    def copy_to_clipboard(self, text: str) -> None:
        selector = "#details-hint" if isinstance(self.screen, DetailsScreen) else (
            "#approval-hint" if self._approval is not None else "#hint")
        hint = self.screen.query_one(selector, FooterHint)
        try:
            if not self.is_headless:
                pyperclip.copy(text)
        except pyperclip.PyperclipException:
            hint.flash("复制失败，请重试", error=True)
            return
        self._clipboard = text
        hint.flash("已复制到剪贴板")

    def _copy_selection(self) -> bool:
        text = self._selection_widget.selected_text if self._selection_widget is not None else ""
        if not text and isinstance(self.focused, Composer):
            text = self.focused.selected_text
        if not text:
            return False
        self.copy_to_clipboard(text)
        return True

    def action_copy_selection(self) -> None:
        self._copy_selection()

    def _show_approval(self) -> None:
        if self._approval is not None:
            if not self._approval.future.done():
                return
            if isinstance(self.screen, DetailsScreen) and self.screen.request is self._approval:
                self.screen.dismiss()
            self._hide_approval()
        if self.closing or self.stopping:
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

        self._main_screen.query_one(ApprovalPanel).show_request(request, view.label)
        self._refresh_bottom()

    @on(ApprovalPanel.Decided)
    def on_approval_decided(self, event: ApprovalPanel.Decided) -> None:
        if event.request is self._approval:
            if not event.request.future.done():
                try:
                    event.request.future.set_result(event.allowed)
                except InvalidStateError:
                    pass  # A cancellation callback won the approval race.
            self._hide_approval()

    def _hide_approval(self) -> None:
        self._approval = None
        panel = self._main_screen.query_one(ApprovalPanel)
        panel.request = None
        panel.display = False
        self._refresh_bottom()
        if not self.closing and self.selected_agent_id == "main":
            self._main_screen.query_one("#task", Composer).focus()

    def action_request_exit(self) -> None:
        if self.closing:
            return
        self.closing = True
        self._cancel_approvals()
        if self.current_run is not None:
            self.current_run.cancel()
        if self._approval is not None and not self._approval.future.done():
            self._approval.future.cancel()

        def close() -> None:
            error = ""
            try:
                if self._runner is not None:
                    self._runner.join()
                if self.agent is not None:
                    self.agent.close()
            except Exception as failure:
                error = str(failure)
            self._results.put((None, "closed", error))

        Thread(target=close, name="praxis-close", daemon=True).start()

    def on_unmount(self) -> None:
        if not self.is_headless and self._driver is not None:
            self._driver.write("\x1b[0 q")
        self.closing = True
        self._cancel_approvals()
        if self.current_run is not None:
            self.current_run.cancel()
        if self._approval is not None:
            self._approval.future.cancel()
