"""One Textual screen with memory-only, independently scrollable agent pages."""

from __future__ import annotations

import asyncio
import json
import sys
from concurrent.futures import Future, InvalidStateError
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Thread
from typing import ClassVar

import pyperclip
from rich.console import RenderableType
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.theme import Theme
from textual.widgets import Static

from praxis_agent.terminal.cells import (
    ACCENT,
    MessageBlock,
    WelcomeHeader,
)
from praxis_agent.terminal.composer import Composer
from praxis_agent.terminal.delete_dialog import DeleteSessionDialog
from praxis_agent.terminal.screen import ConversationScreen
from praxis_agent.terminal.selection import SelectableText
from praxis_agent.terminal.views import ViewEventSink, ViewStore
from praxis_agent.terminal.widgets import (
    AgentStrip,
    AgentTab,
    Approval,
    ApprovalPanel,
    ConversationLayout,
    DetailsScreen,
    FooterHint,
    ToolRecord,
    _duration,
    shortcut_line,
)


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
        Binding("alt+left", "previous_agent", "Previous agent", priority=True),
        Binding("alt+right", "next_agent", "Next agent", priority=True),
        Binding("ctrl+t", "toggle_details", "Full transcript", priority=True),
        Binding("ctrl+q", "request_exit", "Quit", priority=True),
        Binding("ctrl+c", "interrupt", "Stop or quit", priority=True),
        Binding("ctrl+shift+c", "copy_selection", "Copy selected text", priority=True),
        Binding("escape", "stop_run", "Stop turn", priority=True),
        Binding("ctrl+r", "resume_pending", "Send retained follow-ups", priority=True),
        Binding("f1", "show_shortcuts", "Shortcuts", priority=True),
        Binding("shift+left", "edit_pending", "Edit pending message", priority=True),
        Binding("pageup", "scroll_history(-1)", "Page up", priority=True),
        Binding("pagedown", "scroll_history(1)", "Page down", priority=True),
        Binding("ctrl+home", "history_edge(False)", "First message", priority=True),
        Binding("ctrl+end", "history_edge(True)", "Jump to latest", priority=True),
        Binding("alt+up", "jump_message(-1)", "Previous input", priority=True),
        Binding("alt+down", "jump_message(1)", "Next input", priority=True),
    ]

    def __init__(self, sink: ViewEventSink, *, initial_task: str = "", connection=None) -> None:
        super().__init__()
        self.register_theme(Theme(
            name="praxis", primary=ACCENT, foreground="#f2f2f2",
            background="#0c0c0c", surface="#0c0c0c", panel="#0c0c0c",
            success="ansi_green", error="ansi_red", variables={"text-muted": "#a0a0a0"},
        ))
        self.theme = "praxis"
        self.ansi_color = True
        self.sink = sink
        self.connection = connection
        self.store = ViewStore()
        self.selected_agent_id = "main"
        self.initial_task = initial_task
        self.busy = False
        self.closing = False
        self.stopping = False
        self.current_turn_id: str | None = None
        self.paused_inputs: list[str] = []
        self._results: SimpleQueue[tuple[str | None, str, str]] = SimpleQueue()
        self._approvals: SimpleQueue[Approval] = SimpleQueue()
        self._approval: Approval | None = None
        self._remote_approvals = {}
        self._tab_agents: dict[str, str] = {"agent-0": "main"}
        # Timer callbacks and page events run in separate asyncio tasks.
        self._view_lock = asyncio.Lock()
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
        yield Static(Text("↓ Jump to latest · Ctrl+End", no_wrap=True, overflow="ellipsis"), id="jump-latest")
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
        async with self._view_lock:
            await self._draw(switching=True)
        self._main_screen.query_one("#task", Composer).focus()
        if self.initial_task:
            editor = self._main_screen.query_one("#task", Composer)
            editor.text = self.initial_task
            self.on_message_submitted(Composer.Submitted(editor))
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

    async def _tick(self) -> None:
        async with self._view_lock:
            with self.batch_update():
                await self._update_views()

    async def _update_views(self) -> None:
        if not self.is_running:
            return
        self._remember_scroll()
        changed = self.sink.drain(self.store)
        if self.connection is not None:
            changed.update(self.connection.drain(self.store))
            if getattr(self.connection, "deleted", False) and getattr(self, "_deleted_session_id", None) != self.connection.id:
                self._deleted_session_id = self.connection.id
                self.sink.discard_pending()
                self.store = ViewStore()
                self.selected_agent_id = "main"
                self._mounted_blocks = []
                changed.add("main")
                await self._main_screen.query_one("#blocks", Vertical).remove_children()
                self.notify("Chat deleted. Use /clear to start a new chat.")
            active = self.connection.summary.get("active_turn")
            self.busy = active is not None
            self.stopping = bool(active and active.get("cancel_requested"))
            self.current_turn_id = active["id"] if active else None
            self.paused_inputs = [item["content"] for item in self.connection.pending]
            current = self._approval.approval_id if self._approval else None
            if current and current not in self.connection.approvals:
                self._approval.future.cancel()
            queued = {getattr(item, "approval_id", "") for item in list(self._remote_approvals.values())}
            for approval_id, data in self.connection.approvals.items():
                if approval_id not in queued:
                    approval = Approval(data["tool_name"], data["arguments"], Future(), approval_id)
                    self._remote_approvals[approval_id] = approval
                    self._approvals.put(approval)
            for approval_id, approval in list(self._remote_approvals.items()):
                if approval_id not in self.connection.approvals:
                    approval.future.cancel()
                    del self._remote_approvals[approval_id]
        while True:
            try:
                operation, error = self._results.get_nowait()
            except Empty:
                break
            if error:
                self.notify(error, severity="error")
            if operation == "clear" and not error:
                self.sink.discard_pending()
                self.store = ViewStore()
                self.selected_agent_id = "main"
                self._mounted_blocks = []
                changed.add("main")
                await self._main_screen.query_one("#blocks", Vertical).remove_children()
        await self._update_agents()
        if self.selected_agent_id in changed or self._mounted_agent != self.selected_agent_id:
            await self._draw(switching=self._mounted_agent != self.selected_agent_id)
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
        agents.display = len(self.store.visible_agent_ids) > 1 and not approval
        screen.query_one("#composer").display = is_main and not approval
        status = screen.query_one("#status", Static)
        running = view.state in {"running", "closing"} or (
            self.busy and is_main and view.state not in {"ended", "failed", "cancelled"})
        reconnecting = bool(self.connection and getattr(self.connection, "connection_error", ""))
        status.display = not approval and (not self.empty_session or reconnecting)
        if status.display:
            settling = view.state == "closing" or running and (self.stopping or self.closing)
            if settling:
                phase = "Closing" if self.closing or view.state == "closing" else "Stopping · waiting for cleanup"
            else:
                phase = "Working" if view.has_active_tools else view.phase
            text = Text(f"• {phase}")
            if view.started_at is not None:
                text.append(f" · {'Running for' if running else 'Elapsed'} {_duration(view.elapsed())}")
            text.truncate(width, overflow="ellipsis")
            if view._capacity is not None:
                capacity = view._capacity.copy()
                capacity.truncate(width, overflow="ellipsis")
                text.append(" · " if text.cell_len + capacity.cell_len + 3 <= width else "\n")
                text.append_text(capacity)
            if status.renderable != text:
                status.update(text)
            if reconnecting:
                status.update("Reconnecting to Praxis…")
        editor = screen.query_one(Composer)
        editor.read_only = not is_main
        retained = self.paused_inputs
        preview = screen.query_one("#pending", Static)
        preview.display = is_main and bool(retained) and not approval
        if preview.display:
            pending = retained
            lines = [Text(f"› {' '.join(item.split())}", no_wrap=True, overflow="ellipsis")
                     for item in pending[:2]]
            if len(pending) > 2:
                lines.append(Text(f"  … {len(pending) - 2} more"))
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
            welcome_space -= min(3, len(retained))
        welcome = screen.query_one("#welcome", Static)
        welcome.display = self.empty_session and welcome_space >= 4
        if welcome.display:
            content = WelcomeHeader(
                (Path(self.connection.summary["workspace"]) if self.connection else None), (self.connection.summary["model"] if self.connection else None),
                compact=welcome_space < 7,
            )
            if welcome.renderable != content:
                welcome.update(content)
        hint = screen.query_one("#hint", FooterHint)
        hint.display = not approval
        if editor.read_only:
            actions = [("Alt+←/→", "Switch agent")]
        elif self.stopping or self.closing:
            actions = [("Enter", "Save input for later")]
        elif retained and not running:
            actions = [("Ctrl+R", "Send retained input"), ("Shift+←", "Edit pending")]
        elif retained and not editor.text:
            actions = [("Shift+←", "Edit pending"), ("?", "Shortcuts")]
        elif editor.text:
            actions = [("Enter", "Send follow-up" if running else "Send"), ("Shift+Enter", "New line")]
        elif running:
            actions = [("Enter", "Send follow-up")]
        else:
            actions = [("/", "Commands"), ("?", "Shortcuts")]
        if self.busy and not (self.stopping or self.closing):
            actions.insert(1, ("Esc", "Stop turn"))
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
            ("Enter", "Send a message or follow-up while running"),
            ("Shift+Enter", "New line; pasted line breaks are preserved"),
            ("Esc", "Stop the turn; continue after cleanup"),
            ("Ctrl+R", "Send follow-ups retained after stopping"),
            ("Shift+←", "Edit the next pending message when the input is empty"),
            ("↑/↓", "Browse input history from the first/last line"),
            ("PgUp/PgDn · Ctrl+Home/End", "Scroll history · first/latest"),
            ("Alt+↑/↓", "Jump to the previous/next user message"),
            ("Ctrl+T", "Full transcript; full arguments during approval"),
            ("Alt+←/→", "Switch agent"),
            ("Ctrl+C / Ctrl+Shift+C", "Copy selected text"),
            ("Ctrl+C", "Stop or quit when no text is selected"), ("Ctrl+Q", "Quit Praxis"),
            ("F1 / ? in empty input", "Keyboard shortcuts"),
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
            ("/clear", "Clear current chat"), ("/help", "Keyboard shortcuts"), ("/exit", "Quit Praxis"),
            ("/commands", "List command sessions"), ("/stop-command", "Stop a command by its session ID"),
            ("/mcp", "Show MCP service status"),
            ("/delete", "Delete this chat"),
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
        editor = self._main_screen.query_one("#task", Composer)
        if editor.text or not self.connection or not self.connection.pending:
            return
        item = self.connection.pending[0]
        def remove():
            self.connection.delete(item["command_id"])
            self.call_from_thread(editor.load_draft, item["content"])
        self._request("delete_input", remove)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "edit_pending":
            editor = self.focused
            return (isinstance(editor, Composer) and not editor.read_only and not editor.text
                    and bool(self.paused_inputs))
        return super().check_action(action, parameters)

    @on(events.Click, "#agents-prev")
    def scroll_agents_left(self) -> None:
        self._main_screen.query_one(AgentStrip).move(-1)

    @on(events.Click, "#agents-next")
    def scroll_agents_right(self) -> None:
        self._main_screen.query_one(AgentStrip).move(1)

    async def _update_agents(self) -> None:
        agents = self._main_screen.query_one(AgentStrip)
        visible = set(self.store.visible_agent_ids)
        if self.selected_agent_id not in visible:
            self.clear_text_selection()
            self.selected_agent_id = "main"
            if self._approval is None:
                self._main_screen.query_one("#task", Composer).focus()
        for tab_id, agent_id in list(self._tab_agents.items()):
            if agent_id not in visible:
                del self._tab_agents[tab_id]
                await agents.query_one(f"#{tab_id}", AgentTab).remove()
        known = set(self._tab_agents.values())
        new_tabs = []
        for index, agent_id in enumerate(self.store.views):
            if agent_id in visible and agent_id not in known:
                tab_id = f"agent-{index}"
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
        async with self._view_lock:
            agent_id = event.agent_id
            # A queued selection may refer to an agent removed by a session reset.
            if agent_id not in self.store.visible_agent_ids or agent_id == self.selected_agent_id:
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
                f"{view.label} · {request.tool} · Full arguments",
                Text(json.dumps(request.arguments, ensure_ascii=False, indent=2)), request=request,
            ))
        else:
            view = self.store.get(self.selected_agent_id)
            self.push_screen(DetailsScreen(
                f"{view.label} · Full execution transcript", view.transcript_content(), agent_id=view.agent_id,
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
        if not task or self.closing or self.connection is None or self.selected_agent_id != "main":
            return
        if getattr(self, "_deleting_session", False):
            self.notify("Deleting this chat…")
            return
        if getattr(self.connection, "deleted", False) and task.lower() not in {"/clear", "/delete", "/exit", "/help", "/mcp"}:
            self.notify("This chat was deleted. Use /clear to start a new chat.", severity="warning")
            return
        if task.lower() == "/clear" and self.busy:
            self.notify("Wait for the current task to finish before clearing history.", severity="warning")
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
        elif task.lower() == "/mcp":
            self.run_worker(self._show_mcp(), group="mcp-status", exclusive=True)
        elif task.lower() == "/delete":
            self.push_screen(DeleteSessionDialog(self.connection.summary), self._confirm_delete)
        elif task.lower() == "/commands":
            commands = self.connection.commands()
            text = "\n".join(f"{item['session_id']}  {item['status']}  {' '.join(item['argv'])}"
                             for item in commands) or "No command sessions."
            self.push_screen(DetailsScreen("Command sessions · /stop-command <session_id>", Text(text)))
        elif task.lower().split()[0] == "/stop-command":
            parts = task.split()
            if len(parts) != 2:
                self.notify("Usage: /stop-command <session_id>", severity="warning")
            else:
                self.run_worker(self._stop_command(parts[1]))
        else:
            self._start(task)
        self._refresh_bottom()

    async def _show_mcp(self):
        states = {
            "disabled": "已禁用", "queued": "等待连接", "connecting": "连接中",
            "retrying": "等待重试", "connected": "已连接", "failed": "连接失败",
            "disconnected": "已断开",
        }
        try:
            servers = await asyncio.to_thread(self.connection.mcp_servers)
            text = "\n".join(f"{item['name']}    {states.get(item['state'], item['state'])}"
                             for item in servers) or "未配置 MCP 服务"
            self.push_screen(DetailsScreen("MCP 服务", Text(text)))
        except Exception as error:
            self.notify(str(error), severity="error")

    def _confirm_delete(self, confirmed):
        if confirmed:
            self.run_worker(self._delete_session(), group="delete-session", exclusive=True)

    async def _delete_session(self):
        self._deleting_session = True
        try:
            await asyncio.to_thread(self.connection.delete_session)
        except Exception as error:
            self.notify(f"Could not finish deletion: {error}. Use /delete to retry.", severity="error")
        finally:
            self._deleting_session = False

    async def _stop_command(self, session_id):
        try:
            await asyncio.to_thread(self.connection.stop_command, session_id)
        except Exception as error:
            self.notify(str(error), severity="error")

    def _request(self, operation, action):
        def execute():
            try:
                action()
                self._results.put((operation, ""))
            except Exception as error:
                self._results.put((operation, str(error)))
        Thread(target=execute, name="praxis-request", daemon=True).start()

    def _start(self, task: str) -> None:
        if self.connection is None or self.closing:
            return
        if task.lower() == "/clear":
            self._request("clear", self.connection.clear)
        else:
            self._request("input", lambda: self.connection.send(task))

    def action_stop_run(self) -> None:
        if isinstance(self.screen, DeleteSessionDialog):
            self.screen.dismiss(False)
            return
        if self.busy and self.current_turn_id and not self.stopping:
            turn_id = self.current_turn_id
            self._request("cancel", lambda: self.connection.cancel(turn_id))
        elif isinstance(self.screen, DetailsScreen):
            self.screen.dismiss()
        elif self._approval is not None:
            self._main_screen.query_one(ApprovalPanel).action_deny()
        elif self._shortcuts_open:
            self._shortcuts_open = False
            self._refresh_bottom()

    def action_resume_pending(self) -> None:
        if self.connection and not self.busy and self.paused_inputs:
            self._request("resume", self.connection.resume)

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
            hint.flash("Copy failed. Try again.", error=True)
            return
        self._clipboard = text
        hint.flash("Copied to clipboard")

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
        if self.connection and event.request is self._approval:
            def decide():
                result = self.connection.decide(event.request.approval_id, event.allowed)
                try:
                    if result["status"] == "cancelled":
                        event.request.future.cancel()
                    else:
                        event.request.future.set_result(result["status"] == "approved")
                except InvalidStateError:
                    pass  # The SSE result may already have closed this display.
            self._request("approval", decide)

    def _hide_approval(self) -> None:
        self._approval = None
        panel = self._main_screen.query_one(ApprovalPanel)
        panel.request = None
        panel.display = False
        self._refresh_bottom()
        if not self.closing and self.selected_agent_id == "main":
            self._main_screen.query_one("#task", Composer).focus()

    def action_request_exit(self) -> None:
        self.closing = True
        self.exit(0)

    def on_unmount(self) -> None:
        if not self.is_headless and self._driver is not None:
            self._driver.write("\x1b[0 q")
        self.closing = True
        if self.connection:
            self.connection.close()
