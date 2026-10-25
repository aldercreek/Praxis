"""Reusable Textual widgets and layout helpers for session views."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from typing import ClassVar

from rich.console import Group
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import HorizontalScroll, Vertical, VerticalScroll
from textual.geometry import Region, Size
from textual.layouts.vertical import VerticalLayout
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static

from praxis_agent.terminal.cells import (
    ToolHeading,
    approval_content,
    tool_preview,
)
from praxis_agent.terminal.selection import SelectableText
from praxis_agent.terminal.views import ToolBlock


@dataclass
class Approval:
    tool: str
    arguments: dict
    future: Future
    approval_id: str = ""


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
        self.update(
            Text(text, style="red" if error else "", no_wrap=True, overflow="ellipsis")
        )
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
        Binding("y,1", "allow", "Allow once", show=False),
        Binding("n,2,escape", "deny", "Deny", show=False),
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
        yield OptionList("1. Allow once (y)", "2. Deny (n)", id="approval-choices")
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
        question = (
            "Run this command?"
            if self.request.tool == "run_command"
            else "Allow this action?"
        )
        heading = Text(f"{self.label} · {question}", no_wrap=True, overflow="ellipsis")
        self.query_one("#approval-heading", Static).update(heading)
        content = approval_content(self.request.tool, self.request.arguments)
        rows = self.app.console.render_lines(
            content, self.app.console.options.update(width=width), pad=False
        )
        # Reserve one history row plus question, two choices and confirmation hints.
        budget = max(0, min(10, self.app.size.height - 5))
        hidden = len(rows) > budget
        shown = max(0, budget - int(hidden))
        self.query_one("#approval-content", Static).update(
            Group(
                *(
                    Text.assemble(*((s.text, s.style) for s in row))
                    for row in rows[:shown]
                )
            )
        )
        self.query_one("#approval-content").display = shown > 0
        overflow = self.query_one("#approval-overflow", Static)
        overflow.display = hidden and budget > 0
        overflow.update(
            Text(
                f"… {len(rows) - shown} lines omitted · Ctrl+T for details",
                no_wrap=True,
                overflow="ellipsis",
            )
        )
        self.query_one("#approval-hint", FooterHint).update_hint(
            shortcut_line(
                [
                    ("Enter", "Confirm"),
                    ("Esc", "Stop"),
                    ("↑/↓", "Select"),
                    ("Ctrl+T", "Details"),
                ],
                width,
            )
        )

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
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape,q", "close", "Back", show=False)
    ]

    def __init__(
        self,
        title: str,
        content,
        *,
        agent_id: str | None = None,
        request: Approval | None = None,
    ) -> None:
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
            self.query_one("#details-hint", FooterHint).update_hint(
                shortcut_line(
                    [
                        ("Ctrl+T", "Back"),
                        ("Esc", "Stop" if self.app.busy else "Back"),
                        ("↑/↓", "Scroll"),
                        ("Ctrl+C", "Copy selected text"),
                    ],
                    max(1, self.app.size.width - 2),
                )
            )

    def update_content(self, content) -> None:
        pane = self.query_one("#details-scroll", VerticalScroll)
        position = pane.scroll_y
        self.query_one("#details-content", Static).update(content)
        self.call_after_refresh(
            pane.scroll_to, y=position, animate=False, immediate=True
        )

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

    def _title(self) -> ToolHeading:
        # Wrap during rendering, when Rich has the actual layout width, including the first frame.
        return ToolHeading(self.calls)


class AgentStrip(HorizontalScroll):
    """Keep agent navigation on one line, including ordinary mouse-wheel input."""

    def move(self, direction: int) -> None:
        self.scroll_relative(
            x=direction * max(4, self.size.width // 3), animate=False, immediate=True
        )

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
        self.parent.query_one("#agents-prev").styles.visibility = (
            "visible" if self.scroll_x > 0 else "hidden"
        )
        self.parent.query_one("#agents-next").styles.visibility = (
            "visible" if self.scroll_x < self.max_scroll_x else "hidden"
        )


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
                region.size, parent.app.size, max(0, region.width - gutter.width)
            )
            unused = max(0, region.height - content_height - gutter.height)
            placements[index] = placement._replace(
                region=Region(region.x, region.y, region.width, region.height - unused)
            )
            for following in range(index + 1, len(placements)):
                item = placements[following]
                placements[following] = item._replace(
                    region=item.region.translate((0, -unused))
                )
            break
        return placements
