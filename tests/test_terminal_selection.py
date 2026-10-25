from support.terminal import FakeConnection
"""Exercise real mouse/key events without touching the user's system clipboard."""

import asyncio
import json
from concurrent.futures import Future
from unittest.mock import Mock

import pyperclip
import pytest
from rich.cells import cell_len
from rich.text import Text
from textual.containers import VerticalScroll

from praxis_agent.events import AssistantTextDelta, ModelRequestStarted
from praxis_agent.execution import Run
from praxis_agent.terminal.app import (
    Approval,
    ApprovalPanel,
    DetailsScreen,
    FooterHint,
    PraxisApp,
)
from praxis_agent.terminal.cells import MessageBlock
from praxis_agent.terminal.composer import Composer
from praxis_agent.terminal.selection import SelectableText
from praxis_agent.terminal.views import ToolBlock


@pytest.mark.parametrize("source,user,start,end,expected", [
    ("中文复制验证", False, (2, 0), (14, 0), "中文复制验证"),
    ("中文复制验证", True, (14, 0), (2, 0), "中文复制验证"),
    ("中A文B", False, (3, 0), (7, 0), "中A文"),
    ("hello world", False, (2, 0), (7, 0), "hello"),
    ("hello world", True, (7, 0), (2, 0), "hello"),
    ("第一段中文 hello\n\n第二段中文 world", False, (2, 0), (8, 2), "第一段中文 hello\n\n  第二段"),
    ("第一段中文 hello\n\n第二段中文 world", False, (8, 2), (2, 0), "第一段中文 hello\n\n  第二段"),
    ("末尾不带填充空格", False, (2, 0), (60, 0), "末尾不带填充空格"),
])
def test_drag_copies_exact_unicode_text(terminal_app, drag_selection, source, user, start, end, expected):
    async def exercise():
        app = terminal_app
        app.store.get("main").committed_blocks.append(MessageBlock(source, user=user))
        async with app.run_test(size=(80, 26)) as pilot:
            widget = app.query_one("#record-0", SelectableText)
            await drag_selection(pilot, widget, start, end)
            assert widget.selected_text == expected
            assert app.mouse_captured is None
            highlighted = [s for s in widget.render_line(0) if s.style and s.style.bgcolor
                           and s.style.bgcolor.name == "#334d55"]
            assert highlighted
            await pilot.press("ctrl+c")
            assert app.clipboard == expected
            assert app.is_running
            assert app.query_one("#hint").renderable.plain == "Copied to clipboard"
    asyncio.run(exercise())


@pytest.mark.parametrize("action", ["click_editor", "type", "resize", "switch", "details", "hide", "remove"])
def test_selection_is_cleared_when_its_display_context_changes(terminal_app, drag_selection, action):
    async def exercise():
        app = terminal_app
        app.store.get("main").committed_blocks.append(MessageBlock("需要清理的选区"))
        app.store.get("child")
        async with app.run_test(size=(80, 28)) as pilot:
            widget = app.query_one("#record-0", SelectableText)
            await drag_selection(pilot, widget, (2, 0), (10, 0))
            assert widget.selected_text
            if action == "click_editor":
                await pilot.click("#task")
            elif action == "type":
                app.query_one(Composer).focus()
                await pilot.press("x")
            elif action == "resize":
                await pilot.resize_terminal(60, 28)
            elif action == "switch":
                await pilot.press("alt+right")
            elif action == "details":
                await pilot.press("ctrl+t")
            elif action == "hide":
                widget.display = False
            else:
                await widget.remove()
            await pilot.pause()
            assert not widget.selected_text
            assert app._selection_widget is None
            assert app.mouse_captured is None
            assert widget._scroll_timer is None
    asyncio.run(exercise())


def test_click_is_not_selection_and_only_one_message_can_be_selected(terminal_app, drag_selection):
    async def exercise():
        app = terminal_app
        app.store.get("main").committed_blocks.extend([MessageBlock("first message"), MessageBlock("second message")])
        async with app.run_test(size=(80, 26)) as pilot:
            first, second = [app.query_one(f"#record-{i}", SelectableText) for i in range(2)]
            await pilot.click(first, offset=(4, 0))
            assert first.selected_text == ""
            await pilot.press("ctrl+shift+c")
            assert app.clipboard == "" and app.is_running
            await drag_selection(pilot, first, (2, 0), (7, 0))
            await drag_selection(pilot, second, (2, 0), (8, 0))
            assert not first.selected_text and second.selected_text == "second"
            await pilot.press("ctrl+shift+c")
            assert app.clipboard == "second"
    asyncio.run(exercise())


@pytest.mark.parametrize("action", ["hide", "remove", "resize"])
def test_interrupted_drag_releases_mouse_and_stops_timer(terminal_app, action):
    async def exercise():
        app = terminal_app
        app.store.get("main").committed_blocks.append(MessageBlock("拖选中途改变界面"))
        async with app.run_test(size=(80, 26)) as pilot:
            widget = app.query_one("#record-0", SelectableText)
            await pilot.mouse_down(widget, offset=(2, 0))
            await pilot.hover(widget, offset=(8, 0))
            assert app.mouse_captured is widget and widget._scroll_timer is not None
            if action == "hide":
                widget.display = False
            elif action == "remove":
                await widget.remove()
            else:
                await pilot.resize_terminal(60, 26)
            await pilot.pause()
            assert not widget.selected_text and app._selection_widget is None
            assert app.mouse_captured is None and widget._scroll_timer is None
            await pilot.mouse_up(offset=(0, 0))
    asyncio.run(exercise())


def test_idle_control_c_without_selection_still_exits(terminal_app, wait_for_ui):
    async def exercise():
        app = terminal_app
        async with app.run_test(size=(80, 26)) as pilot:
            await pilot.press("ctrl+shift+c")
            assert app.is_running and not app.closing
            await pilot.press("ctrl+c")
            assert app.closing
            await wait_for_ui(pilot, lambda: app.return_value is not None)
        assert app.return_value == 0
    asyncio.run(exercise())


async def select_literal(pilot, widget, literal, drag):
    """Locate rendered cells, but compare copied text against a fixed source literal."""
    for row in range(widget.size.height):
        line = widget.render_line(row).text
        if literal in line:
            column = cell_len(line[:line.index(literal)])
            await drag(pilot, widget, (column, row), (column + cell_len(literal), row))
            return
    pytest.fail(f"{literal!r} not rendered")


@pytest.mark.parametrize("surface", ["message", "code", "tool", "details", "approval"])
def test_copy_feedback_is_inline_without_focus_or_layout_change(
    terminal_app, drag_selection, wait_for_ui, monkeypatch, surface,
):
    async def exercise():
        app = terminal_app
        notify = Mock()
        monkeypatch.setattr(app, "notify", notify)
        blocks = app.store.get("main").committed_blocks
        if surface == "code":
            blocks.append(MessageBlock("```python\nprint('hello')\n```"))
            literal = "print('hello')"
        elif surface == "tool":
            result = {"status": "exited", "exit_code": 0, "stdout": "copy output", "stderr": "", "error": None}
            blocks.append(ToolBlock("Running command", "{}", json.dumps(result), structured_output=result,
                                    name="run_command", detail="echo copy output"))
            literal = "copy output"
        else:
            blocks.append(MessageBlock("复制这段中文"))
            literal = "复制这段中文"
        async with app.run_test(size=(100, 30)) as pilot:
            hint_selector = "#hint"
            if surface == "details":
                await pilot.press("ctrl+t")
                assert isinstance(app.screen, DetailsScreen)
                widget = app.screen.query_one("#details-content", SelectableText)
                hint_selector = "#details-hint"
            elif surface == "approval":
                literal = "echo hello"
                request = Approval("run_command", {"argv": ["cmd", "/c", literal]}, Future())
                app._approvals.put(request)
                await wait_for_ui(pilot, lambda: app.query_one(ApprovalPanel).display)
                widget = app.query_one("#approval-content", SelectableText)
                hint_selector = "#approval-hint"
            elif surface == "tool":
                widget = app.query_one(".tool-preview", SelectableText)
            else:
                widget = app.query_one("#record-0", SelectableText)
            await select_literal(pilot, widget, literal, drag_selection)
            hint = app.screen.query_one(hint_selector, FooterHint)
            focused, region = app.focused, hint.region
            await pilot.press("ctrl+c")
            assert app.clipboard == literal
            assert hint.renderable.plain == "Copied to clipboard"
            assert app.focused is focused and hint.region == region
            assert hint.size.height == 1
            assert not notify.called
            if surface == "approval":
                assert not request.future.done()  # Copy never decides an approval.
            if surface == "message":
                # Normal footer changes during the flash must appear after it expires.
                app.query_one(Composer).load_draft("new draft")
                await pilot.pause()
                assert hint.renderable.plain == "Copied to clipboard"
                await wait_for_ui(pilot, lambda: "Enter" in hint.renderable.plain)
                assert "Send" in hint.renderable.plain
    asyncio.run(exercise())


@pytest.mark.parametrize("failure", [False, True])
def test_native_clipboard_result_never_cancels_execution(
    terminal_app, drag_selection, monkeypatch, failure,
):
    async def exercise():
        app = terminal_app
        app.connection = FakeConnection(active=True)
        app.busy, app.current_turn_id = True, "T1"
        app.store.get("main").committed_blocks.append(MessageBlock("copy safely"))
        native_copy = Mock(side_effect=pyperclip.PyperclipException("unavailable") if failure else None)
        monkeypatch.setattr(pyperclip, "copy", native_copy)
        notify = Mock()
        monkeypatch.setattr(app, "notify", notify)
        async with app.run_test(size=(80, 26)) as pilot:
            await drag_selection(pilot, app.query_one("#record-0"), (2, 0), (13, 0))
            # Only the public clipboard call needs native mode; retain the headless driver.
            original = app.copy_to_clipboard

            def copy(text):
                with monkeypatch.context() as patch:
                    patch.setattr(PraxisApp, "is_headless", property(lambda self: False))
                    original(text)

            monkeypatch.setattr(app, "copy_to_clipboard", copy)
            await pilot.press("ctrl+c")
            native_copy.assert_called_once_with("copy safely")
            assert not app.connection.cancelled
            assert app.busy and not app.stopping and app.is_running
            assert not notify.called
            hint = app.query_one("#hint", FooterHint)
            if failure:
                assert hint.renderable.plain == "Copy failed. Try again."
                assert hint.renderable.style == "red"
                assert app.clipboard == ""
            else:
                assert hint.renderable.plain == "Copied to clipboard"
                assert app.clipboard == "copy safely"
    asyncio.run(exercise())


def test_editor_selection_copies_before_cancel_and_no_selection_cancels(terminal_app, wait_for_ui):
    async def exercise():
        app = terminal_app
        app.connection = FakeConnection(active=True)
        app.busy, app.current_turn_id = True, "T1"
        async with app.run_test(size=(80, 26)) as pilot:
            editor = app.query_one(Composer)
            editor.load_draft("草稿也能复制")
            editor.select_all()
            await pilot.press("ctrl+c")
            assert app.clipboard == "草稿也能复制"
            assert not app.connection.cancelled
            editor.move_cursor(editor.document.end)
            await pilot.press("ctrl+shift+c")
            assert not app.connection.cancelled
            await pilot.press("ctrl+c")
            await wait_for_ui(pilot, lambda: app.connection.cancelled and app.stopping)
            assert app.is_running and editor.text == "草稿也能复制"
    asyncio.run(exercise())


def test_drag_at_viewport_edges_scrolls_and_live_output_does_not_steal_position(
    terminal_app, wait_for_ui,
):
    async def exercise():
        app = terminal_app
        view = app.store.get("main")
        view.consume(ModelRequestStarted("root", 1))
        view.consume(AssistantTextDelta("root", 1, "\n\n".join(f"line {i}" for i in range(100))))
        async with app.run_test(size=(80, 24)) as pilot:
            pane = app.query_one("#transcript", VerticalScroll)
            pane.scroll_to(y=15, animate=False, immediate=True)
            await pilot.pause()
            widget = app.query_one("#record-0", SelectableText)
            x, y = pane.content_region.x + 3, pane.content_region.y
            await pilot.mouse_down(offset=(x, y + 3))
            await pilot.hover(offset=(x + 5, pane.content_region.bottom - 1))
            await wait_for_ui(pilot, lambda: pane.scroll_y > 20)
            await pilot.hover(offset=(x, y))
            # Reversing the drag must continue past its original anchor, where the
            # selection briefly collapses before extending in the other direction.
            await wait_for_ui(pilot, lambda: pane.scroll_y < 15)
            await pilot.mouse_up(offset=(x, y))
            await pilot.pause()
            position = pane.scroll_y
            assert widget.selected_text
            assert widget._scroll_timer is None and app.mouse_captured is None
            app.sink.emit(AssistantTextDelta("root", 1, "\nnew streamed text"))
            await wait_for_ui(pilot, lambda: "new streamed text" in widget.renderable.text)
            assert pane.scroll_y == position and not view.follow_output
            assert app.query_one("#jump-latest").display
            await pilot.click("#jump-latest")
            await wait_for_ui(pilot, lambda: pane.is_vertical_scroll_end)
            assert view.follow_output and not app.query_one("#jump-latest").display
    asyncio.run(exercise())


def test_copy_flash_resets_on_agent_switch(terminal_app, drag_selection, wait_for_ui):
    async def exercise():
        app = terminal_app
        app.store.get("main").committed_blocks.append(Text("copy"))
        app.store.get("child")
        async with app.run_test(size=(80, 26)) as pilot:
            await drag_selection(pilot, app.query_one("#record-0"), (0, 0), (4, 0))
            await pilot.press("ctrl+c", "alt+right")
            await wait_for_ui(pilot, lambda: app.selected_agent_id == "child")
            assert "Copied" not in app.query_one("#hint").renderable.plain
            assert "Switch agent" in app.query_one("#hint").renderable.plain
    asyncio.run(exercise())
