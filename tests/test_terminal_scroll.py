"""Following output and showing an unseen-content affordance are separate states."""

import asyncio

import pytest
from rich.text import Text
from textual.containers import VerticalScroll

from praxis_agent.events import AssistantTextDelta, ChildAgentStatus
from praxis_agent.terminal.selection import SelectableText


@pytest.mark.parametrize("interaction", ["click", "select", "clear"])
def test_short_content_never_shows_jump_latest_after_mouse_interaction(
    terminal_app, drag_selection, interaction,
):
    async def exercise():
        app = terminal_app
        view = app.store.get("main")
        view.committed_blocks.append(Text("这是一段可以选择的短文字"))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            pane = app.query_one("#transcript", VerticalScroll)
            widget = app.query_one("#record-0", SelectableText)
            assert pane.max_scroll_y == 0
            if interaction == "click":
                await pilot.click(widget, offset=(4, 0))
            else:
                await drag_selection(pilot, widget, (0, 0), (12, 0))
                assert widget.selected_text
                if interaction == "clear":
                    await pilot.click("#task")
            for _ in range(3):
                await app._tick()
                await pilot.pause()
                assert pane.max_scroll_y == 0
                assert not app.query_one("#jump-latest").display
            assert view.follow_output == (interaction != "select")
    asyncio.run(exercise())


def test_jump_button_does_not_create_its_own_overflow(terminal_app, drag_selection, wait_for_ui):
    async def exercise():
        app = terminal_app
        app.store.get("main").committed_blocks.append(Text("\n".join(f"line {i}" for i in range(12))))
        async with app.run_test(size=(60, 24)) as pilot:
            await pilot.pause()
            pane = app.query_one("#transcript", VerticalScroll)
            widget = app.query_one("#record-0", SelectableText)
            await drag_selection(pilot, widget, (0, 0), (4, 0))
            # Exactly one unseen row before the button occupies an extra row.
            await pilot.resize_terminal(60, 18)
            await wait_for_ui(pilot, lambda: app.query_one("#jump-latest").display)
            assert pane.max_scroll_y == 2
            pane.scroll_to(y=1, animate=False, immediate=True)
            await wait_for_ui(pilot, lambda: not app.query_one("#jump-latest").display)
            assert pane.is_vertical_scroll_end
            for _ in range(3):
                await app._tick()
                await pilot.pause()
                assert not app.query_one("#jump-latest").display
                assert pane.max_scroll_y == 1 and pane.scroll_y == 1
    asyncio.run(exercise())


@pytest.mark.parametrize("return_action", ["click", "ctrl+end", "scroll"])
def test_jump_button_tracks_unseen_history_and_disappears_at_bottom(terminal_app, wait_for_ui, return_action):
    async def exercise():
        app = terminal_app
        app.store.get("main").committed_blocks.append(Text("\n".join(f"line {i}" for i in range(60))))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            pane = app.query_one("#transcript", VerticalScroll)
            button = app.query_one("#jump-latest")
            assert pane.is_vertical_scroll_end and not button.display
            await pilot.press("pageup")
            await wait_for_ui(pilot, lambda: button.display)
            assert not app.store.get("main").follow_output
            if return_action == "click":
                await pilot.click(button)
            elif return_action == "ctrl+end":
                await pilot.press("ctrl+end")
            else:
                pane.scroll_end(animate=False, immediate=True)
            await wait_for_ui(pilot, lambda: not button.display and pane.is_vertical_scroll_end)
            assert app.store.get("main").follow_output
    asyncio.run(exercise())


def test_resize_content_shrink_and_agent_switch_clear_jump_button(terminal_app, wait_for_ui):
    async def exercise():
        app = terminal_app
        main = app.store.get("main")
        main.committed_blocks.append(Text("\n".join(f"line {i}" for i in range(60))))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("ctrl+home")
            await wait_for_ui(pilot, lambda: app.query_one("#jump-latest").display)
            app.sink.emit(ChildAgentStatus("main", "child", "running", "a short page"))
            await wait_for_ui(pilot, lambda: bool(app.query("#agent-1")))
            await pilot.press("alt+right")
            await wait_for_ui(pilot, lambda: app.selected_agent_id == "child")
            assert not app.query_one("#jump-latest").display
            await pilot.press("alt+left")
            await wait_for_ui(pilot, lambda: app.query_one("#jump-latest").display)
            await pilot.resize_terminal(80, 90)
            await wait_for_ui(pilot, lambda: not app.query_one("#jump-latest").display)
            assert main.follow_output
            await pilot.resize_terminal(80, 24)
            await pilot.press("ctrl+home")
            await wait_for_ui(pilot, lambda: app.query_one("#jump-latest").display)
            main.committed_blocks.clear()
            await app._draw()
            await wait_for_ui(pilot, lambda: not app.query_one("#jump-latest").display)
            assert app.query_one("#transcript", VerticalScroll).max_scroll_y == 0
    asyncio.run(exercise())


def test_selection_pauses_stream_until_explicit_return_to_latest(terminal_app, drag_selection, wait_for_ui):
    async def exercise():
        app = terminal_app
        view = app.store.get("main")
        view.consume(AssistantTextDelta("turn", 1, "可选中的回答"))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            pane = app.query_one("#transcript", VerticalScroll)
            widget = app.query_one("#record-0", SelectableText)
            await drag_selection(pilot, widget, (2, 0), (8, 0))
            selected = widget.selected_text
            assert selected and not view.follow_output
            assert not app.query_one("#jump-latest").display
            app.sink.emit(AssistantTextDelta("turn", 1, "\n\n" + "\n\n".join(["新的段落"] * 40)))
            await wait_for_ui(pilot, lambda: app.query_one("#jump-latest").display)
            assert pane.scroll_y == 0 and widget.selected_text == selected
            await pilot.press("ctrl+end")
            await wait_for_ui(pilot, lambda: pane.is_vertical_scroll_end and not app.query_one("#jump-latest").display)
            assert view.follow_output and not widget.selected_text
            app.sink.emit(AssistantTextDelta("turn", 1, "\n\n最后一段"))
            await wait_for_ui(pilot, lambda: widget.renderable.text.endswith("最后一段")
                              and pane.is_vertical_scroll_end)
            assert view.follow_output
    asyncio.run(exercise())
