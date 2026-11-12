import asyncio

import pytest
from textual import events

from praxis_agent.terminal.composer import Composer
from praxis_agent.terminal.cells import MessageBlock
from praxis_agent.terminal.selection import SelectableText


@pytest.mark.parametrize("target", ["#shortcut-content", "#welcome"])
def test_returning_to_terminal_and_clicking_output_keeps_input(terminal_app, target):
    async def exercise():
        app = terminal_app
        async with app.run_test(size=(90, 30)) as pilot:
            editor = app.query_one(Composer)
            editor.load_draft("/")
            await pilot.pause()
            app.post_message(events.AppBlur())
            await pilot.pause()
            app.post_message(events.AppFocus())
            await pilot.pause()
            await pilot.click(target)
            await pilot.press("o", "b", "s")
            assert editor.text == ("/clearobs" if target == "#shortcut-content" else "/obs")
            editor.load_draft("/obs")
            await pilot.press("down", "tab")
            assert editor.text == "/observability"
    asyncio.run(exercise())


def test_leaving_terminal_during_selection_releases_mouse_and_restores_typing(terminal_app):
    async def exercise():
        app = terminal_app
        app.store.get("main").committed_blocks.append(MessageBlock("select some output"))
        async with app.run_test(size=(90, 30)) as pilot:
            editor = app.query_one(Composer)
            await pilot.mouse_down("#record-0", offset=(2, 0))
            widget = app.query_one("#record-0", SelectableText)
            assert app.mouse_captured is widget
            app.post_message(events.AppBlur())
            await pilot.pause()
            assert app.mouse_captured is None
            assert widget._scroll_timer is None
            app.post_message(events.AppFocus())
            await pilot.pause()
            await pilot.press("x")
            assert editor.text == "x"
    asyncio.run(exercise())
