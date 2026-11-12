import asyncio

from praxis_agent.terminal.composer import Composer
from support.terminal import FakeConnection


def test_command_navigation_completion_and_input_history(terminal_app):
    async def exercise():
        app = terminal_app
        app.connection = FakeConnection()
        app._input_history = ["previous task"]
        async with app.run_test(size=(38, 12)) as pilot:
            editor = app.query_one(Composer)
            editor.load_draft("/")
            await pilot.pause()
            await pilot.press("up")
            await pilot.pause()
            assert editor.text == "/"
            assert app._command_index == 7
            content = app.query_one("#shortcut-content").renderable
            assert "› /delete         Delete this chat" in content.plain
            assert any(content.plain[s.start:s.end] == "         Delete this chat"
                       and str(s.style) == "bold #ffffff" for s in content.spans)
            assert not any("reverse" in str(s.style) for s in content.spans)
            menu = app.query_one("#shortcuts")
            assert menu.scroll_y > 0
            assert menu.scroll_y <= 7 < menu.scroll_y + menu.scrollable_content_region.height
            await pilot.press("down", "down", "tab")
            await pilot.pause()
            assert editor.text == "/help"
            await pilot.press("enter")
            await pilot.pause()
            assert editor.text == ""
            assert "Keyboard shortcuts" in str(app.store.get("main").committed_blocks[-1])
            editor.load_draft("/obs")
            await pilot.press("enter")
            await pilot.pause()
            assert editor.text == ""
            editor.load_draft("draft")
            await pilot.press("up")
            assert editor.text == "/observability"
            await pilot.press("down")
            assert editor.text == "draft"
            editor.load_draft("/COMMAND")
            await pilot.pause()
            assert [c[0] for c in app._matching_commands(editor.text)] == ["/commands", "/stop-command"]
            await pilot.press("down", "tab")
            assert editor.text == "/stop-command "
            editor.load_draft("obs")
            await pilot.pause()
            await pilot.press("tab")
            assert editor.text == "obs"
            editor.load_draft("help")
            await pilot.pause()
            await pilot.press("enter")
            assert app.connection.pending[-1]["content"] == "help"
    asyncio.run(exercise())
