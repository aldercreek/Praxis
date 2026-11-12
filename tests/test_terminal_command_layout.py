import asyncio

import pytest

from praxis_agent.terminal.cells import MessageBlock
from praxis_agent.terminal.composer import Composer
from support.terminal import FakeConnection


@pytest.mark.parametrize("size", [(100, 30), (44, 18)])
@pytest.mark.parametrize("history", [False, True])
def test_command_results_follow_history_and_keep_prompt_close(terminal_app, tmp_path, size, history):
    async def exercise():
        app = terminal_app
        app.connection = FakeConnection()
        if history:
            app.store.get("main").committed_blocks.append(MessageBlock("Previous conversation remains visible."))
        async with app.run_test(size=size) as pilot:
            await app._command_result("/observability", "Provider: phoenix\nStatus: ready\nLocal viewer: http://127.0.0.1:51209")
            await pilot.pause()
            await app._command_result("/mcp", "No MCP servers configured.")
            await pilot.pause()
            assert app.screen is app._main_screen
            last = app.query_one(f"#record-{2 if history else 1}")
            editor = app.query_one(Composer)
            assert 0 <= editor.region.y - last.region.bottom <= 4
            assert editor.region.bottom < size[1]
            assert not app.query_one("#status").display
            assert "/observability" in str(app.store.get("main").committed_blocks[-2])
            assert last.is_on_screen
            editor.load_draft("/")
            await pilot.press("down")
            await pilot.pause()
            assert editor.is_on_screen
            path = app.save_screenshot("command-layout.svg", path=str(tmp_path))
            print(path)
            editor.load_draft("continue")
            await pilot.press("enter")
            assert app.connection.pending[-1]["content"] == "continue"
    asyncio.run(exercise())
