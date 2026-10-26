import asyncio
import json

import httpx
import pytest
from test_sessions import BlockingModel
from textual.widgets import TextArea

from praxis_agent.mcp import MCPConnections
from praxis_agent.terminal.app import PraxisApp
from praxis_agent.terminal.connection import TerminalSession
from praxis_agent.terminal.views import ViewEventSink
from praxis_agent.terminal.widgets import DetailsScreen
from praxis_agent.terminal.delete_dialog import DeleteSessionDialog


def test_terminal_can_submit_supplementary_input_and_disconnect_without_cancel(service):
    model = BlockingModel()
    h = service(model)
    session = h.create()
    client = h.client()
    connection = TerminalSession(client, session)

    async def exercise():
        app = PraxisApp(ViewEventSink(), connection=connection, initial_task="first")
        async with app.run_test() as pilot:
            async with asyncio.timeout(5):
                while not app.busy:
                    await pilot.pause(0.05)
            app.query_one("#task", TextArea).text = "follow up"
            await pilot.press("enter")
            async with asyncio.timeout(5):
                while not h.store.pending(session["id"]):
                    await pilot.pause(0.05)
            await pilot.press("ctrl+q")
        assert h.store.summary(session["id"])["active_turn"]

    try:
        asyncio.run(exercise())
    finally:
        model.release.set()
    h.settled(session)
    assert "follow up" in str(model.calls[-1])


def test_two_terminals_cancel_and_resume_same_session(service):
    model = BlockingModel()
    h = service(model)
    session = h.create()
    first = TerminalSession(h.client(), session)
    second = TerminalSession(h.client(), session)
    first.send("task")
    assert model.started.wait(3)
    first.send("retained")

    async def exercise():
        app = PraxisApp(ViewEventSink(), connection=second)
        async with app.run_test() as pilot:
            async with asyncio.timeout(5):
                while not app.busy:
                    await pilot.pause(0.05)
            await pilot.press("escape")
            async with asyncio.timeout(5):
                while app.busy or not app.paused_inputs:
                    await pilot.pause(0.05)
            assert app.paused_inputs == ["retained"]
            await pilot.press("ctrl+r")
            async with asyncio.timeout(5):
                while len(model.calls) < 2 or app.busy or app.paused_inputs:
                    await pilot.pause(0.05)
            assert not app.paused_inputs

    asyncio.run(exercise())


def test_terminal_delete_confirms_and_other_terminal_receives_removal(service):
    h = service()
    session = h.create()
    h.submit(session, "delete this chat")
    h.settled(session)
    connection = TerminalSession(h.client(), session)
    other = TerminalSession(h.client(), session)

    async def exercise():
        app = PraxisApp(ViewEventSink(), connection=connection)
        async with app.run_test() as pilot:
            editor = app.query_one("#task", TextArea)
            editor.text = "/delete"
            await pilot.press("enter")
            assert isinstance(app.screen, DeleteSessionDialog)
            await pilot.press("escape")
            assert h.manager.list()["sessions"]
            editor.text = "/delete"
            await pilot.press("enter")
            await pilot.click("#delete-chat")
            async with asyncio.timeout(5):
                while not connection.deleted:
                    await pilot.pause(0.02)
            assert h.manager.list()["sessions"] == []
            async with asyncio.timeout(5):
                while not other.deleted or getattr(app, "_deleting_session", False):
                    other.drain(app.store)
                    await pilot.pause(0.02)
            editor.text = "must not start a turn"
            await pilot.press("enter")
            assert len(h.agents) == 1
            editor.text = "/clear"
            await pilot.press("enter")
            async with asyncio.timeout(5):
                while connection.id == session["id"]:
                    await pilot.pause(0.02)
            assert not connection.deleted
    asyncio.run(exercise())


@pytest.mark.parametrize("initial", [False, True])
def test_mcp_command_reads_shared_status_without_starting_a_turn(service, tmp_path, initial):
    h = service()
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {"数据库": {"enabled": False}, "浏览器": {"enabled": False}}}), encoding="utf-8")
    h.manager.mcp = MCPConnections(path)
    h.manager.mcp._state("数据库", "connected")
    session = h.create()
    client = h.client()
    assert client.request("GET", "/api/mcp/servers") == {"servers": [
        {"name": "数据库", "state": "connected"}, {"name": "浏览器", "state": "disabled"}
    ]}
    assert httpx.get(h.endpoint["url"] + "/api/mcp/servers").status_code == 401

    async def exercise():
        app = PraxisApp(ViewEventSink(), connection=TerminalSession(client, session), initial_task="/mcp" if initial else "")
        async with app.run_test() as pilot:
            if not initial:
                app.query_one("#task", TextArea).text = "/mcp"
                await pilot.press("enter")
            async with asyncio.timeout(5):
                while not isinstance(app.screen, DetailsScreen):
                    await pilot.pause(0.02)
            assert "数据库" in str(app.screen.content)
            assert "已连接" in str(app.screen.content)
            assert "工具" not in str(app.screen.content)
            await pilot.press("escape")
    asyncio.run(exercise())
    assert h.agents == []
    assert h.store.snapshot(session["id"])["messages"] == []
