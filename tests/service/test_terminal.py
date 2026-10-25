import asyncio

from test_sessions import BlockingModel
from textual.widgets import TextArea

from praxis_agent.terminal.app import PraxisApp
from praxis_agent.terminal.connection import TerminalSession
from praxis_agent.terminal.views import ViewEventSink


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
