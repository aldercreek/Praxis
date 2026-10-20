import asyncio
from queue import SimpleQueue
from threading import Event

import pytest
from textual.widgets import Static, TextArea

from praxis_agent.agent import Agent
from praxis_agent.events import (
    MessagesReceived,
    ToolStarted,
    TurnCompleted,
    TurnStarted,
)
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.sessions import SQLiteSessionStore
from praxis_agent.terminal.app import PraxisApp
from praxis_agent.terminal.cells import MessageBlock
from praxis_agent.terminal.views import ViewEventSink
from praxis_agent.tools import create_default_registry


def answer(text):
    return ModelResponse(({"role": "assistant", "content": text},), text, ())


class Model:
    def __init__(self, responses, on_first_request=lambda: None):
        self.responses = iter(responses)
        self.on_first_request = on_first_request
        self.inputs = []

    def create_response(self, input_items, tools, *, on_text_delta=None):
        self.inputs.append(list(input_items))
        if len(self.inputs) == 1:
            self.on_first_request()
        response = next(self.responses)
        if on_text_delta and response.output_text:
            on_text_delta(response.output_text)
        return response


def user_texts(items):
    return [item["content"].split("\n", 1)[-1] for item in items
            if isinstance(item, dict) and item.get("role") == "user"]


@pytest.mark.parametrize("arrival", ["before_request", "during_model", "during_tool"])
def test_steer_is_ordered_persisted_and_consumed_in_same_turn(tmp_path, arrival):
    pending = SimpleQueue()
    events = []

    def enqueue():
        pending.put("also check tests")
        pending.put("only change Python files")

    class Sink:
        def emit(self, event):
            events.append(event)
            if arrival == "during_tool" and isinstance(event, ToolStarted):
                enqueue()

    responses = [answer("done")]
    if arrival == "before_request":
        enqueue()
    elif arrival == "during_model":
        responses.insert(0, answer("original answer"))
    else:
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        call = FunctionCall("read-1", "read_file", '{"relative_path":"notes.txt"}')
        responses.insert(0, ModelResponse(({
            "type": "function_call", "call_id": call.call_id,
            "name": call.name, "arguments": call.arguments,
        },), "", (call,)))
    model = Model(responses, enqueue if arrival == "during_model" else lambda: None)
    store = SQLiteSessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(tmp_path, "test")
    agent = Agent(model_client=model, tool_registry=create_default_registry(), workspace=tmp_path,
                  event_sink=Sink(), user_input=pending, session_store=store,
                  session_id=session.id, session_model="test")
    try:
        assert agent.run("inspect repository") == "done"
        expected = ["inspect repository", "also check tests", "only change Python files"]
        assert user_texts(model.inputs[-1]) == expected
        assert len(model.inputs) == (1 if arrival == "before_request" else 2)
        assert pending.empty()
        assert sum(isinstance(event, TurnStarted) for event in events) == 1
        assert sum(isinstance(event, TurnCompleted) for event in events) == 1
        received = [event for event in events if isinstance(event, MessagesReceived)]
        assert len(received) == 1 and received[0].source == "user"
        history = store.load_conversation_items(session.id)
        assert user_texts(history) == expected
        if arrival == "during_tool":
            call_index = next(i for i, item in enumerate(history)
                              if item.get("type") == "function_call")
            assert history[call_index + 1]["type"] == "function_call_output"
            assert history[call_index + 1]["call_id"] == "read-1"
            assert history[call_index + 2] == {"role": "user", "content": expected[1]}
        restored_model = Model([answer("continued")])
        restored = Agent(model_client=restored_model, tool_registry=create_default_registry(),
                         workspace=tmp_path)
        try:
            restored.restore_history(history)
            restored.run("continue")
            assert user_texts(restored_model.inputs[0]) == [*expected, "continue"]
        finally:
            restored.close()
    finally:
        agent.close()


@pytest.mark.parametrize("arrival", ["during_model", "after_final_check"])
def test_cli_accepts_steer_and_keeps_late_input(tmp_path, arrival):
    reached = Event()
    release = Event()
    events = []

    def block():
        reached.set()
        assert release.wait(10), "test did not release agent"

    class Sink(ViewEventSink):
        def emit(self, event):
            events.append(event)
            super().emit(event)
            if (arrival == "after_final_check" and isinstance(event, TurnCompleted)
                    and not reached.is_set()):
                block()

    sink = Sink()
    model = Model([answer("original answer"), answer("updated answer")],
                  block if arrival == "during_model" else lambda: None)
    agent = Agent(model_client=model, tool_registry=create_default_registry(),
                  workspace=tmp_path, event_sink=sink)
    app = PraxisApp(sink, initial_task="inspect repository")
    app.agent = agent

    async def exercise():
        async with app.run_test() as pilot:
            try:
                assert await asyncio.to_thread(reached.wait, 5)
                field = app.query_one("#task", TextArea)
                assert app.busy and not field.disabled
                field.text = "/clear"
                await pilot.press("enter")
                assert app.user_input.empty()
                assert field.text == "/clear"
                for text in ("also check tests", "only change Python files"):
                    field.text = text
                    await pilot.press("enter")
                    assert field.text == ""
                assert app.user_input.qsize() == 2
                pending = app.query_one("#pending", Static)
                assert pending.renderable.plain == "› also check tests\n› only change Python files"
                assert all(segment.style.color.name == "#a0a0a0"
                           for y in range(pending.size.height)
                           for segment in pending.render_line(y) if segment.text.strip())
                release.set()
                async with asyncio.timeout(10):
                    while app.busy or len(model.inputs) < 2:
                        await pilot.pause(0.05)
                await pilot.pause(0.15)
                assert app.user_input.empty()
                expected = ["inspect repository", "also check tests", "only change Python files"]
                assert user_texts(model.inputs[1]) == expected
                blocks = app.store.views["main"].committed_blocks
                assert [block.text for block in blocks
                        if isinstance(block, MessageBlock) and block.user] == expected
                assert not pending.display
                for widget in app.query("#blocks Static"):
                    if isinstance(widget.renderable, MessageBlock) and widget.renderable.text in expected[1:]:
                        body = [segment for y in range(widget.size.height)
                                for segment in widget.render_line(y) if segment.text.strip() not in {"", "›"}]
                        assert body and all(segment.style.color == widget.rich_style.color
                                            and not segment.style.dim for segment in body)
                turns = [event for event in events if isinstance(event, TurnStarted)]
                assert len(turns) == (1 if arrival == "during_model" else 2)
                assert not field.disabled
            finally:
                release.set()
                if app._runner is not None:
                    await asyncio.to_thread(app._runner.join, 5)

    try:
        asyncio.run(exercise())
    finally:
        agent.close()
