import asyncio
import json
import sys
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from textual.widgets import TextArea

from praxis_agent.agent import Agent
from praxis_agent.execution import ExecutionStopped, Run, current_handle
from praxis_agent.mcp import MCPConnections
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.multi_agent.models import AgentStatus, MultiAgentConfig
from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.planning import RunStatus
from praxis_agent.sessions import SQLiteSessionStore
from praxis_agent.terminal.app import ApprovalPanel, PraxisApp
from praxis_agent.terminal.views import ViewEventSink
from praxis_agent.tools import create_default_registry


def answer(text="done"):
    return ModelResponse(({"role": "assistant", "content": text},), text, ())


def persisted_agent(path, model, sink, **kwargs):
    store = SQLiteSessionStore(path / "sessions.sqlite3")
    session = store.create_session(path, "test")
    return Agent(model, create_default_registry(), path, event_sink=sink,
                 session_store=store, session_id=session.id, session_model="test", **kwargs)


async def until(pilot, predicate):
    for _ in range(60):
        if predicate():
            return
        await pilot.pause(0.05)
    assert predicate()


def test_ui_waits_for_cleanup_preserves_queued_input_and_runs_again(tmp_path):
    started, wake, cleaning, release = (Event() for _ in range(4))

    class Client:
        calls = 0
        def create_response(self, *args, **kwargs):
            self.calls += 1
            if self.calls != 1:
                return answer()
            handle = current_handle()
            handle.on_cancel(wake.set)
            started.set()
            try:
                assert wake.wait(5)
                handle.check()
            finally:
                cleaning.set()
                assert release.wait(5)

    async def exercise():
        sink, client = ViewEventSink(), Client()
        app = PraxisApp(sink, initial_task="first")
        app.agent = persisted_agent(tmp_path, client, sink)
        async with app.run_test() as pilot:
            try:
                await until(pilot, started.is_set)
                first = app.current_run
                app.user_input.put("keep this queued instruction")
                await pilot.press("escape", "escape")
                await until(pilot, cleaning.is_set)
                assert app.stopping and app.busy
                assert not first.finished.done()
                field = app.query_one("#task", TextArea)
                field.text = "second"
                await pilot.press("enter")
                assert field.text == "second"
                assert client.calls == 1
                release.set()
                await until(pilot, lambda: not app.busy)
                assert app.is_running
                assert app.store.views["main"].state == "cancelled"
                assert app.agent.run_state.status == RunStatus.CANCELLED
                assert isinstance(first.finished.exception(), ExecutionStopped)
                assert app.paused_inputs == ["keep this queued instruction"]
                assert app.user_input.empty()
                with app.agent.session_store._connection() as connection:
                    assert connection.execute("SELECT status FROM runs").fetchone()[0] == "cancelled"
                await pilot.press("enter")
                await until(pilot, lambda: client.calls == 2 and not app.busy)
                assert app.agent.run_state.status == RunStatus.COMPLETED
                assert app.current_run is not first
                assert not app.current_run.cancelled.is_set()
                await pilot.press("ctrl+r")
                await until(pilot, lambda: client.calls == 3 and not app.busy)
                assert app.paused_inputs == []
            finally:
                release.set()
                app._runner.join(3)
        app.agent.close()
    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["approval", "command"])
def test_esc_cancels_tool_or_approval_repairs_history_and_preserves_session(tmp_path, phase):
    call = FunctionCall("command-1", "run_command", json.dumps({
        "argv": [sys.executable, "-u", "-c",
                 "from pathlib import Path; import time; Path('started').write_text('yes'); print('partial', flush=True); time.sleep(60)"],
    }))

    class Client:
        calls = 0
        def create_response(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(({
                    "type": "function_call", "call_id": call.call_id,
                    "name": call.name, "arguments": call.arguments,
                },), "", (call,))
            return answer()

    async def exercise():
        sink, client = ViewEventSink(), Client()
        app = PraxisApp(sink, initial_task="run command")
        app.agent = persisted_agent(tmp_path, client, sink,
                                    approve_tool_call=app.approve_tool_call,
                                    bypass_approvals=phase == "command")
        session_id = app.agent.session_id
        async with app.run_test() as pilot:
            if phase == "approval":
                await until(pilot, lambda: app.query_one(ApprovalPanel).display)
                await pilot.press("ctrl+t")  # Esc must stop even in the details modal.
            else:
                await until(pilot, lambda: (tmp_path / "started").exists())
            await pilot.press("escape")
            await until(pilot, lambda: not app.busy)
            assert app.is_running
            assert app.agent.run_state.status == RunStatus.CANCELLED
            assert not app.query_one(ApprovalPanel).display
            if phase == "approval":
                assert not (tmp_path / "started").exists()
            history = app.agent.session_store.load_conversation_items(session_id)
            outputs = [item for item in history if isinstance(item, dict) and item.get("type") == "function_call_output"]
            assert [item["call_id"] for item in outputs] == ["command-1"]
            if phase == "command":
                assert "partial" in outputs[0]["output"]
                assert any("partial" in getattr(block, "output", "")
                           for block in app.store.views["main"].committed_blocks)
            app.query_one("#task", TextArea).text = "continue"
            await pilot.press("enter")
            await until(pilot, lambda: client.calls == 2 and not app.busy)
            assert app.agent.session_id == session_id
            assert app.agent.run_state.status == RunStatus.COMPLETED
        app.agent.close()
    asyncio.run(exercise())


def test_cancel_parent_drains_running_and_queued_workers_then_reuses_runtime(tmp_path):
    started, cleanup, release = Event(), Event(), Event()
    invoked = []
    workers = []

    def factory(agent_id, control, sink):
        class Client:
            calls = 0
            def create_response(self, *args, **kwargs):
                self.calls += 1
                invoked.append(agent_id)
                if self.calls > 1:
                    return answer()
                handle, wake = current_handle(), Event()
                handle.on_cancel(wake.set)
                started.set()
                try:
                    assert wake.wait(5)
                    handle.check()
                finally:
                    cleanup.set()
                    assert release.wait(5)
        worker = Agent(Client(), create_default_registry(), tmp_path, execution_control=control, event_sink=sink)
        workers.append(worker)
        return worker

    runtime = AgentRuntime(factory, MultiAgentConfig(max_workers=1), workspace=tmp_path)
    parent = Run("parent")
    runtime.run = parent
    ids = []
    def execute():
        try:
            parent.execute(lambda: body())
        except ExecutionStopped:
            pass
    def body():
        ids.append(runtime.spawn("first", working_directory=".")["agent_id"])
        ids.append(runtime.spawn("queued", working_directory=".")["agent_id"])
        runtime.wait_agents(ids, 60, lambda: None)
        parent.check()
    thread = Thread(target=execute, daemon=True)
    thread.start()
    try:
        assert started.wait(3)
        with runtime.condition:
            assert runtime.condition.wait_for(lambda: len(runtime.handles) == 2, timeout=3)
        parent.cancel()
        assert cleanup.wait(3)
        assert not parent.finished.done()
        release.set()
        with pytest.raises(ExecutionStopped):
            parent.finished.result(timeout=3)
        thread.join(3)
        assert len(invoked) == 1  # Queued worker did no model/tool work.
        assert all(handle.status == AgentStatus.IDLE for handle in runtime.handles.values())
        assert all(not worker._closed for worker in workers)
        next_run = Run("next parent")
        runtime.run = next_run
        def again():
            runtime.send_many([{"agent_id": ids[0], "content": "continue"}])
            runtime.wait_agents([ids[0]], 5, lambda: None)
        next_run.execute(again)
        assert len(invoked) == 2
        assert runtime.accepting
    finally:
        release.set()
        thread.join(3)
        runtime.shutdown(lambda: None)


def test_early_cancel_still_records_a_cancelled_turn_and_never_calls_model(tmp_path):
    scope = Run()
    scope.cancel()
    model = SimpleNamespace(create_response=lambda *a, **kw: pytest.fail("model started"))
    agent = persisted_agent(tmp_path, model, ViewEventSink())
    try:
        with pytest.raises(ExecutionStopped):
            scope.execute(lambda: agent.run("stop before dispatch"))
        assert agent.run_state.status == RunStatus.CANCELLED
        assert isinstance(scope.finished.exception(), ExecutionStopped)
    finally:
        agent.close()


def test_mcp_cancel_waits_for_coroutine_cleanup_and_keeps_connection(tmp_path):
    entered, cleaning, release = Event(), Event(), Event()
    loop = asyncio.new_event_loop()
    serving = Thread(target=loop.run_forever, daemon=True)
    serving.start()
    mcp = MCPConnections(tmp_path / "absent.json")
    mcp.loop = loop
    mcp.servers = {"test": {}}
    mcp.locks = {"test": asyncio.Lock()}

    class Client:
        calls = 0
        async def call_tool(self, tool, arguments):
            self.calls += 1
            if self.calls > 1:
                return SimpleNamespace(content=[], is_error=False, model_dump=lambda **kw: {"ok": True})
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await asyncio.to_thread(release.wait, 5)

    client = Client()
    mcp.clients = {"test": client}
    scope = Run()
    def execute():
        try:
            scope.execute(lambda: mcp.call("test", "blocked", {}))
        except ExecutionStopped:
            pass
    thread = Thread(target=execute, daemon=True)
    thread.start()
    try:
        assert entered.wait(3)
        scope.cancel()
        assert cleaning.wait(3)
        assert not scope.finished.done()
        release.set()
        with pytest.raises(ExecutionStopped):
            scope.finished.result(timeout=3)
        assert Run().execute(lambda: mcp.call("test", "again", {})).success
        assert client.calls == 2
    finally:
        release.set()
        thread.join(3)
        asyncio.run_coroutine_threadsafe(loop.shutdown_default_executor(), loop).result(3)
        loop.call_soon_threadsafe(loop.stop)
        serving.join(3)
        loop.close()
