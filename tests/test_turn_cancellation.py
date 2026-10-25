import asyncio
import json
import sys
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult
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
from praxis_agent.terminal.views import AgentView, ToolBlock, ViewEventSink
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
                return CallToolResult(content=[])
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
