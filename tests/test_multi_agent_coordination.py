import json
from threading import Event
from types import SimpleNamespace

from praxis_agent.agent import Agent
from praxis_agent.context.capacity import ContextPressure, ContextStatus
from praxis_agent.context.manager import ContextManager
from praxis_agent.context.model_catalog import ModelLimits
from praxis_agent.events import ContextCompaction, ModelRequestStarted, ToolFinished
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.multi_agent.models import AgentStatus, MultiAgentConfig
from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.multi_agent.tools import register_parent_tools, register_worker_tools
from praxis_agent.tools import ToolContext, ToolRegistry, ToolResult
from praxis_agent.planning.state import RunState


def snapshot(runtime):
    return json.loads(runtime.context_item()["content"].split("\n", 1)[1])


def wait_idle(runtime, agent_id):
    with runtime.condition:
        assert runtime.condition.wait_for(
            lambda: runtime.handles[agent_id].status == AgentStatus.IDLE, timeout=5,
        )


def test_snapshot_counts_successful_compactions_and_freezes_time(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("praxis_agent.multi_agent.runtime.monotonic", lambda: clock[0])
    entered, finish, resumed, finish_resume = (Event() for _ in range(4))
    status = ContextStatus(40000, 52800, 75.76, ContextPressure.SOFT)
    calls = []

    def factory(agent_id, control, sink):
        def run(task):
            calls.append(task)
            if len(calls) == 1:
                sink.emit(ModelRequestStarted("r1", 1))
                for stage in ("started", "chunk_started", "chunk_completed", "completed", "failed"):
                    sink.emit(ContextCompaction("r1", stage, 60000))
                sink.emit(ToolFinished("r1", "a", "compress", ToolResult(True, "saved")))
                sink.emit(ToolFinished("r1", "b", "compress", ToolResult(False, "failed")))
                sink.emit(ToolFinished("r1", "c", "delete", ToolResult(True, "saved")))
                entered.set()
                assert finish.wait(5)
            else:
                resumed.set()
                assert finish_resume.wait(5)
            return "Task incomplete; awaiting narrower scope."
        return SimpleNamespace(run=run, close=lambda: None, context_status=status)

    runtime = AgentRuntime(factory, MultiAgentConfig())
    try:
        agent_id = runtime.spawn("Full task", "Initial scope", working_directory=".")["agent_id"]
        assert entered.wait(5)
        clock[0] = 112
        row = snapshot(runtime)[0]
        assert set(row) == {"agent_id", "task", "status", "elapsed_seconds", "workspace"}
        assert runtime.handles[agent_id].context_status.estimated_input_tokens == 40000
        assert runtime.handles[agent_id].compaction_count == 2
        assert row["elapsed_seconds"] == 12
        finish.set()
        wait_idle(runtime, agent_id)
        clock[0] = 500
        assert snapshot(runtime)[0]["elapsed_seconds"] == 12
        ended = runtime.wait_agents([agent_id], 0, lambda: None)
        assert "run ended" in ended["messages"][0]["content"]
        assert "does not imply task completion" in ended["messages"][0]["content"]
        assert "context" not in ended["agents"][0]
        assert "compaction_count" not in ended["agents"][0]
        runtime.send_many([{"agent_id": agent_id, "content": "Only handle A", "task_summary": "Scope A"}])
        assert resumed.wait(5)
        row = snapshot(runtime)[0]
        assert row["task"] == "Scope A"
        assert runtime.handles[agent_id].compaction_count == 0
        assert row["elapsed_seconds"] == 0
        assert runtime.handles[agent_id].context_status.estimated_input_tokens == 40000
    finally:
        finish.set()
        finish_resume.set()
        runtime.shutdown(lambda: None)


def test_queue_time_excluded_and_task_changes_on_message_consumption(monkeypatch):
    clock = [1.0]
    monkeypatch.setattr("praxis_agent.multi_agent.runtime.monotonic", lambda: clock[0])
    started, release = Event(), Event()

    def factory(agent_id, control, sink):
        def run(task):
            started.set()
            assert release.wait(5)
            return "Done"
        return SimpleNamespace(run=run, close=lambda: None, context_status=None)

    runtime = AgentRuntime(factory, MultiAgentConfig(max_workers=1))
    try:
        first = runtime.spawn("First", "First scope", working_directory=".")["agent_id"]
        assert started.wait(5)
        second = runtime.spawn("Second", "Second scope", working_directory=".")["agent_id"]
        clock[0] = 90
        queued = next(r for r in snapshot(runtime) if r["agent_id"] == second)
        assert queued["status"] == "queued"
        assert queued["elapsed_seconds"] == 0
        assert "context" not in queued
        runtime.send_many([{"agent_id": first, "content": "Ordinary feedback"}])
        runtime.receive(first, lambda messages: None)
        assert runtime.handles[first].task_summary == "First scope"
        runtime.send_many([{"agent_id": first, "content": "Narrow work", "task_summary": "Narrow scope"}])
        assert runtime.handles[first].task_summary == "First scope"
        runtime.receive(first, lambda messages: None)
        assert runtime.handles[first].task_summary == "Narrow scope"
        release.set()
        wait_idle(runtime, first)
        wait_idle(runtime, second)
        assert next(r for r in snapshot(runtime) if r["agent_id"] == second)["elapsed_seconds"] == 0
    finally:
        release.set()
        runtime.shutdown(lambda: None)


def test_actual_agent_mail_wakes_wait_then_idle_resumes_with_history(tmp_path):
    final_allowed = Event()
    inputs = []

    class Model:
        def create_response(self, input_items, tools, **kwargs):
            inputs.append(list(input_items))
            if len(inputs) == 1:
                call = FunctionCall("help", "send_message", json.dumps({"content": "Scope is too large. Please split B; I can continue A."}))
                item = {"type": "function_call", "call_id": call.call_id, "name": call.name, "arguments": call.arguments}
                return ModelResponse((item,), "", (call,))
            if len(inputs) == 2:
                assert final_allowed.wait(5)
                text = "Task incomplete; awaiting reassignment."
            else:
                text = "Scope A completed."
            return ModelResponse(({"role": "assistant", "content": text},), text, ())

    def factory(agent_id, control, sink):
        registry = ToolRegistry()
        register_worker_tools(registry, runtime, agent_id)
        return Agent(model_client=Model(), tool_registry=registry, workspace=tmp_path,
                     event_sink=sink, execution_control=control,
                     context_manager=ContextManager(model_limits=ModelLimits(context_window_tokens=64000)),
                     receive_messages=lambda accept: runtime.receive(agent_id, accept))

    runtime = AgentRuntime(factory, MultiAgentConfig())
    try:
        agent_id = runtime.spawn("Investigate A and B", "A and B", working_directory=".")["agent_id"]
        response = runtime.wait_agents([agent_id], 5, lambda: None)
        assert "Please split B" in response["messages"][0]["content"]
        assert response["agents"][0]["status"] == "running"
        assert runtime.handles[agent_id].context_status is not None
        final_allowed.set()
        wait_idle(runtime, agent_id)
        old_agent = runtime.handles[agent_id].agent
        runtime.send_many([{"agent_id": agent_id, "content": "Continue only A; B is reassigned.", "task_summary": "A only"}])
        wait_idle(runtime, agent_id)
        assert runtime.handles[agent_id].agent is old_agent
        assert "Please split B" in str(inputs[-1])
        assert "Continue only A" in str(inputs[-1])
        assert snapshot(runtime)[0]["task"] == "A only"
    finally:
        final_allowed.set()
        runtime.shutdown(lambda: None)


def test_tool_task_labels_validate_without_changing_ordinary_mail(tmp_path):
    runtime = AgentRuntime(lambda *args: SimpleNamespace(run=lambda task: "Done", close=lambda: None), MultiAgentConfig())
    registry = ToolRegistry()
    register_parent_tools(registry, runtime, lambda: None)
    context = ToolContext(tmp_path, RunState("test"))
    try:
        result = registry.execute("spawn_agent", {"task": "Details", "task_summary": "Short scope", "working_directory": "."}, context=context)
        assert result.success
        agent_id = json.loads(result.output)["agent_id"]
        wait_idle(runtime, agent_id)
        result = registry.execute("send_messages", {"messages": [{"agent_id": agent_id, "content": "Feedback", "task_summary": ""}]}, context=context)
        assert json.loads(result.output)["results"][0]["accepted"]
        wait_idle(runtime, agent_id)
        assert snapshot(runtime)[0]["task"] == "Short scope"
        rejected = runtime.send_many([{"agent_id": agent_id, "content": "Bad update", "task_summary": "x" * 201}])
        assert not rejected["results"][0]["accepted"]
        assert snapshot(runtime)[0]["task"] == "Short scope"
        runtime.close_agent(agent_id)
        assert snapshot(runtime) == []
        assert runtime.wait_agents([agent_id], 0, lambda: None)["agents"][0]["status"] == "closed"
    finally:
        runtime.shutdown(lambda: None)
