"""Concurrency and delivery contracts exercised without a live model."""
import json
from queue import SimpleQueue
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from praxis_agent.agent import Agent
from praxis_agent.context import ConversationHistory, ContextManager
from praxis_agent.context.operations import ContextOperation
from praxis_agent.execution import ExecutionStopped, Run, run_operation
from praxis_agent.model import ModelResponse
from praxis_agent.multi_agent.context import inherited_context
from praxis_agent.multi_agent.models import AgentHandle, AgentMessage, AgentStatus, ExecutionControl, MultiAgentConfig
from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.multi_agent.tools import register_parent_tools, register_worker_tools
from praxis_agent.planning import RunState
from praxis_agent.tools import ToolContext, ToolRegistry


@pytest.fixture
def runtime(tmp_path):
    value = AgentRuntime(None, MultiAgentConfig(), workspace=tmp_path)
    for name in ("A", "B"):
        value.handles[name] = AgentHandle(name, "turn", ExecutionControl(), status=AgentStatus.RUNNING)
    yield value
    for handle in value.handles.values():
        handle.status = AgentStatus.IDLE
    value.shutdown(lambda: None)


def waiting(runtime, agent_id="parent", **kwargs):
    runtime.context_item(agent_id)
    entered, finished = Event(), Event()
    result = []

    def run():
        def pump():
            entered.set()
        result.append(runtime.wait_agents(5, pump, agent_id=agent_id, **kwargs))
        finished.set()
    thread = Thread(target=run)
    thread.start()
    assert entered.wait(2)
    return thread, finished, result


def test_peer_message_wakes_wait_and_is_not_consumed(runtime):
    thread, finished, result = waiting(runtime, "B")
    receipt = runtime.send_message("A", "B", "Need the cache contract")
    assert finished.wait(2)
    thread.join()
    assert result[0]["reason"] == "message"
    assert result[0]["unread_messages"] == 1
    assert len(runtime.handles["B"].inbox) == 1
    accepted = []
    runtime.receive("B", accepted.extend)
    assert receipt["message_id"] in str(accepted)
    assert '"sender_id": "A"' in str(accepted)
    assert not runtime.handles["B"].inbox


def test_any_worker_can_wake_parent(runtime):
    thread, finished, result = waiting(runtime)
    runtime.send_message("B", "parent", "Unrelated worker needs a decision")
    assert finished.wait(2)
    thread.join()
    assert result[0]["reason"] == "message"
    assert len(runtime.inbox) == 1


def test_user_input_interrupts_wait_without_consuming_it(runtime):
    inputs = SimpleQueue()
    thread, finished, result = waiting(runtime, has_user_input=lambda: not inputs.empty())
    inputs.put("Change requirements")
    runtime.wake_waiters()
    assert finished.wait(2)
    thread.join()
    assert result[0]["reason"] == "user_input"
    assert inputs.get_nowait() == "Change requirements"


def test_arrival_between_input_check_and_condition_wait_is_observed(runtime):
    checks = []
    def input_check():
        checks.append(True)
        runtime.send_message("A", "parent", "Arrived at the boundary")
        return False
    assert runtime.wait_agents(5, lambda: None, has_user_input=input_check)["reason"] == "message"
    assert len(checks) == 1


def test_failed_accept_and_late_mail_are_preserved(runtime):
    runtime.send_message("A", "parent", "first")
    def fail(_):
        raise OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        runtime.receive("parent", fail)
    assert runtime.wait_agents(0, lambda: None)["unread_messages"] == 1
    def save(_):
        runtime.send_message("B", "parent", "arrived while accepting")
    assert runtime.receive("parent", save)
    assert [m.content for m in runtime.inbox] == ["arrived while accepting"]


@pytest.mark.parametrize("status", [AgentStatus.CLOSING, AgentStatus.CLOSED])
def test_send_to_closed_peer_rejected_without_enqueuing(runtime, status):
    runtime.handles["B"].status = status
    with pytest.raises(ValueError, match="closed"):
        runtime.send_message("A", "B", "Too late")
    assert not runtime.handles["B"].inbox


def test_foreign_and_self_targets_rejected(runtime):
    with pytest.raises(ValueError, match="unknown"):
        runtime.send_message("A", "different-runtime-id", "test")
    with pytest.raises(ValueError, match="yourself"):
        runtime.send_message("A", "A", "test")


def test_close_and_rescope_wake_wait_but_regular_status_does_not(runtime):
    thread, finished, result = waiting(runtime, "A")
    runtime.close_agent("B")
    assert finished.wait(2)
    thread.join()
    assert result[0]["reason"] == "team_changed"
    runtime.context_item("A")
    with runtime.condition:
        runtime._status(runtime.handles["B"], AgentStatus.RUNNING)
    assert runtime.wait_agents(0, lambda: None, agent_id="A")["reason"] == "timeout"
    with runtime.condition:
        runtime._update_task(runtime.handles["B"], [AgentMessage("parent", "B", "scope", task_summary="new scope")])
    assert runtime.wait_agents(0, lambda: None, agent_id="A")["reason"] == "team_changed"


def test_idle_peer_resumes_with_history_and_membership_does_not_start_it(tmp_path):
    inputs = []
    class Model:
        def create_response(self, input_items, tools, **kwargs):
            inputs.append(str(input_items))
            return ModelResponse(({"role": "assistant", "content": "done"},), "done", ())
    def factory(agent_id, control, sink):
        return Agent(Model(), ToolRegistry(), tmp_path, execution_control=control, event_sink=sink,
                     team_context=lambda: runtime.context_item(agent_id),
                     receive_messages=lambda accept: runtime.receive(agent_id, accept))
    runtime = AgentRuntime(factory, MultiAgentConfig(), workspace=tmp_path)
    def idle(agent_id):
        with runtime.condition:
            assert runtime.condition.wait_for(lambda: runtime.handles[agent_id].status == AgentStatus.IDLE, 3)
    try:
        first = runtime.spawn("task A", "backend cache", working_directory=".")["agent_id"]
        idle(first)
        worker = runtime.handles[first].agent
        second = runtime.spawn("task B", "frontend", working_directory=".")["agent_id"]
        idle(second)
        assert len(inputs) == 2
        assert runtime.handles[first].agent is worker
        runtime.send_message(second, first, "Useful frontend contract")
        idle(first)
        assert len(inputs) == 3
        assert "task A" in inputs[-1] and "Useful frontend contract" in inputs[-1]
        assert second in inputs[-1] and "frontend" in inputs[-1]
        assert "collaboration_status" not in str(worker._history.snapshot())
    finally:
        runtime.shutdown(lambda: None)


def test_wait_cancel_wakes_and_releases_operation(runtime):
    scope = Run("waiting worker")
    entered, stopped = Event(), Event()
    def execute():
        try:
            scope.execute(lambda: run_operation("wait", lambda _: runtime.wait_agents(60, entered.set)))
        except ExecutionStopped:
            stopped.set()
    thread = Thread(target=execute)
    thread.start()
    assert entered.wait(2)
    scope.cancel()
    assert stopped.wait(2)
    thread.join()


def test_inherited_context_uses_projection_and_excludes_unfinished_batch(tmp_path):
    history = ConversationHistory()
    history.extend([
        {"role": "user", "content": "Keep public API; codeword orchid"},
        {"type": "function_call", "call_id": "complete", "name": "read_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "complete", "output": "retained evidence"},
        {"role": "developer", "content": "parent-only orchestration"},
        {"type": "reasoning", "id": "pending-reasoning", "summary": []},
        {"type": "function_call", "call_id": "partial", "name": "read_file", "arguments": "{}"},
        {"type": "function_call", "call_id": "spawn", "name": "spawn_agent", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "partial", "output": "partial batch"},
    ])
    manager = ContextManager()
    context = ToolContext(tmp_path, RunState("test"), history=history, context_manager=manager)
    inherited = inherited_context(context)
    assert len(inherited) == 3
    assert "orchid" in str(inherited) and "retained evidence" in str(inherited)
    assert "spawn" not in str(inherited) and "pending-reasoning" not in str(inherited)
    inherited[0]["content"] = "changed"
    assert "orchid" in str(history.snapshot()[0])


def test_inherit_is_optional_and_worker_tools_support_wait_and_target(tmp_path):
    workers = []
    class Worker:
        def __init__(self):
            self.history = ()
        def restore_history(self, history):
            self.history = history
        def run(self, task):
            return "done"
        def close(self):
            pass
    def factory(*args):
        worker = Worker()
        workers.append(worker)
        return worker
    runtime = AgentRuntime(factory, MultiAgentConfig(), workspace=tmp_path)
    registry = ToolRegistry()
    register_parent_tools(registry, runtime, lambda: None)
    history = ConversationHistory()
    history.append({"role": "user", "content": "Do not change API"})
    context = ToolContext(tmp_path, RunState("test"), history=history)
    try:
        for inherit in (False, True):
            result = registry.execute("spawn_agent", {"task": "bounded work", "working_directory": ".", "inherit_context": inherit}, context=context)
            assert result.success, result.output
        assert not workers[0].history
        assert "Do not change API" in str(workers[1].history)
        worker_registry = ToolRegistry()
        agent_id = next(iter(runtime.handles))
        register_worker_tools(worker_registry, runtime, agent_id)
        assert {"send_message", "wait_agents"} <= set(worker_registry.names)
        assert "agent_ids" not in str(worker_registry.definitions())
    finally:
        runtime.shutdown(lambda: None)


def test_inheritance_keeps_summary_without_resurrecting_hidden_content(tmp_path):
    history = ConversationHistory()
    history.extend([
        {"role": "user", "content": "original context that was compressed"},
        {"role": "assistant", "content": "obsolete detail"},
        {"role": "user", "content": "new task"},
    ])
    manager = ContextManager()
    manager.operations.record(ContextOperation(
        operation_id="op1", action="compress", message_ids=("m1", "m2"),
        replacement={"message_id": "summary1", "payload": {"role": "user", "content": "retained summary: preserve API"}},
    ))
    result = inherited_context(ToolContext(tmp_path, RunState("test"), history=history, context_manager=manager))
    assert "retained summary" in str(result) and "new task" in str(result)
    assert "obsolete detail" not in str(result) and "original context" not in str(result)
