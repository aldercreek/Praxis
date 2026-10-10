import json
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from praxis_agent.agent import Agent
from praxis_agent.events import AssistantTextCompleted, TurnCompleted
from praxis_agent.model import ModelResponse
from praxis_agent.multi_agent.models import AgentHandle, AgentStatus, ExecutionControl, MultiAgentConfig
from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.tools import ToolRegistry


def text_response(text):
    return ModelResponse(({"role": "assistant", "content": text},), text, ())


def test_parent_rejects_early_answer_and_reads_results_before_finishing(tmp_path):
    started, release = Event(), Event()
    inputs, events = [], []

    def factory(agent_id, control, sink):
        def run(task):
            started.set()
            assert release.wait(5)
            return "Correction: the expected value is 42."
        return SimpleNamespace(run=run, close=lambda: None)

    runtime = AgentRuntime(factory, MultiAgentConfig(max_workers=1))
    first = runtime.spawn("Verify the value")["agent_id"]
    assert started.wait(5)
    second = runtime.spawn("Check the citation")["agent_id"]

    class Model:
        def create_response(self, input_items, tools, **kwargs):
            inputs.append(list(input_items))
            assert not any(isinstance(e, TurnCompleted) for e in events)
            if len(inputs) == 1:
                return text_response("Premature answer: 41.")
            if len(inputs) == 2:
                reminder = next(item for item in input_items if isinstance(item, dict)
                                and "This turn cannot finish" in item.get("content", ""))
                workers = json.loads(reminder["content"].split("\n", 1)[1])
                assert {w["status"] for w in workers} == {"running", "queued"}
                release.set()
                with runtime.condition:
                    assert runtime.condition.wait_for(
                        lambda: all(runtime.handles[a].status == AgentStatus.IDLE
                                    for a in (first, second)), timeout=5,
                    )
                # Both workers finish during this model request. Their unread mail
                # must still prevent returning an answer based on the old context.
                return text_response("Stale answer: 41.")
            assert len(inputs) == 3
            assert "expected value is 42" in str(input_items)
            assert not runtime.inbox
            return text_response("Verified answer: 42.")

    agent = Agent(model_client=Model(), tool_registry=ToolRegistry(), workspace=tmp_path,
                  collaboration=runtime, event_sink=SimpleNamespace(emit=events.append),
                  receive_messages=lambda accept: runtime.receive("parent", accept))
    try:
        assert agent.run("Give the verified value") == "Verified answer: 42."
        assert len(inputs) == 3
        assert sum(isinstance(e, TurnCompleted) for e in events) == 1
        assert sum(isinstance(e, AssistantTextCompleted) for e in events) == 1
        assert all(h.status == AgentStatus.IDLE for h in runtime.handles.values())
    finally:
        release.set()
        agent.close()


@pytest.mark.parametrize("status", [AgentStatus.RUNNING, AgentStatus.QUEUED])
def test_active_worker_without_mail_blocks_finalization(status):
    runtime = AgentRuntime(None, MultiAgentConfig())
    handle = AgentHandle("worker", "turn", ExecutionControl(), status=status)
    runtime.handles[handle.agent_id] = handle
    items = []
    try:
        assert not runtime.can_finalize(items.extend)
        assert "wait_agents" in items[0]["content"]
        assert status.value in items[0]["content"]
    finally:
        handle.status = AgentStatus.IDLE
        runtime.shutdown(lambda: None)


def test_idle_help_request_must_be_delivered_but_does_not_block_afterward():
    runtime = AgentRuntime(None, MultiAgentConfig())
    handle = AgentHandle("worker", "turn", ExecutionControl())
    runtime.handles[handle.agent_id] = handle
    runtime.send_parent(handle.agent_id, "Task incomplete. Please reassign remaining work.")
    items = []
    try:
        assert not runtime.can_finalize(items.extend)
        assert "Please reassign" in str(items)
        assert runtime.can_finalize(items.extend)
        runtime.close_agent(handle.agent_id)
        assert runtime.can_finalize(items.extend)
        assert len(items) == 1
    finally:
        runtime.shutdown(lambda: None)


def test_final_message_is_not_lost_when_persistence_fails():
    runtime = AgentRuntime(None, MultiAgentConfig())
    handle = AgentHandle("worker", "turn", ExecutionControl())
    runtime.handles[handle.agent_id] = handle
    runtime.send_parent(handle.agent_id, "Important correction")

    def fail(items):
        raise OSError("database unavailable")

    try:
        with pytest.raises(OSError, match="database unavailable"):
            runtime.can_finalize(fail)
        assert len(runtime.inbox) == 1
        items = []
        assert not runtime.can_finalize(items.extend)
        assert "Important correction" in str(items)
        assert runtime.can_finalize(items.extend)
    finally:
        runtime.shutdown(lambda: None)


def test_worker_finishing_between_mail_and_status_checks_is_not_missed(monkeypatch):
    runtime = AgentRuntime(None, MultiAgentConfig())
    handle = AgentHandle("worker", "turn", ExecutionControl(), status=AgentStatus.RUNNING)
    runtime.handles[handle.agent_id] = handle
    mail_checked, finish_attempted, finished = Event(), Event(), Event()
    original_receive = runtime.receive

    def receive(agent_id, accept):
        result = original_receive(agent_id, accept)
        mail_checked.set()
        assert finish_attempted.wait(5)
        return result

    monkeypatch.setattr(runtime, "receive", receive)

    def finish():
        assert mail_checked.wait(5)
        finish_attempted.set()
        with runtime.condition:
            runtime._report(handle, "Late correction")
            runtime._status(handle, AgentStatus.IDLE)
        finished.set()

    thread = Thread(target=finish)
    thread.start()
    items = []
    try:
        assert not runtime.can_finalize(items.extend)
        assert finished.wait(5)
        assert not runtime.can_finalize(items.extend)
        assert "Late correction" in str(items)
        assert runtime.can_finalize(items.extend)
    finally:
        mail_checked.set()
        thread.join(timeout=5)
        handle.status = AgentStatus.IDLE
        runtime.shutdown(lambda: None)
