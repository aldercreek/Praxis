from threading import Lock
from types import SimpleNamespace

import pytest

from praxis_agent.agent import Agent
from praxis_agent.events import (
    AssistantTextDelta, ChildAgentEvent, CompositeEventSink,
    ModelRequestCancelled, ModelRequestCompleted, ModelRequestFailed,
    ModelRequestStarted, TurnCancelled, TurnCompleted, TurnFailed, TurnStarted,
)
from praxis_agent.execution import ExecutionStopped
from praxis_agent.model import ModelResponse
from praxis_agent.model_errors import ModelClientError
from praxis_agent.multi_agent.models import AgentStatus, MultiAgentConfig
from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.service.events import ServiceEventSink
from praxis_agent.tools import ToolRegistry


class RecordingSink:
    def __init__(self):
        self.events = []
        self.lock = Lock()

    def emit(self, event):
        with self.lock:
            self.events.append(event)


class LiveRecordingSink(RecordingSink):
    emit_live = RecordingSink.emit


def model(failure=None):
    def create_response(input_items, tools, on_text_delta):
        on_text_delta("answer")
        if failure is not None:
            raise failure
        return ModelResponse(({"role": "assistant", "content": "answer"},), "answer", ())

    return SimpleNamespace(create_response=create_response)


@pytest.mark.parametrize("failure,request_end,turn_end", [
    (None, ModelRequestCompleted, TurnCompleted),
    (ModelClientError("unavailable"), ModelRequestFailed, TurnFailed),
    (ExecutionStopped("stop"), ModelRequestCancelled, TurnCancelled),
])
def test_main_agent_fans_out_to_service_and_recorder(tmp_path, failure, request_end, turn_end):
    published = []
    ui = ServiceEventSink(SimpleNamespace(publish=lambda *args: published.append(args)))
    recorder = RecordingSink()
    agent = Agent(model(failure), ToolRegistry(), tmp_path,
                  event_sink=CompositeEventSink((ui, recorder)))
    try:
        if failure is None:
            assert agent.run("work") == "answer"
        else:
            with pytest.raises(type(failure)):
                agent.run("work")
        events = recorder.events
        assert isinstance(events[0], TurnStarted)
        assert isinstance(events[-1], turn_end)
        assert sum(isinstance(e, ModelRequestStarted) for e in events) == 1
        assert sum(isinstance(e, request_end) for e in events) == 1
        assert [row[0] for row in published] == [type(e).__name__ for e in events]
        assert [row[1] for row in published] == [e.turn_id for e in events]
        assert [row[3]["timestamp"] for row in published] == [e.timestamp for e in events]
        assert len({id(e) for e in events}) == len(events)
    finally:
        agent.close()


@pytest.mark.parametrize("recorder_type", [RecordingSink, LiveRecordingSink])
def test_worker_fanout_delivers_once_across_live_and_deferred_paths(tmp_path, recorder_type):
    published = []
    ui = ServiceEventSink(SimpleNamespace(publish=lambda *args: published.append(args)))
    recorder = recorder_type()
    sink = CompositeEventSink((ui, recorder))

    def factory(agent_id, control, worker_sink):
        return Agent(model(), ToolRegistry(), tmp_path, event_sink=worker_sink,
                     execution_control=control)

    runtime = AgentRuntime(factory, MultiAgentConfig(max_workers=2), live_output_sink=sink)
    runtime.parent_turn_id = "parent-turn"
    try:
        agent_ids = [runtime.spawn("work", working_directory=str(tmp_path))["agent_id"]
                     for _ in range(2)]
        with runtime.condition:
            assert runtime.condition.wait_for(
                lambda: all(runtime.handles[a].status == AgentStatus.IDLE for a in agent_ids),
                timeout=5,
            )
        runtime.pump_events(sink)
        before = (len(published), len(recorder.events))
        runtime.pump_events(sink)
        assert (len(published), len(recorder.events)) == before

        children = [e for e in recorder.events if isinstance(e, ChildAgentEvent)]
        assert {e.agent_id for e in children} == set(agent_ids)
        assert all(e.turn_id == "parent-turn" for e in children)
        assert len({id(e) for e in recorder.events}) == len(recorder.events)
        child_turns = set()
        for agent_id in agent_ids:
            events = [e.event for e in children if e.agent_id == agent_id]
            child_turns.update(e.turn_id for e in events)
            for event_type in (TurnStarted, ModelRequestStarted, ModelRequestCompleted, TurnCompleted):
                assert sum(isinstance(e, event_type) for e in events) == 1
            assert sum(isinstance(e, AssistantTextDelta) for e in events) == (
                1 if recorder_type is LiveRecordingSink else 0
            )
            displayed = [row for row in published if row[2] == agent_id]
            assert sum(row[0] == "AssistantTextDelta" for row in displayed) == 1
            for event in events:
                matching = [row for row in displayed
                            if row[0] == type(event).__name__ and row[3]["timestamp"] == event.timestamp]
                assert len(matching) == 1
                assert matching[0][3]["child_turn_id"] == event.turn_id
                assert matching[0][3]["occurred_at"] == event.occurred_at
        assert len(child_turns) == 2
    finally:
        runtime.shutdown(lambda: runtime.pump_events(sink))
