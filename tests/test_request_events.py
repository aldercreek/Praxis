from datetime import datetime
from types import SimpleNamespace

import pytest

from praxis_agent.agent import Agent
from praxis_agent.context.manager import ContextManager
from praxis_agent.context.model_metadata import ModelLimits
from praxis_agent.context.models import ConversationHistory
from praxis_agent.context.summarizer import ModelContextSummarizer
from praxis_agent.events import (
    AssistantTextDelta, ChildAgentEvent, CompositeEventSink, ContextCompaction,
    ModelRequestCancelled, ModelRequestCompleted, ModelRequestFailed, ModelRequestStarted,
    TurnCancelled, TurnFailed,
)
from praxis_agent.execution import ExecutionStopped
from praxis_agent.model import ModelResponse, TokenUsage
from praxis_agent.model_errors import IncompleteResponseError, ModelClientError
from praxis_agent.request_events import compaction_events, model_request_events, request_model
from praxis_agent.service.events import ServiceEventSink, display_event
from praxis_agent.service.transcript import Transcript
from praxis_agent.terminal.wire_events import decode_event
from praxis_agent.tools import ToolRegistry
from tests.support.models import model_config


def answer(text="done"):
    return ModelResponse(({"role": "assistant", "content": text},), text, (),
                         "response-1", "actual-model", TokenUsage(100, 10, 110))


@pytest.mark.parametrize("failure", [IncompleteResponseError("partial"), ModelClientError("bad"),
                                    ExecutionStopped("stopped")])
def test_agent_requests_always_end_and_retry_numbers_are_unique(tmp_path, failure):
    events, calls = [], []

    def call(items, tools, on_text_delta):
        calls.append(items)
        on_text_delta("partial" if len(calls) == 1 else "done")
        if len(calls) == 1:
            raise failure
        return answer()

    config = model_config(model="requested-model", model_key="coding", api_key="never-export",
                          parameters={"reasoning_effort": "high"})
    agent = Agent(SimpleNamespace(create_response=call), ToolRegistry(), tmp_path,
                  event_sink=SimpleNamespace(emit=events.append), model_config=config)
    try:
        if isinstance(failure, IncompleteResponseError):
            assert agent.run("work") == "done"
        else:
            with pytest.raises(type(failure)):
                agent.run("work")
        starts = [e for e in events if isinstance(e, ModelRequestStarted)]
        ends = [e for e in events if isinstance(e, (ModelRequestCompleted, ModelRequestFailed, ModelRequestCancelled))]
        assert [e.request_number for e in starts] == list(range(1, len(starts) + 1))
        assert [e.request_number for e in ends] == [e.request_number for e in starts]
        assert isinstance(ends[0], ModelRequestCancelled if isinstance(failure, ExecutionStopped) else ModelRequestFailed)
        assert starts[0].model == "requested-model"
        assert starts[0].model_key == "coding"
        assert starts[0].parameters == {"reasoning_effort": "high"}
        assert starts[0].max_output_tokens == config.model_limits.max_output_tokens
        assert "never-export" not in repr(events)
        assert all(datetime.fromisoformat(e.timestamp).utcoffset().total_seconds() == 0 for e in events)
        if isinstance(failure, ExecutionStopped):
            assert isinstance(events[-1], TurnCancelled)
        elif not isinstance(failure, IncompleteResponseError):
            assert isinstance(events[-1], TurnFailed)
        # A fresh turn gets its own request counter.
        events.clear()
        assert agent.run("again") == "done"
        assert next(e for e in events if isinstance(e, ModelRequestStarted)).request_number == 1
    finally:
        agent.close()


def test_summary_chunks_retries_and_separate_compactions_are_correlated():
    events, calls = [], []

    def call(items, tools, max_output_tokens):
        calls.append(items)
        if len(calls) == 1:
            raise IncompleteResponseError("stream interrupted")
        return answer("short summary")

    config = model_config(model="summary-model", model_key="summary",
                          model_limits=ModelLimits(4000, max_output_tokens=200))
    summary = ModelContextSummarizer(SimpleNamespace(create_response=call),
                                     ModelLimits(4000, max_output_tokens=200), input_limit_tokens=1400,
                                     model_config=config)
    messages = [{"role": "assistant", "content": "evidence " * 1500}]
    sink = SimpleNamespace(emit=events.append)
    with model_request_events("turn", events.append):
        for _ in range(2):
            with compaction_events("turn", sink, 10000, "manual") as progress:
                assert summary.summarize(messages, "retain evidence", on_progress=progress.progress) == "short summary"
                progress.after_tokens = 100
        request_model(SimpleNamespace(create_response=lambda *a, **k: answer()), (), (), config=config)
    starts = [e for e in events if isinstance(e, ModelRequestStarted)]
    ends = [e for e in events if isinstance(e, (ModelRequestCompleted, ModelRequestFailed))]
    assert [e.request_number for e in starts] == list(range(1, len(starts) + 1))
    assert [e.request_number for e in ends] == [e.request_number for e in starts]
    groups = [e for e in events if isinstance(e, ContextCompaction) and e.stage == "started"]
    assert len({e.compaction_id for e in groups}) == 2
    assert len([e for e in events if isinstance(e, ContextCompaction) and e.stage == "completed"]) == 2
    for request in starts[:-1]:
        assert request.purpose == "compression"
        assert request.compaction_id in {e.compaction_id for e in groups}
        assert request.max_output_tokens == 200
        chunk = next(e for e in events if isinstance(e, ContextCompaction)
                     and e.stage == "chunk_started" and e.compaction_id == request.compaction_id
                     and e.round_number == request.round_number and e.chunk_number == request.chunk_number)
        assert chunk.occurred_at <= request.occurred_at
    assert starts[0].round_number == 1 and starts[1].round_number == 2
    assert any(e.chunk_number > 1 for e in starts)
    assert starts[-1].purpose == "agent" and starts[-1].compaction_id is None


@pytest.mark.parametrize("failure", [ExecutionStopped("stop"), ModelClientError("bad gateway")])
def test_compaction_failure_and_cancellation_close_both_lifecycles(failure):
    events = []
    def fail(*args, **kwargs):
        raise failure
    summary = ModelContextSummarizer(SimpleNamespace(create_response=fail), ModelLimits(128000))
    with pytest.raises(type(failure)):
        with compaction_events("turn", SimpleNamespace(emit=events.append), 1000, "manual") as progress:
            summary.summarize([{"role": "user", "content": "evidence"}], "summarize",
                              on_progress=progress.progress)
    assert events[-1].stage == ("cancelled" if isinstance(failure, ExecutionStopped) else "failed")
    assert events[-1].compaction_id == events[0].compaction_id
    assert len([e for e in events if isinstance(e, ModelRequestStarted)]) == 1
    assert len([e for e in events if isinstance(e, (ModelRequestFailed, ModelRequestCancelled))]) == 1
    assert not any(isinstance(e, ContextCompaction) and e.stage == "completed" for e in events)


def test_manual_compression_has_events_and_direct_summary_has_no_model_call():
    events = []
    manager = ContextManager(
        model_limits=ModelLimits(128000),
        summarizer=ModelContextSummarizer(SimpleNamespace(create_response=lambda *a, **k: answer("summary")),
                                          ModelLimits(128000)),
    )
    manager.turn_id = "turn"
    manager.event_sink = SimpleNamespace(emit=events.append)
    history = ConversationHistory()
    history.extend([{"role": "assistant", "content": "old evidence"}, {"role": "user", "content": "continue"}])
    manager.build_view(history)
    mid = manager._current_messages[0]["message_id"]
    manager.change_messages("compress", [mid], instructions="summarize")
    assert [e.stage for e in events if isinstance(e, ContextCompaction)] == [
        "started", "chunk_started", "chunk_completed", "completed",
    ]
    assert len([e for e in events if isinstance(e, ModelRequestCompleted)]) == 1
    manager.build_view(history)
    mid = manager._current_messages[0]["message_id"]
    events.clear()
    manager.change_messages("compress", [mid], summary="already summarized")
    assert [e.stage for e in events if isinstance(e, ContextCompaction)] == ["started", "completed"]
    assert events[-1].request_count == 0


def test_display_excludes_summary_and_retains_source_times_and_request_failure():
    published, observed = [], []
    service = ServiceEventSink(SimpleNamespace(publish=lambda *args: published.append(args)))
    sink = CompositeEventSink((service, SimpleNamespace(emit=observed.append)))
    summary = ChildAgentEvent("parent", "worker", ModelRequestCompleted(
        "child", 1, answer("private summary"), purpose="compression", compaction_id="compaction",
    ))
    sink.emit_live(summary)
    sink.emit_deferred(summary)
    assert observed == [summary] and published == []
    failure = ModelRequestFailed("child", 2, "IncompleteResponseError", "broken", occurred_at=100,
                                 timestamp="2026-11-11T07:00:00+00:00")
    event = ChildAgentEvent("parent", "worker", failure, occurred_at=200)
    kind, turn, agent, data = display_event(event)
    decoded = decode_event(dict(event=kind, turn_id=turn, agent_id=agent, data=data))
    assert decoded.event.occurred_at == 100
    assert decoded.event.timestamp == failure.timestamp
    transcript = Transcript()
    transcript.consume("AssistantTextDelta", turn, agent,
                       {"request_number": 2, "text": "partial", "child_turn_id": "child"})
    transcript.consume(kind, turn, agent, data)
    assert transcript.messages[0]["status"] == "failed"
    transcript.consume("ModelRequestCompleted", turn, agent,
                       {"request_number": 3, "output_text": "private summary", "purpose": "compression"})
    assert len(transcript.messages) == 1
