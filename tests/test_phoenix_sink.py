from datetime import datetime, timezone

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from praxis_agent import events as e
from praxis_agent.model import ModelResponse, TokenUsage
from praxis_agent.phoenix_sink import PhoenixSink
from praxis_agent.tools import ToolResult


def test_span_tree_keeps_original_timing_and_usage_without_duplicate_chunks():
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    sink = PhoenixSink("unused", on_error=lambda message: None, provider=provider)
    clock = dict(occurred_at=100.0, timestamp="2026-11-12T00:00:00+00:00")
    send = lambda event: sink.emit_for_session("session", event)
    send(e.TurnStarted("main", "task", **clock))
    send(e.ContextCompaction("main", "started", 10000, compaction_id="c", **clock))
    send(e.ModelRequestStarted("main", 1, purpose="compression", compaction_id="c", model="summary", **clock))
    response = ModelResponse((), "summary", (), model="actual", usage=TokenUsage(10, 2, 12))
    send(e.ModelRequestCompleted("main", 1, response, purpose="compression", compaction_id="c",
                                 occurred_at=102, timestamp="2026-11-12T01:00:00+00:00"))
    send(e.ContextCompaction("main", "chunk_completed", 10000, compaction_id="c", input_tokens=10, **clock))
    send(e.ContextCompaction("main", "completed", 10000, compaction_id="c", occurred_at=103))
    send(e.ChildAgentEvent("main", "child", e.TurnStarted("child-turn", "child", **clock)))
    send(e.ChildAgentEvent("main", "child", e.ModelRequestStarted("child-turn", 1, **clock)))
    send(e.ChildAgentEvent("main", "child", e.ModelRequestFailed("child-turn", 1, "NetworkError", "broken", occurred_at=104)))
    send(e.ChildAgentEvent("main", "child", e.ModelRequestStarted("child-turn", 2, **clock)))
    send(e.ChildAgentEvent("main", "child", e.ModelRequestCancelled("child-turn", 2, occurred_at=105)))
    send(e.ChildAgentEvent("main", "child", e.TurnCancelled("child-turn", occurred_at=106)))
    send(e.ToolCallPrepared("main", "tool", "read", {"path": "file"}, False, **clock))
    send(e.ToolStarted("main", "tool", "read", **clock))
    send(e.ToolFinished("main", "tool", "read", ToolResult(True, "contents"), occurred_at=107))
    send(e.TurnCompleted("main", occurred_at=108))
    sink.close()
    spans = exporter.get_finished_spans()
    assert len(spans) == 7
    assert len({span.context.trace_id for span in spans}) == 1
    root = next(s for s in spans if s.name == "agent main")
    child = next(s for s in spans if s.name == "agent child")
    compaction = next(s for s in spans if s.name == "context compression")
    summary = next(s for s in spans if s.attributes.get("praxis.purpose") == "compression")
    assert summary.parent.span_id == compaction.context.span_id
    assert child.parent.span_id == root.context.span_id
    assert summary.end_time - summary.start_time == 2_000_000_000
    assert sum(s.attributes.get("llm.token_count.total", 0) for s in spans) == 12
    assert sum(s.status.is_ok is False and s.attributes.get("praxis.outcome") == "failed" for s in spans) == 1
    assert all(s.attributes["session.id"] == "session" for s in spans)
    assert not sink.active


def test_sessions_do_not_share_trace_and_close_marks_unfinished_work():
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    sink = PhoenixSink("unused", on_error=lambda message: None, provider=provider)
    for session in ("one", "two"):
        sink.emit_for_session(session, e.TurnStarted("turn", "task"))
        sink.emit_for_session(session, e.ModelRequestStarted("turn", 1))
        sink.emit_for_session(session, e.AssistantTextDelta("turn", 1, "text"))
    sink.close()
    sink.close()
    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    assert len({s.context.trace_id for s in spans}) == 2
    assert all(s.attributes["praxis.outcome"] == "incomplete" for s in spans)
