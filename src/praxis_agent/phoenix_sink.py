"""Translate Praxis operations to OpenInference spans without changing agent execution."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
import json
from time import monotonic

from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult
from opentelemetry.trace import Status, StatusCode, set_span_in_context
import requests

from praxis_agent import events as ev


def encoded(value):
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > 16000:
        return json.dumps({"truncated": True, "preview": text[:16000]}, ensure_ascii=False)
    return text


@dataclass
class Operation:
    span: object
    wall: int
    clock: float

    def time(self, event):
        return self.wall + max(0, int((event.occurred_at - self.clock) * 1_000_000_000))


class PhoenixSink:
    def __init__(self, url, *, on_error, provider=None):
        if provider is None:
            class Exporter(OTLPSpanExporter):
                def export(self, spans):
                    try:
                        result = super().export(spans)
                    except Exception:
                        on_error("Phoenix export failed. Tracing is unavailable; agent execution continues.")
                        return SpanExportResult.FAILURE
                    if result is SpanExportResult.FAILURE:
                        on_error("Phoenix export failed. Tracing is unavailable; agent execution continues.")
                    return result
            session = requests.Session()
            session.trust_env = False
            exporter = Exporter(endpoint=url + "/v1/traces", timeout=3, session=session)
            provider = TracerProvider(resource=Resource.create({
                "service.name": "praxis", "openinference.project.name": "Praxis",
            }), shutdown_on_exit=False)
            provider.add_span_processor(BatchSpanProcessor(exporter, schedule_delay_millis=500,
                                                           max_queue_size=2048))
        self.provider = provider
        self.tracer = provider.get_tracer("praxis")
        self.active = {}
        self.parents = OrderedDict()
        self.prepared = {}
        self.closed = False

    def _start(self, key, name, kind, event, parent=None, **attributes):
        if key in self.active:
            return
        span = self.tracer.start_span(
            name, context=set_span_in_context(parent) if parent is not None else Context(),
            start_time=int(datetime.fromisoformat(event.timestamp).timestamp() * 1_000_000_000),
            attributes={"openinference.span.kind": kind, "session.id": key[0],
                        "praxis.agent.id": key[1], "praxis.turn.id": key[2],
                        **{k: v for k, v in attributes.items() if v is not None}},
        )
        self.active[key] = Operation(span, int(datetime.fromisoformat(event.timestamp).timestamp() * 1e9),
                                     event.occurred_at)
        return span

    def _span(self, key):
        operation = self.active.get(key)
        return operation.span if operation else self.parents.get(key)

    def _end(self, key, event, outcome="completed", error=""):
        operation = self.active.pop(key, None)
        if operation is None:
            return
        span = operation.span
        span.set_attribute("praxis.outcome", outcome)
        if outcome == "failed":
            span.set_status(Status(StatusCode.ERROR, error))
            span.add_event("exception", {"exception.message": error}, timestamp=operation.time(event))
        elif outcome == "completed":
            span.set_status(Status(StatusCode.OK))
        span.end(end_time=operation.time(event))
        if key[-1] == "turn":
            self.parents[key] = span
            while len(self.parents) > 512:
                self.parents.popitem(last=False)
            for prepared in list(self.prepared):
                if prepared[:3] == key[:3]:
                    self.prepared.pop(prepared)

    def emit_for_session(self, session_id, event):
        if self.closed:
            return
        agent_id, parent_turn = "main", None
        if isinstance(event, ev.ChildAgentEvent):
            agent_id, parent_turn, event = event.agent_id, event.turn_id, event.event
        scope = (session_id, agent_id, event.turn_id)
        turn = (*scope, "turn")
        root = self._span(turn)
        if isinstance(event, ev.TurnStarted):
            parent = self._span((session_id, "main", parent_turn, "turn")) if parent_turn else None
            self._start(turn, f"agent {agent_id}", "AGENT", event, parent,
                        **{"input.value": event.task[:16000]})
        elif isinstance(event, (ev.TurnCompleted, ev.TurnFailed, ev.TurnCancelled)):
            outcome = "failed" if isinstance(event, ev.TurnFailed) else "cancelled" if isinstance(event, ev.TurnCancelled) else "completed"
            # Interrupted tools may not have a terminal event; close outstanding operations honestly.
            for key in list(self.active):
                if key[:3] == scope and key != turn:
                    self._end(key, event, "cancelled" if outcome == "cancelled" else "incomplete")
            self._end(turn, event, outcome, getattr(event, "message", ""))
        elif isinstance(event, ev.ModelRequestEvent):
            key = (*scope, "model", event.request_number)
            if isinstance(event, ev.ModelRequestStarted):
                parent = self._span((*scope, "compression", event.compaction_id)) if event.compaction_id else root
                self._start(key, f"model {event.request_number}", "LLM", event, parent,
                            **{"llm.model_name": event.model, "input.value": encoded(event.input_items),
                               "input.mime_type": "application/json", "praxis.model.key": event.model_key,
                               "praxis.model.requested": event.model,
                               "llm.invocation_parameters": encoded({**event.parameters, "max_output_tokens": event.max_output_tokens}),
                               "praxis.purpose": event.purpose, "praxis.compaction.id": event.compaction_id,
                               "praxis.round": event.round_number, "praxis.chunk": event.chunk_number})
            elif key in self.active:
                op = self.active[key]
                if isinstance(event, ev.AssistantTextDelta):
                    if "praxis.first_text_seconds" not in op.span.attributes:
                        op.span.set_attribute("praxis.first_text_seconds", max(0, event.occurred_at - op.clock))
                elif isinstance(event, ev.ModelRequestCompleted):
                    response = event.response
                    if response is not None:
                        op.span.set_attribute("output.value", encoded(response.output_items))
                        op.span.set_attribute("output.mime_type", "application/json")
                        if response.model:
                            op.span.set_attribute("llm.model_name", response.model)
                        if response.usage:
                            for name, value in (("prompt", response.usage.input_tokens),
                                                ("completion", response.usage.output_tokens),
                                                ("total", response.usage.total_tokens)):
                                op.span.set_attribute("llm.token_count." + name, value)
                    self._end(key, event)
                elif isinstance(event, ev.ModelRequestFailed):
                    self._end(key, event, "failed", event.message)
                elif isinstance(event, ev.ModelRequestCancelled):
                    self._end(key, event, "cancelled")
        elif isinstance(event, ev.ContextCompaction):
            key = (*scope, "compression", event.compaction_id)
            if event.stage == "started":
                self._start(key, "context compression", "CHAIN", event, root,
                            **{"praxis.compaction.id": event.compaction_id, "praxis.reason": event.reason,
                               "praxis.tokens.before": event.before_tokens})
            elif key in self.active:
                op = self.active[key]
                op.span.add_event(event.stage, {"round": event.round_number, "chunk": event.chunk_number},
                                  timestamp=op.time(event))
                if event.stage in {"completed", "failed", "cancelled"}:
                    if event.after_tokens is not None:
                        op.span.set_attribute("praxis.tokens.after", event.after_tokens)
                    self._end(key, event, event.stage, event.error)
        elif isinstance(event, ev.ToolCallPrepared):
            self.prepared[(*scope, event.call_id)] = encoded(event.arguments)
        elif isinstance(event, ev.ToolStarted):
            self._start((*scope, "tool", event.call_id), event.tool_name, "TOOL", event, root,
                        **{"tool.name": event.tool_name, "tool.id": event.call_id,
                           "input.value": self.prepared.pop((*scope, event.call_id), "{}"),
                           "input.mime_type": "application/json"})
        elif isinstance(event, ev.ToolFinished):
            key = (*scope, "tool", event.call_id)
            if key in self.active:
                self.active[key].span.set_attribute("output.value", str(event.result.output)[:16000])
                self._end(key, event, "completed" if event.result.success else "failed",
                          "" if event.result.success else str(event.result.output)[:1000])
        elif isinstance(event, ev.ApprovalRequested):
            self._start((*scope, "approval", event.call_id), "approval " + event.tool_name, "CHAIN", event, root)
        elif isinstance(event, ev.ApprovalResolved):
            self._end((*scope, "approval", event.call_id), event, "completed" if event.approved else "rejected")
        elif isinstance(event, ev.AssistantTextCompleted):
            if root:
                root.set_attribute("output.value", event.text[:16000])
        elif root is not None and not isinstance(event, ev.ToolOutputDelta):
            root.add_event(type(event).__name__, {"detail": encoded({
                name: getattr(event, name) for name in ("status", "agent_id", "content", "source")
                if hasattr(event, name)
            })}, timestamp=int(datetime.fromisoformat(event.timestamp).timestamp() * 1e9))

    def close(self):
        if self.closed:
            return
        self.closed = True
        for operation in self.active.values():
            operation.span.set_attribute("praxis.outcome", "incomplete")
            operation.span.end(end_time=operation.wall + max(0, int((monotonic() - operation.clock) * 1e9)))
        self.active.clear()
        self.prepared.clear()
        self.parents.clear()
        self.provider.force_flush(timeout_millis=5000)
        self.provider.shutdown()
