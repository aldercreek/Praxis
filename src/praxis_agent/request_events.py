"""Per-turn model events and explicit compression correlation, without exporters."""
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from uuid import uuid4

from praxis_agent.events import (
    AgentEvent, AssistantTextDelta, ContextCompaction, ModelRequestCancelled,
    ModelRequestCompleted, ModelRequestFailed, ModelRequestStarted,
)
from praxis_agent.execution import ExecutionStopped, run_operation


@dataclass
class _Requests:
    turn_id: str
    emit: Callable[[AgentEvent], None]
    number: int = 0
    lock: Lock = field(default_factory=Lock)

    def next_number(self):
        with self.lock:
            self.number += 1
            return self.number


_requests = ContextVar("model_request_events", default=None)
_compaction = ContextVar("compaction_events", default=None)


@contextmanager
def model_request_events(turn_id, emit):
    token = _requests.set(_Requests(turn_id, emit))
    compaction_token = _compaction.set(None)
    try:
        yield
    finally:
        _compaction.reset(compaction_token)
        _requests.reset(token)


def request_model(client, items, tools, *, config=None, context_status=None,
                  purpose="agent", max_output_tokens=None, validate=None):
    """One logical client invocation, including SDK retries, has one terminal event."""
    requests = _requests.get()
    compaction = _compaction.get() if purpose == "compression" else None
    identity = dict(
        turn_id=requests.turn_id if requests else "",
        request_number=requests.next_number() if requests else 0,
        purpose=purpose,
        compaction_id=compaction.id if compaction else None,
        round_number=compaction.round_number if compaction else 0,
        chunk_number=compaction.chunk_number if compaction else 0,
    )
    limit = config.model_limits.max_output_tokens if config else None
    if max_output_tokens is not None:
        limit = min(limit, max_output_tokens) if limit is not None else max_output_tokens
    if requests:
        requests.emit(ModelRequestStarted(
            **identity, input_items=tuple(items), context_status=context_status,
            model_key=config.model_key if config else None,
            model=config.model if config else None,
            protocol=config.protocol if config else None,
            parameters=dict(config.parameters) if config else {}, max_output_tokens=limit,
        ))
    options = {}
    if max_output_tokens is not None:
        options["max_output_tokens"] = max_output_tokens
    if purpose == "agent":
        options["on_text_delta"] = (
            lambda text: requests.emit(AssistantTextDelta(**identity, text=text))
        ) if requests else None
    try:
        response = run_operation(
            "summary model" if purpose == "compression" else "model",
            lambda handle: client.create_response(items, tools=tools, **options),
        )
        if validate is not None:
            validate(response)
    except (ExecutionStopped, KeyboardInterrupt):
        if requests:
            requests.emit(ModelRequestCancelled(**identity))
        raise
    except Exception as error:
        if requests:
            requests.emit(ModelRequestFailed(
                **identity, error_type=type(error).__name__, message=str(error),
            ))
        raise
    if requests:
        requests.emit(ModelRequestCompleted(**identity, response=response))
    return response


class CompactionEvents:
    def __init__(self, turn_id, emit, before_tokens, reason):
        self.id = uuid4().hex
        self.turn_id, self.emit = turn_id, emit
        self.before_tokens, self.reason = before_tokens, reason
        self.started = monotonic()
        self.request_count = self.round_number = self.chunk_number = 0
        self.after_tokens = None
        self._rounds = 0

    def next_round(self):
        self._rounds += 1
        return self._rounds

    def event(self, stage, **values):
        self.emit(ContextCompaction(
            self.turn_id, stage, self.before_tokens, compaction_id=self.id,
            reason=self.reason, request_count=self.request_count,
            elapsed_seconds=monotonic() - self.started, **values,
        ))

    def progress(self, round_number, chunk_number, chunk_count, response):
        self.round_number, self.chunk_number = round_number, chunk_number
        if response is None:
            self.request_count += 1
        usage = response.usage if response is not None else None
        self.event(
            "chunk_started" if response is None else "chunk_completed",
            round_number=round_number, chunk_number=chunk_number, chunk_count=chunk_count,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            model=response.model if response else None,
        )


def next_summary_round(attempt):
    compaction = _compaction.get()
    return compaction.next_round() if compaction else attempt + 1


@contextmanager
def compaction_events(turn_id, sink, before_tokens, reason):
    emit = sink.emit if sink is not None else lambda event: None
    requests_token = None
    if _requests.get() is None:
        requests_token = _requests.set(_Requests(turn_id, emit))
    progress = CompactionEvents(turn_id, emit, before_tokens, reason)
    token = _compaction.set(progress)
    try:
        progress.event("started")
        try:
            yield progress
        except (ExecutionStopped, KeyboardInterrupt):
            progress.event("cancelled")
            raise
        except Exception as error:
            progress.event("failed", error=str(error))
            raise
        else:
            progress.event("completed", after_tokens=progress.after_tokens)
    finally:
        _compaction.reset(token)
        if requests_token is not None:
            _requests.reset(requests_token)
