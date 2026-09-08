from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from types import TracebackType
from typing import Any

from praxis_agent.config import LangfuseConfig
from praxis_agent.events import (
    AssistantTextCompleted,
    ModelRequestCompleted,
    ModelRequestStarted,
    ToolCallPrepared,
    ToolFinished,
    ToolStarted,
    TurnCompleted,
    TurnStarted,
)
from praxis_agent.model import ModelResponse, TokenUsage
from praxis_agent.observability import LangfuseEventSink
from praxis_agent.tools import ToolResult


class FakeObservation:
    def __init__(self, arguments: dict[str, object]) -> None:
        self.arguments = arguments
        self.children: list[FakeObservation] = []
        self.updates: list[dict[str, object]] = []
        self.trace_io: list[dict[str, object]] = []
        self.ended = False

    def start_observation(self, **arguments: object) -> "FakeObservation":
        child = FakeObservation(dict(arguments))
        self.children.append(child)
        return child

    def update(self, **values: object) -> object:
        self.updates.append(dict(values))
        return self

    def end(self, *, end_time: int | None = None) -> object:
        del end_time
        self.ended = True
        return self

    def set_trace_io(self, **values: object) -> object:
        self.trace_io.append(dict(values))
        return self


class FakeObservationContext(AbstractContextManager[FakeObservation]):
    def __init__(self, observation: FakeObservation) -> None:
        self.observation = observation

    def __enter__(self) -> FakeObservation:
        return self.observation

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        del exc_type, exc_value, traceback
        self.observation.ended = True
        return None


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.roots: list[FakeObservation] = []
        self.flush_count = 0

    def start_as_current_observation(
        self,
        **arguments: object,
    ) -> FakeObservationContext:
        root = FakeObservation(dict(arguments))
        self.roots.append(root)
        return FakeObservationContext(root)

    def flush(self) -> None:
        self.flush_count += 1


def _ignore_arguments(arguments: object) -> AbstractContextManager[None]:
    del arguments
    return nullcontext()


def _sink(
    tmp_path: Path,
    monkeypatch: Any,
) -> tuple[LangfuseEventSink, FakeLangfuseClient]:
    client = FakeLangfuseClient()
    monkeypatch.setattr(
        "praxis_agent.observability.propagate_attributes",
        lambda **arguments: _ignore_arguments(arguments),
    )
    sink = LangfuseEventSink(
        LangfuseConfig("public-key", "secret-key"),
        tmp_path,
        client=client,
        session_id="session-1",
    )
    return sink, client


def test_langfuse_sink_builds_agent_generation_and_tool_trace(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    sink, client = _sink(tmp_path, monkeypatch)
    turn_id = "0123456789abcdef0123456789abcdef"
    response = ModelResponse(
        output_items=(),
        output_text="",
        function_calls=(),
        response_id="response-1",
        model="example-model",
        usage=TokenUsage(10, 4, 14),
    )

    sink.emit(TurnStarted(turn_id, "inspect repository"))
    sink.emit(
        ModelRequestStarted(
            turn_id,
            1,
            ({"role": "user", "content": "inspect repository"},),
        )
    )
    sink.emit(ModelRequestCompleted(turn_id, 1, response))
    sink.emit(
        ToolCallPrepared(
            turn_id,
            "call-1",
            "read_file",
            {"relative_path": "README.md", "api_key": "do-not-send"},
            False,
        )
    )
    sink.emit(ToolStarted(turn_id, "call-1", "read_file"))
    sink.emit(
        ToolFinished(
            turn_id,
            "call-1",
            "read_file",
            ToolResult(True, "README content"),
        )
    )
    sink.emit(AssistantTextCompleted(turn_id, "Repository inspected."))
    sink.emit(TurnCompleted(turn_id))
    sink.close()

    root = client.roots[0]
    assert root.arguments["trace_context"] == {"trace_id": turn_id}
    assert root.arguments["as_type"] == "agent"
    assert root.ended is True
    assert root.trace_io[-1] == {
        "output": {"text": "Repository inspected."}
    }

    generation, tool = root.children
    assert generation.arguments["as_type"] == "generation"
    assert generation.updates[0]["usage_details"] == {
        "input": 10,
        "output": 4,
        "total": 14,
    }
    assert generation.ended is True
    assert tool.arguments["as_type"] == "tool"
    assert tool.arguments["input"]["api_key"] == "do-not-send"
    assert tool.ended is True
    assert client.flush_count == 1


def test_langfuse_sink_records_declined_tool_without_execution(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    sink, client = _sink(tmp_path, monkeypatch)
    turn_id = "fedcba9876543210fedcba9876543210"

    sink.emit(TurnStarted(turn_id, "run command"))
    sink.emit(
        ToolCallPrepared(
            turn_id,
            "call-1",
            "run_command",
            {"argv": ["cmd", "/c", "dir"]},
            True,
        )
    )
    sink.emit(
        ToolFinished(
            turn_id,
            "call-1",
            "run_command",
            ToolResult(False, "tool execution was declined by the user"),
        )
    )
    sink.emit(TurnCompleted(turn_id))

    tool = client.roots[0].children[0]
    assert tool.arguments["metadata"]["executed"] is False
    assert tool.updates[0]["level"] == "ERROR"
    assert tool.ended is True
