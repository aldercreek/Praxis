from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from praxis_agent.events import (
    AgentEvent,
    ChildAgentEvent,
    ModelRequestCompleted,
    ModelRequestStarted,
    ToolFinished,
)

if TYPE_CHECKING:
    from .runtime import SessionRuntime


def json_value(value):
    """Only normalize explicitly selected display data, never arbitrary model objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            field.name: json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return str(value)


def display_event(event: AgentEvent) -> tuple[str, str, str, dict]:
    turn_id, agent_id = event.turn_id, "main"
    child_turn = None
    if isinstance(event, ChildAgentEvent):
        agent_id, event = event.agent_id, event.event
        child_turn = event.turn_id
    elif getattr(event, "agent_id", ""):
        agent_id = event.agent_id
    if isinstance(event, ModelRequestStarted):
        data = {
            "request_number": event.request_number,
            "context_status": json_value(event.context_status),
        }
    elif isinstance(event, ModelRequestCompleted):
        response = event.response
        data = {
            "request_number": event.request_number,
            "output_text": response.output_text if response else "",
            "model": response.model if response else None,
            "usage": json_value(response.usage) if response else None,
        }
    elif isinstance(event, ToolFinished):
        # UI consumes summaries, not model attachments or raw programmatic MCP data.
        data = {
            "call_id": event.call_id,
            "tool_name": event.tool_name,
            "result": {
                "success": event.result.success,
                "output": event.result.output,
                "structured_output": json_value(event.result.structured_output)
                if event.tool_name in {"run_command", "wait_command"}
                else None,
                "file_change": json_value(event.result.file_change),
            },
        }
    else:
        data = {
            field.name: json_value(getattr(event, field.name))
            for field in fields(event)
            if field.name not in {"turn_id", "occurred_at", "agent_id"}
        }
    if child_turn is not None:
        data["child_turn_id"] = child_turn
    return type(event).__name__, turn_id, agent_id, data


class ServiceEventSink:
    def __init__(self, runtime: SessionRuntime):
        self.runtime = runtime

    def emit(self, event: AgentEvent) -> None:
        self.runtime.publish(*display_event(event))

    emit_live = emit

    def emit_deferred(self, event: AgentEvent) -> None:
        # Child events were already delivered by emit_live on the producing thread.
        pass
