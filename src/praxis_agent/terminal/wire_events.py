"""Convert the public display protocol to terminal presentation events."""

from dataclasses import fields, is_dataclass

from praxis_agent import events
from praxis_agent.context.capacity import ContextPressure, ContextStatus
from praxis_agent.model import ModelResponse
from praxis_agent.planning import PlanStep, PlanStepStatus
from praxis_agent.tools import ToolResult
from praxis_agent.tools.filesystem import FileChange


def decode_event(frame):
    kind = frame["event"]
    cls = getattr(events, kind, None)
    if cls is None or not is_dataclass(cls) or not issubclass(cls, events.AgentEvent):
        return None
    data = dict(frame["data"])
    turn_id = data.pop("child_turn_id", None) or frame.get("turn_id", "")
    if kind == "ModelRequestStarted" and data.get("context_status"):
        status = data["context_status"]
        data["context_status"] = ContextStatus(
            **{**status, "pressure": ContextPressure(status["pressure"])}
        )
    elif kind == "ModelRequestCompleted":
        data["response"] = ModelResponse((), data.pop("output_text", ""), ())
    elif kind == "ToolFinished":
        result = dict(data["result"])
        if result.get("file_change"):
            result["file_change"] = FileChange(**result["file_change"])
        data["result"] = ToolResult(**result)
    elif kind == "PlanUpdated":
        data["plan"] = tuple(
            PlanStep(item["description"], PlanStepStatus(item["status"]))
            for item in data["plan"]
        )
    if "agent_id" in {field.name for field in fields(cls)}:
        data["agent_id"] = frame.get("agent_id", "main")
    allowed = {field.name for field in fields(cls)} - {"turn_id", "occurred_at"}
    event = cls(
        turn_id, **{key: value for key, value in data.items() if key in allowed}
    )
    agent_id = frame.get("agent_id", "main")
    if agent_id != "main" and kind not in {
        "ChildAgentStatus",
        "ChildAgentMessage",
        "ToolOutputDelta",
    }:
        event = events.ChildAgentEvent(frame.get("turn_id", ""), agent_id, event)
    return event
