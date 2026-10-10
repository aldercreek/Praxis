from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from praxis_agent.model import ModelResponse
    from praxis_agent.planning.plan import PlanStep
    from praxis_agent.tools import ToolResult


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """所有 Agent 事件都携带当前任务的 turn_id。"""

    turn_id: str


@dataclass(frozen=True, slots=True)
class TurnStarted(AgentEvent):
    """表示 Agent 已经开始处理一条用户任务。"""

    task: str


@dataclass(frozen=True, slots=True)
class ContextCompaction(AgentEvent):
    """应急压缩及其分块请求的进度；容量为估算值，usage 为服务端值。"""

    stage: str
    before_tokens: int
    reason: str = "context_overflow"
    after_tokens: int | None = None
    round_number: int = 0
    chunk_number: int = 0
    chunk_count: int = 0
    request_count: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None


@dataclass(frozen=True, slots=True)
class ModelRequestStarted(AgentEvent):
    """表示一次模型请求已经开始。"""

    request_number: int
    input_items: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class AssistantTextDelta(AgentEvent):
    """表示模型流式返回的一段助手文本。"""

    request_number: int
    text: str


@dataclass(frozen=True, slots=True)
class ModelRequestCompleted(AgentEvent):
    """表示一次模型响应已经完整接收并解析。"""

    request_number: int
    response: ModelResponse | None = None


@dataclass(frozen=True, slots=True)
class ToolCallPrepared(AgentEvent):
    """表示模型的工具调用已经解析完成，正在等待审批或执行。"""

    call_id: str
    tool_name: str
    arguments: Mapping[str, object]
    requires_approval: bool


@dataclass(frozen=True, slots=True)
class ApprovalRequested(AgentEvent):
    """表示一个工具调用正在等待用户审批。"""

    call_id: str
    tool_name: str


@dataclass(frozen=True, slots=True)
class ApprovalResolved(AgentEvent):
    """表示用户已经同意或拒绝工具调用。"""

    call_id: str
    tool_name: str
    approved: bool


@dataclass(frozen=True, slots=True)
class ToolStarted(AgentEvent):
    """表示工具已经真正开始执行。"""

    call_id: str
    tool_name: str


@dataclass(frozen=True, slots=True)
class ToolFinished(AgentEvent):
    """表示工具执行已经结束，并携带最终结果。"""

    call_id: str
    tool_name: str
    result: ToolResult


@dataclass(frozen=True, slots=True)
class PlanUpdated(AgentEvent):
    """表示当前任务计划已经成功更新。"""

    plan: tuple[PlanStep, ...]


@dataclass(frozen=True, slots=True)
class AssistantTextCompleted(AgentEvent):
    """表示模型已经生成本轮最终回答。"""

    text: str


@dataclass(frozen=True, slots=True)
class TurnCompleted(AgentEvent):
    """表示当前任务已经正常完成。"""


@dataclass(frozen=True, slots=True)
class TurnFailed(AgentEvent):
    """表示当前任务因为异常而结束。"""

    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class ChildAgentEvent(AgentEvent):
    """A worker event routed on the parent thread, isolated from its transcript."""

    agent_id: str
    event: AgentEvent


@dataclass(frozen=True, slots=True)
class ChildAgentStatus(AgentEvent):
    agent_id: str
    status: str


@dataclass(frozen=True, slots=True)
class ChildAgentMessage(AgentEvent):
    agent_id: str
    content: str


class EventSink(Protocol):
    """接收 Agent 事件；终端、日志等组件都可以实现这个接口。"""

    def emit(self, event: AgentEvent) -> None:
        """处理一个已经发生的事件。"""
        ...


@dataclass(frozen=True, slots=True)
class CompositeEventSink:
    """把同一个事件依次发送给多个消费者。"""

    sinks: tuple[EventSink, ...]

    def emit(self, event: AgentEvent) -> None:
        for sink in self.sinks:
            sink.emit(event)


__all__ = [
    "AgentEvent",
    "ApprovalRequested",
    "ApprovalResolved",
    "AssistantTextDelta",
    "AssistantTextCompleted",
    "CompositeEventSink",
    "ChildAgentEvent",
    "ChildAgentStatus",
    "ChildAgentMessage",
    "ContextCompaction",
    "EventSink",
    "ModelRequestCompleted",
    "ModelRequestStarted",
    "PlanUpdated",
    "ToolCallPrepared",
    "ToolFinished",
    "ToolStarted",
    "TurnCompleted",
    "TurnFailed",
    "TurnStarted",
]
