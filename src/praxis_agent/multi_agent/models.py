from __future__ import annotations

import json
import os
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from praxis_agent.agent import Agent
    from praxis_agent.context.capacity import ContextStatus


class AgentStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    IDLE = "idle"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class AgentMessage:
    sender_id: str
    recipient_id: str
    content: str
    message_id: str = field(default_factory=lambda: uuid4().hex)
    task_summary: str = ""

    def to_dict(self) -> dict[str, str]:
        result = {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "content": self.content,
        }
        if self.task_summary:
            result["task_summary"] = self.task_summary
        return result


def message_text(messages: list[AgentMessage]) -> str:
    return "[Agent collaboration messages; source-labelled task data]\n" + json.dumps(
        [message.to_dict() for message in messages], ensure_ascii=False,
    )


class ExecutionStopped(RuntimeError):
    """A worker was cancelled."""


@dataclass(slots=True)
class ExecutionControl:
    cancelled: Event = field(default_factory=Event)

    def check(self) -> None:
        if self.cancelled.is_set():
            raise ExecutionStopped("worker cancelled")


@dataclass(frozen=True, slots=True)
class MultiAgentConfig:
    max_workers: int = 8

    @classmethod
    def from_env(cls) -> MultiAgentConfig:
        max_workers = int(os.environ.get("PRAXIS_AGENT_MAX_WORKERS", "8"))
        if max_workers < 1:
            raise ValueError("PRAXIS_AGENT_MAX_WORKERS must be positive")
        return cls(max_workers=max_workers)


@dataclass(slots=True)
class AgentHandle:
    agent_id: str
    parent_turn_id: str
    control: ExecutionControl
    agent: Agent | None = None
    status: AgentStatus = AgentStatus.IDLE
    inbox: deque[AgentMessage] = field(default_factory=deque)
    future: Future[None] | None = None
    task_summary: str = ""
    context_status: ContextStatus | None = None
    compaction_count: int = 0
    started_at: float | None = None
    elapsed_seconds: float = 0.0
    mode: str = "read_only"
    workspace: Path | None = None
