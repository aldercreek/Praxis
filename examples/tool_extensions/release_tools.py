"""Offline example: no network, credentials, or company API required."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from praxis_agent.sdk import tool


@tool(description="Greet a person in English or Chinese", requires_approval=False, parallel_safe=True)
def greet(name: str, *, language: Literal["en", "zh"] = "en") -> str:
    return f"你好，{name}！" if language == "zh" else f"Hello, {name}!"


@dataclass
class ReleaseQuery:
    release_id: str
    include_logs: bool = False


@dataclass
class ReleaseStatus:
    release_id: str
    status: str
    logs: list[str] = field(default_factory=list)


@tool(description="Look up a simulated release status by release ID", requires_approval=True)
def get_release_status(query: ReleaseQuery) -> ReleaseStatus:
    return ReleaseStatus(query.release_id, "success", ["Build passed", "Deploy completed"] if query.include_logs else [])
