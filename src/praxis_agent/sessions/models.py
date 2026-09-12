from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from praxis_agent.planning.state import RunStatus


class SessionStoreError(RuntimeError):
    """表示会话数据无法创建、读取或更新。"""


class SessionNotFoundError(SessionStoreError):
    """表示指定的持久化会话不存在。"""


class SessionWorkspaceMismatchError(SessionStoreError):
    """表示会话所属工作区与当前工作区不一致。"""


class AmbiguousSessionIdError(SessionStoreError):
    """表示一个 Session ID 前缀匹配到多个会话。"""


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """保存一条可恢复会话的元数据。"""

    id: str
    workspace: Path
    model: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """保存会话列表展示所需的最小信息。"""

    id: str
    title: str
    updated_at: datetime
    latest_run_status: RunStatus | None
