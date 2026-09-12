from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class SessionStoreError(RuntimeError):
    """表示会话数据无法创建、读取或更新。"""


class SessionNotFoundError(SessionStoreError):
    """表示指定的持久化会话不存在。"""


class SessionWorkspaceMismatchError(SessionStoreError):
    """表示会话所属工作区与当前工作区不一致。"""


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """保存一条可恢复会话的元数据。"""

    id: str
    workspace: Path
    model: str
    title: str
    created_at: datetime
    updated_at: datetime
