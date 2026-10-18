from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath

from praxis_agent.planning.plan import PlanStep


class RunStatus(StrEnum):
    """表示一次 Agent 任务的整体运行状态。"""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class RunState:
    """保存一次任务中需要跨模型请求保留的最小状态。"""

    original_task: str
    plan: list[PlanStep] = field(default_factory=list)
    changed_files: set[str] = field(default_factory=set)
    status: RunStatus = RunStatus.RUNNING
    error: str | None = None

    def record_changed_file(self, relative_path: str) -> None:
        """记录一个已经被文件工具成功修改的工作区相对路径。"""
        normalized_path = PurePosixPath(relative_path.replace("\\", "/"))
        self.changed_files.add(normalized_path.as_posix())

    def complete(self) -> None:
        """把当前任务标记为正常完成。"""
        self.status = RunStatus.COMPLETED
        self.error = None

    def fail(self, error: Exception) -> None:
        """把当前任务标记为失败并保存简短原因。"""
        self.status = RunStatus.FAILED
        self.error = f"{type(error).__name__}: {error}"

    def cancel(self) -> None:
        self.status = RunStatus.CANCELLED
        self.error = "Stopped by user"
