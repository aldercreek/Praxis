import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


# 只探测模型可能实际使用的常见 shell，避免把整个 PATH 暴露给模型。
_SHELL_CANDIDATES = ("cmd", "powershell", "pwsh", "bash", "sh")


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """保存 Praxis 启动时采集且会话期间保持不变的运行环境信息。"""

    operating_system: str
    sys_platform: str
    workspace_root: Path
    available_shells: tuple[str, ...]
    unavailable_shells: tuple[str, ...]


def collect_runtime_context(workspace: Path) -> RuntimeContext:
    """采集允许提供给模型的最小运行环境信息。"""
    workspace_root = workspace.resolve()
    if not workspace_root.is_dir():
        raise NotADirectoryError(
            f"workspace is not a directory: {workspace}"
        )

    available_shells = tuple(
        name for name in _SHELL_CANDIDATES if shutil.which(name) is not None
    )
    unavailable_shells = tuple(
        name for name in _SHELL_CANDIDATES if name not in available_shells
    )
    return RuntimeContext(
        operating_system=platform.system() or "Unknown",
        sys_platform=sys.platform,
        workspace_root=workspace_root,
        available_shells=available_shells,
        unavailable_shells=unavailable_shells,
    )
