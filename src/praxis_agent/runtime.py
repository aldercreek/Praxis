import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# 只探测模型可能实际使用的常见 shell，避免把整个 PATH 暴露给模型。
_SHELL_CANDIDATES = ("cmd", "powershell", "pwsh", "bash", "sh")


@dataclass(frozen=True, slots=True)
class CommandShell:
    name: str
    path: str

    def argv(self, command: str) -> list[str]:
        if self.name in {"pwsh", "powershell"}:
            return [self.path, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command]
        return [self.path, "-c", command]


def select_command_shell() -> CommandShell | None:
    names = ("pwsh", "powershell") if sys.platform == "win32" else ("bash", "sh")
    for name in names:
        path = shutil.which(name)
        if path is not None:
            return CommandShell(name, path)
    return None


def _shell_version(shell: CommandShell | None) -> str:
    if shell is None:
        return "Not detected"
    command = "$PSVersionTable.PSVersion.ToString()" if shell.name in {"pwsh", "powershell"} else 'printf "%s" "${BASH_VERSION:-unknown}"'
    try:
        result = subprocess.run(
            shell.argv(command), stdin=subprocess.DEVNULL, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "Unknown version"
    except (OSError, subprocess.TimeoutExpired):
        return "Unknown version"


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """保存 Praxis 启动时采集且会话期间保持不变的运行环境信息。"""

    operating_system: str
    sys_platform: str
    workspace_root: Path
    available_shells: tuple[str, ...]
    unavailable_shells: tuple[str, ...]
    command_shell: CommandShell | None
    shell_version: str


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
    command_shell = select_command_shell()
    return RuntimeContext(
        operating_system=platform.system() or "Unknown",
        sys_platform=sys.platform,
        workspace_root=workspace_root,
        available_shells=available_shells,
        unavailable_shells=unavailable_shells,
        command_shell=command_shell,
        shell_version=_shell_version(command_shell),
    )
