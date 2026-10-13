import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import monotonic

from praxis_agent.tools.processes import CommandProcess


_DEFAULT_TIMEOUT_SECONDS = 60
_MAX_TIMEOUT_SECONDS = 120


def _resolve_command_argv(argv: list[str]) -> list[str]:
    executable = Path(argv[0])
    if os.name == "nt" and not executable.drive and executable.name == argv[0]:
        # CreateProcess searches the parent executable's directory before PATH.
        # A virtualenv Python may therefore launch its base interpreter even when
        # PATH selects another environment. Resolve bare names before spawning;
        # explicit executable paths keep their original meaning.
        resolved = shutil.which(argv[0])
        if resolved is not None:
            return [str(Path(resolved).resolve()), *argv[1:]]
    return list(argv)


def _resolve_working_directory(
    workspace: Path,
    relative_path: str,
) -> Path:
    """解析命令工作目录，并确保它位于工作区内。"""
    workspace_root = workspace.resolve()
    if not workspace_root.is_dir():
        raise NotADirectoryError(
            f"workspace is not a directory: {workspace}"
        )

    requested_path = Path(relative_path)
    if requested_path.is_absolute():
        raise ValueError("working directory must be relative to the workspace")

    working_directory = (workspace_root / requested_path).resolve()
    if not working_directory.is_relative_to(workspace_root):
        raise ValueError("working directory must stay inside the workspace")
    if not working_directory.is_dir():
        raise NotADirectoryError(
            f"working directory does not exist: {relative_path}"
        )
    return working_directory


def run_command(
    workspace: Path,
    argv: list[str],
    relative_path: str = ".",
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    *,
    check_execution: Callable[[], None] | None = None,
) -> str:
    """在工作区内运行非交互式命令，并返回退出码和输出。"""
    if not isinstance(argv, list) or not argv:
        raise ValueError("argv must be a non-empty list")
    if any(not isinstance(argument, str) or not argument for argument in argv):
        raise ValueError("every argv item must be a non-empty string")
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds,
        int,
    ):
        raise TypeError("timeout_seconds must be an integer")
    if not 1 <= timeout_seconds <= _MAX_TIMEOUT_SECONDS:
        raise ValueError(
            f"timeout_seconds must be between 1 and {_MAX_TIMEOUT_SECONDS}"
        )

    working_directory = _resolve_working_directory(workspace, relative_path)

    if check_execution is not None:
        check_execution()
    # Disk-backed output also avoids hanging on inherited pipes after the direct
    # child exits. Commands are foreground jobs; detached descendants are cleaned up.
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        command = CommandProcess(_resolve_command_argv(argv), cwd=working_directory, stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=err, shell=False)
        try:
            deadline = monotonic() + timeout_seconds
            while command.process.poll() is None:
                if check_execution is not None:
                    check_execution()
                if monotonic() >= deadline:
                    raise TimeoutError(f"command timed out after {timeout_seconds} seconds")
                try:
                    command.process.wait(timeout=min(0.1, max(0.001, deadline - monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            returncode = command.process.returncode
        finally:
            command.close()
        out.seek(0)
        err.seek(0)
        stdout = out.read().decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        stderr = err.read().decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(
        (
            f"exit_code: {returncode}",
            f"stdout:\n{stdout or '<empty>'}",
            f"stderr:\n{stderr or '<empty>'}",
        )
    )
