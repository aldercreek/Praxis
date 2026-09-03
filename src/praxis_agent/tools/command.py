import subprocess
from pathlib import Path


_DEFAULT_TIMEOUT_SECONDS = 60
_MAX_TIMEOUT_SECONDS = 120
_MAX_OUTPUT_CHARS = 10_000


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


def _truncate_output(output: str) -> str:
    """限制单个输出流的长度，避免一次结果占用过多模型上下文。"""
    if len(output) <= _MAX_OUTPUT_CHARS:
        return output
    removed_chars = len(output) - _MAX_OUTPUT_CHARS
    return (
        output[:_MAX_OUTPUT_CHARS]
        + f"\n... truncated {removed_chars} characters"
    )


def run_command(
    workspace: Path,
    argv: list[str],
    relative_path: str = ".",
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
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

    try:
        completed = subprocess.run(
            argv,
            cwd=working_directory,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"command timed out after {timeout_seconds} seconds"
        ) from error

    stdout = _truncate_output(completed.stdout)
    stderr = _truncate_output(completed.stderr)
    return "\n".join(
        (
            f"exit_code: {completed.returncode}",
            f"stdout:\n{stdout or '<empty>'}",
            f"stderr:\n{stderr or '<empty>'}",
        )
    )
