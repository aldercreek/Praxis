import codecs
import locale
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from praxis_agent.runtime import CommandShell, select_command_shell



class _OutputDecoder:
    """Keep split UTF-8 characters intact; Windows tools may emit the local code page."""

    def __init__(self) -> None:
        self.decoder = codecs.getincrementaldecoder("utf-8")()

    def decode(self, chunk: bytes, *, final: bool = False) -> str:
        pending, _ = self.decoder.getstate()
        try:
            return self.decoder.decode(chunk, final=final)
        except UnicodeDecodeError:
            encoding = locale.getencoding() if os.name == "nt" else "utf-8"
            self.decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
            return self.decoder.decode(pending + chunk, final=final)


def _open_output_reader(path: str):
    # On Windows the writer uses delete-on-close. Reopening must share delete
    # access too; cleanup then remains safe while a killed descendant is exiting.
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_TEMPORARY", 0)
    return os.fdopen(os.open(path, flags), "rb", buffering=0)


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
    workdir: str,
) -> Path:
    """接受绝对目录；相对目录以 Agent 工作目录为基准解析。"""
    if not isinstance(workdir, str) or not workdir:
        raise ValueError("workdir must be a non-empty string")
    working_directory = (workspace / workdir).resolve()
    if not working_directory.is_dir():
        raise NotADirectoryError(
            f"working directory does not exist or is not a directory: {working_directory}"
        )
    return working_directory


def run_command(
    workspace: Path,
    argv: list[str] | None = None,
    workdir: str = ".",
    yield_time_ms: int = 10000,
    *,
    command: str | None = None,
    command_shell: CommandShell | None = None,
    check_execution: Callable[[], None] | None = None,
    on_output: Callable[[str, str], None] | None = None,
    manager=None,
    defer_ack: bool = False,
) -> dict[str, object]:
    """Start once; yielding returns control without stopping the command."""
    from praxis_agent.tools.command_sessions import default_manager, validate_yield

    validate_yield(yield_time_ms)
    if (argv is None) == (command is None):
        raise ValueError("Provide exactly one of argv or command; the command was not executed.")
    if argv is not None:
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
            raise ValueError("argv must be a non-empty list of strings")
        if not argv[0]:
            raise ValueError("argv[0] must be a non-empty executable name")
    elif not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    working_directory = _resolve_working_directory(workspace, workdir)
    if check_execution is not None:
        check_execution()
    manager = manager if manager is not None else default_manager()
    try:
        if command is not None:
            selected_shell = command_shell or select_command_shell()
            if selected_shell is None:
                raise FileNotFoundError("No command shell detected")
            argv = selected_shell.argv(command)
        session_id = manager.start(_resolve_command_argv(argv), working_directory)
    except OSError as error:
        reason = ("Executable or command shell not found. Check the command name, installation, or executable path."
                  if isinstance(error, FileNotFoundError) else str(error))
        return {"session_id": None, "status": "spawn_failed", "exit_code": None,
                "stdout": "", "stderr": "", "error": f"Unable to start command: {reason}"}
    return manager.wait(session_id, yield_time_ms, check_execution=check_execution,
                        on_output=on_output, defer_ack=defer_ack)


def wait_command(session_id: str, yield_time_ms: int = 10000, *, manager=None,
                 check_execution=None, on_output=None, defer_ack: bool = False):
    """Wait for an existing command and return only its unconsumed output."""
    from praxis_agent.tools.command_sessions import default_manager

    manager = manager if manager is not None else default_manager()
    return manager.wait(session_id, yield_time_ms, check_execution=check_execution,
                        on_output=on_output, defer_ack=defer_ack)
