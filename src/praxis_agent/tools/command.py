import codecs
import locale
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import monotonic

from praxis_agent.execution import current_handle
from praxis_agent.tools.processes import CommandProcess

_DEFAULT_TIMEOUT_SECONDS = 60
_MAX_TIMEOUT_SECONDS = 120


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
    argv: list[str],
    workdir: str = ".",
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    *,
    check_execution: Callable[[], None] | None = None,
    on_output: Callable[[str, str], None] | None = None,
) -> str:
    """在指定目录运行非交互式命令，并返回退出码和输出。"""
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

    working_directory = _resolve_working_directory(workspace, workdir)

    if check_execution is not None:
        check_execution()
    # Disk-backed output also avoids hanging on inherited pipes after the direct
    # child exits. Commands are foreground jobs; detached descendants are cleaned up.
    with tempfile.NamedTemporaryFile(buffering=0) as out, \
            tempfile.NamedTemporaryFile(buffering=0) as err, \
            _open_output_reader(out.name) as out_reader, \
            _open_output_reader(err.name) as err_reader:
        # Separate open handles have independent offsets; seeking a duplicated
        # child handle while it is writing could overwrite earlier output.
        readers = (("stdout", out_reader), ("stderr", err_reader))
        decoders = {name: _OutputDecoder() for name, _ in readers}

        def drain(*, final=False):
            if on_output is None:
                return
            for name, reader in readers:
                while True:
                    chunk = reader.read(65536)
                    text = decoders[name].decode(chunk, final=final and not chunk)
                    if text:
                        on_output(name, text)
                    if not final or not chunk:
                        break

        command = CommandProcess(_resolve_command_argv(argv), cwd=working_directory, stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=err, shell=False)
        handle = current_handle()
        unsubscribe = handle.on_cancel(command.request_stop) if handle is not None else lambda: None
        failure = None
        try:
            deadline = monotonic() + timeout_seconds
            while command.process.poll() is None:
                drain()
                if check_execution is not None:
                    check_execution()
                if monotonic() >= deadline:
                    raise TimeoutError(f"command timed out after {timeout_seconds} seconds")
                try:
                    command.process.wait(timeout=min(0.1, max(0.001, deadline - monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            returncode = command.process.returncode
            if handle is not None:
                handle.check()
        except Exception as error:
            failure = error
        finally:
            unsubscribe()
            try:
                command.close()
            finally:
                drain(final=True)
        out.seek(0)
        err.seek(0)
        stdout = _OutputDecoder().decode(out.read(), final=True).replace("\r\n", "\n").replace("\r", "\n")
        stderr = _OutputDecoder().decode(err.read(), final=True).replace("\r\n", "\n").replace("\r", "\n")
        if failure is not None:
            failure.args = (f"{failure}\nstdout:\n{stdout or '<empty>'}\nstderr:\n{stderr or '<empty>'}",)
            raise failure
    return "\n".join(
        (
            f"exit_code: {returncode}",
            f"stdout:\n{stdout or '<empty>'}",
            f"stderr:\n{stderr or '<empty>'}",
        )
    )
