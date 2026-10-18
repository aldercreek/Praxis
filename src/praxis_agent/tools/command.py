import codecs
import locale
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import monotonic

from praxis_agent.execution import ExecutionStopped, current_handle
from praxis_agent.runtime import CommandShell, select_command_shell
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
    argv: list[str] | None = None,
    workdir: str = ".",
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    *,
    command: str | None = None,
    command_shell: CommandShell | None = None,
    check_execution: Callable[[], None] | None = None,
    on_output: Callable[[str, str], None] | None = None,
) -> dict[str, object]:
    """执行程序或 shell 正文，清理进程后返回同一份结构化结果。"""
    if (argv is None) == (command is None):
        raise ValueError("argv 和 command 必须且只能提供一个；命令未执行。")
    if argv is not None:
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
            raise ValueError("argv must be a non-empty list of strings")
        if not argv[0]:
            raise ValueError("argv[0] must be a non-empty executable name")
    elif not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
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

        try:
            if command is not None:
                selected_shell = command_shell or select_command_shell()
                if selected_shell is None:
                    raise FileNotFoundError("未检测到可用的命令 shell")
                argv = selected_shell.argv(command)
            process = CommandProcess(_resolve_command_argv(argv), cwd=working_directory, stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=err, shell=False)
        except OSError as error:
            if isinstance(error, FileNotFoundError):
                reason = "找不到可执行文件或可用的命令 shell。请检查命令名称、安装情况或程序路径。"
            elif isinstance(error, PermissionError):
                reason = "访问被拒绝。请检查执行权限。"
            else:
                reason = str(error)
            return {"status": "spawn_failed", "exit_code": None, "stdout": "", "stderr": "",
                    "error": f"无法启动命令：{reason}"}
        handle = current_handle()
        unsubscribe = handle.on_cancel(process.request_stop) if handle is not None else lambda: None
        status = "exited"
        failure = None
        try:
            deadline = monotonic() + timeout_seconds
            while process.process.poll() is None:
                drain()
                if check_execution is not None:
                    check_execution()
                if monotonic() >= deadline:
                    status = "timed_out"
                    break
                try:
                    process.process.wait(timeout=min(0.1, max(0.001, deadline - monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            if check_execution is not None:
                check_execution()
            if handle is not None:
                handle.check()
        except ExecutionStopped:
            status = "cancelled"
        except Exception as error:
            failure = error
        finally:
            unsubscribe()
            try:
                process.close()
            except Exception as error:
                failure = error
            finally:
                drain(final=True)
        out.seek(0)
        err.seek(0)
        stdout = _OutputDecoder().decode(out.read(), final=True).replace("\r\n", "\n").replace("\r", "\n")
        stderr = _OutputDecoder().decode(err.read(), final=True).replace("\r\n", "\n").replace("\r", "\n")
        if failure is not None:
            failure.args = (f"{failure}\nstdout:\n{stdout or '<empty>'}\nstderr:\n{stderr or '<empty>'}",)
            raise failure
        returncode = process.process.returncode
    error = None
    if status == "timed_out":
        error = f"执行超过 {timeout_seconds} 秒，命令及其子进程已停止。已保留停止前的输出。"
    elif status == "cancelled":
        error = "用户取消了执行，命令及其子进程已停止。已保留停止前的输出。"
    elif returncode != 0:
        error = (f"命令已执行并退出，退出码为 {returncode}。请结合 stdout 和 stderr 判断具体原因；"
                 "非零退出码的含义由具体程序决定。")
    return {"status": status, "exit_code": returncode, "stdout": stdout, "stderr": stderr, "error": error}
