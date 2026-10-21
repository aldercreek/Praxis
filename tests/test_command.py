import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from praxis_agent.tools import run_command
from praxis_agent.runtime import select_command_shell


def test_run_command_returns_exit_code_and_output(tmp_path: Path) -> None:
    result = run_command(
        tmp_path,
        [sys.executable, "-c", "print('hello from command')"],
        ".",
        10,
    )

    assert result == {"status": "exited", "exit_code": 0, "stdout": "hello from command\n",
                      "stderr": "", "error": None}


def test_run_command_uses_a_workspace_subdirectory(tmp_path: Path) -> None:
    working_directory = tmp_path / "nested"
    working_directory.mkdir()

    result = run_command(
        tmp_path,
        [
            sys.executable,
            "-c",
            "from pathlib import Path; print(Path.cwd().name)",
        ],
        "nested",
        10,
    )

    assert result["stdout"] == "nested\n"


@pytest.mark.parametrize("absolute", [False, True])
def test_run_command_uses_a_directory_outside_workspace(
    tmp_path: Path, absolute: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = tmp_path / "other project"
    project.mkdir()
    (project / "input.txt").write_text("project input", encoding="utf-8")
    original_cwd = Path.cwd()

    result = run_command(
        workspace,
        [sys.executable, "-c", "from pathlib import Path; print(Path('input.txt').read_text())"],
        workdir=str(project) if absolute else "../other project",
        timeout_seconds=10,
    )

    assert result["exit_code"] == 0
    assert result["stdout"] == "project input\n"
    assert Path.cwd() == original_cwd


@pytest.mark.parametrize("workdir", ["missing", "file.txt"])
def test_run_command_rejects_an_invalid_workdir(tmp_path: Path, workdir: str) -> None:
    (tmp_path / "file.txt").write_text("file", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="does not exist or is not a directory") as error:
        run_command(tmp_path, [sys.executable, "--version"], workdir=workdir)
    assert str(tmp_path / workdir) in str(error.value)


@pytest.mark.parametrize("workdir", ["", None, 123])
def test_run_command_rejects_an_empty_or_non_string_workdir(tmp_path: Path, workdir) -> None:
    with pytest.raises(ValueError, match="workdir must be a non-empty string"):
        run_command(tmp_path, [sys.executable, "--version"], workdir=workdir)


def test_run_command_rejects_an_invalid_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 1 and 120"):
        run_command(tmp_path, [sys.executable, "--version"], ".", 121)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows executable search order")
def test_run_command_honors_path_without_overriding_explicit_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = tmp_path / "selected environment"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    monkeypatch.setenv("PATH", str(environment / "Scripts") + os.pathsep + os.environ["PATH"])
    argv = ["python", "-c", "import sys; print(sys.prefix)"]

    result = run_command(tmp_path, argv, timeout_seconds=10)
    assert result["stdout"] == f"{environment}\n"
    assert argv == ["python", "-c", "import sys; print(sys.prefix)"]

    explicit = run_command(tmp_path, [sys.executable, *argv[1:]], timeout_seconds=10)
    assert explicit["stdout"] == f"{sys.prefix}\n"


def test_direct_arguments_preserve_empty_unicode_quotes_and_metacharacters(tmp_path):
    arguments = ["", "中文 空格", 'say "hello"', "$HOME", "a|b", "C:\\ending\\"]
    result = run_command(tmp_path, [sys.executable, "-c",
                                   "import json,sys; print(json.dumps(sys.argv[1:]))", *arguments])
    assert result["exit_code"] == 0, result
    assert json.loads(result["stdout"]) == arguments


@pytest.mark.parametrize("arguments", [{}, {"argv": ["unused"], "command": "unused"},
                                       {"argv": []}, {"argv": [""]}, {"command": " \n"}])
def test_invalid_execution_inputs_do_not_start_a_process(tmp_path, monkeypatch, arguments):
    monkeypatch.setattr("praxis_agent.tools.command.CommandProcess", lambda *a, **k: pytest.fail("process started"))
    with pytest.raises(ValueError):
        run_command(tmp_path, **arguments)


def test_missing_executable_returns_generic_startup_error(tmp_path):
    result = run_command(tmp_path, [str(tmp_path / "missing-command.exe")])
    assert result["status"] == "spawn_failed"
    assert result["exit_code"] is None
    assert result["stdout"] == result["stderr"] == ""
    assert result["error"].startswith("Unable to start command: Executable")


def test_nonzero_exit_preserves_both_streams_and_explains_the_code(tmp_path):
    result = run_command(tmp_path, [sys.executable, "-c",
                                   "import sys; print('result'); print('failure', file=sys.stderr); sys.exit(7)"])
    assert result["status"] == "exited"
    assert result["exit_code"] == 7
    assert result["stdout"] == "result\n"
    assert result["stderr"] == "failure\n"
    assert "exited with code 7" in result["error"]


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell execution")
def test_powershell_multiline_pipeline_and_redirection_in_unicode_directory(tmp_path):
    workdir = tmp_path / "中文 project"
    workdir.mkdir()
    result = run_command(tmp_path, command="""
$text = 'literal $HOME; say "hello"; 反引号 `'
$text > '结果 文件.txt'
Get-Content -LiteralPath '结果 文件.txt' | Select-String -SimpleMatch '$HOME'
[Console]::Error.WriteLine('warning')
""", workdir=str(workdir), timeout_seconds=10)
    assert result["status"] == "exited" and result["exit_code"] == 0, result
    assert 'literal $HOME; say "hello"; 反引号 `' in result["stdout"]
    assert result["stderr"] == "warning\n"
    assert [file.name for file in workdir.iterdir()] == ["结果 文件.txt"]


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell native program exit code")
def test_powershell_forwards_native_exit_code_without_command_wrapping(tmp_path):
    script = tmp_path / "测试 script.py"
    script.write_text("import sys; print('native output'); sys.exit(7)", encoding="utf-8")
    python = sys.executable.replace("'", "''")
    result = run_command(tmp_path, command=f"& '{python}' '测试 script.py'\nexit $LASTEXITCODE")
    assert result["exit_code"] == 7, result
    assert result["stdout"] == "native output\n"
    assert "exited with code 7" in result["error"]


def test_selected_shell_executes_script_in_the_requested_directory(tmp_path):
    shell = select_command_shell()
    if shell is None:
        pytest.skip("no supported shell installed")
    script = "(Get-Location).Path" if shell.name in {"pwsh", "powershell"} else "pwd"
    result = run_command(tmp_path, command=script, command_shell=shell)
    assert result["exit_code"] == 0, result
    assert Path(result["stdout"].strip()).resolve() == tmp_path.resolve()


def test_cleanup_error_is_reported_with_output_instead_of_claiming_success(tmp_path, monkeypatch):
    from praxis_agent.tools.processes import CommandProcess
    close = CommandProcess.close

    def fail_after_close(process):
        close(process)
        raise OSError("cleanup failed")

    monkeypatch.setattr(CommandProcess, "close", fail_after_close)
    with pytest.raises(OSError, match="cleanup failed") as failure:
        run_command(tmp_path, [sys.executable, "-c", "print('saved output')"])
    assert "saved output" in str(failure.value)
