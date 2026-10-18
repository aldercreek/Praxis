import os
import subprocess
import sys
from pathlib import Path

import pytest

from praxis_agent.tools import run_command


def test_run_command_returns_exit_code_and_output(tmp_path: Path) -> None:
    result = run_command(
        tmp_path,
        [sys.executable, "-c", "print('hello from command')"],
        ".",
        10,
    )

    assert "exit_code: 0" in result
    assert "hello from command" in result
    assert "stderr:\n<empty>" in result


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

    assert "stdout:\nnested\n" in result


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

    assert "exit_code: 0" in result
    assert "stdout:\nproject input\n" in result
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
    assert f"stdout:\n{environment}\n" in result
    assert argv == ["python", "-c", "import sys; print(sys.prefix)"]

    explicit = run_command(tmp_path, [sys.executable, *argv[1:]], timeout_seconds=10)
    assert f"stdout:\n{sys.prefix}\n" in explicit
