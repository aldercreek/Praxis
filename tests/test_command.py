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


def test_run_command_rejects_a_directory_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="must stay inside the workspace"):
        run_command(workspace, [sys.executable, "--version"], "..", 10)


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
