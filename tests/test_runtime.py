from pathlib import Path

import pytest

from praxis_agent.runtime import collect_runtime_context


def test_collect_runtime_context_detects_known_shells(
    tmp_path: Path,
    monkeypatch,
) -> None:
    detected_shells = {"cmd", "pwsh"}
    monkeypatch.setattr(
        "praxis_agent.runtime.platform.system",
        lambda: "ExampleOS",
    )
    monkeypatch.setattr(
        "praxis_agent.runtime.sys.platform",
        "example-platform",
    )
    monkeypatch.setattr(
        "praxis_agent.runtime.shutil.which",
        lambda name: name if name in detected_shells else None,
    )

    context = collect_runtime_context(tmp_path)

    assert context.operating_system == "ExampleOS"
    assert context.sys_platform == "example-platform"
    assert context.workspace_root == tmp_path.resolve()
    assert context.available_shells == ("cmd", "pwsh")
    assert context.unavailable_shells == ("powershell", "bash", "sh")


def test_collect_runtime_context_requires_a_directory(tmp_path: Path) -> None:
    missing_workspace = tmp_path / "missing"

    with pytest.raises(NotADirectoryError, match="workspace is not a directory"):
        collect_runtime_context(missing_workspace)
