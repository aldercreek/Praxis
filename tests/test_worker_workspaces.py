import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from praxis_agent.multi_agent.models import MultiAgentConfig
from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.planning import RunState
from praxis_agent.tools import ToolContext, create_default_registry


def test_binds_supplied_directory_without_copying_and_keeps_it_on_close(tmp_path):
    parent, worker = tmp_path / "parent", tmp_path / "worker"
    parent.mkdir()
    worker.mkdir()
    (parent / "data.csv").write_text("parent")
    (worker / "data.csv").write_text("prepared input")
    (worker / "empty-output").mkdir()
    finished = Event()

    def factory(agent_id, control, sink):
        def run(task):
            assert runtime.handles[agent_id].workspace == worker
            assert (worker / "empty-output").is_dir()
            (worker / "report.txt").write_text("done")
            finished.set()
            return "done"
        return SimpleNamespace(run=run, close=lambda: None)

    runtime = AgentRuntime(factory, MultiAgentConfig(), workspace=parent)
    try:
        info = runtime.spawn("Process prepared data", working_directory="../worker")
        assert Path(info["workspace"]) == worker
        assert finished.wait(5)
    finally:
        runtime.shutdown(lambda: None)
    assert (parent / "data.csv").read_text() == "parent"
    assert not (parent / "report.txt").exists()
    assert (worker / "data.csv").read_text() == "prepared input"
    assert (worker / "report.txt").read_text() == "done"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["parent", "worker"]


def test_invalid_directory_does_not_create_a_worker_or_files(tmp_path):
    runtime = AgentRuntime(lambda *args: pytest.fail("must validate before creating worker"),
                           MultiAgentConfig(), workspace=tmp_path)
    (tmp_path / "file.txt").write_text("file")
    try:
        for directory in ("", "missing", "file.txt", None):
            with pytest.raises(ValueError, match="working_directory"):
                runtime.spawn("task", working_directory=directory)
        assert runtime.handles == {}
        assert not (tmp_path / "missing").exists()
    finally:
        runtime.shutdown(lambda: None)


def test_absolute_deliveries_from_multiple_directories_remain_readable(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    registry = create_default_registry()
    context = ToolContext(parent, RunState("review"))
    for name in ("worker-a", "worker-b"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "report.txt").write_text(name)
        read = registry.execute("read_file", {"relative_path": str(directory / "report.txt")}, context=context)
        assert read.success and name in read.output
        listing = registry.execute("list_files", {"relative_path": str(directory)}, context=context)
        assert listing.success and "report.txt" in listing.output
        found = registry.execute("find_files", {
            "relative_path": str(directory), "file_pattern": "*.txt",
        }, context=context)
        assert found.success and json.loads(found.output)["files"] == ["report.txt"]
        searched = registry.execute("search_text", {
            "relative_path": str(directory), "query": name,
        }, context=context)
        assert searched.success and json.loads(searched.output)["results"] == [f"report.txt:1:{name}"]
        write = registry.execute("write_file", {"relative_path": str(directory / "bad.txt"), "content": "bad"}, context=context)
        assert not write.success and not (directory / "bad.txt").exists()


def test_git_commands_support_repeated_merges_and_report_real_conflicts(tmp_path):
    registry = create_default_registry()
    context = ToolContext(tmp_path, RunState("git workflow"))

    def git(*args, success=True):
        result = registry.execute("run_command", {
            "argv": ["git", *map(str, args)], "workdir": ".", "timeout_seconds": 30,
        }, context=context)
        assert result.success, result.output
        if success:
            assert result.structured_output["exit_code"] == 0, result.output
        return result.structured_output

    def commit(directory, message):
        git("-C", directory, "add", "settings.txt")
        git("-C", directory, "commit", "-m", message)

    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (tmp_path / "settings.txt").write_text("timeout=10\n")
    commit(tmp_path, "base")
    worker = tmp_path.parent / (tmp_path.name + "-worker")
    git("worktree", "add", "-b", "worker", worker)
    for value in (20, 30):
        (worker / "settings.txt").write_text(f"timeout={value}\n")
        commit(worker, f"timeout {value}")
        git("merge", "--no-edit", "worker")
        assert (tmp_path / "settings.txt").read_text() == f"timeout={value}\n"
    (tmp_path / "settings.txt").write_text("timeout=40\n")
    commit(tmp_path, "parent change")
    (worker / "settings.txt").write_text("timeout=50\n")
    commit(worker, "worker change")
    conflict = git("merge", "--no-edit", "worker", success=False)
    assert conflict["exit_code"] == 1
    assert "settings.txt" in git("diff", "--name-only", "--diff-filter=U")["stdout"]
    git("merge", "--abort")
    assert (tmp_path / "settings.txt").read_text() == "timeout=40\n"
    assert (worker / "settings.txt").read_text() == "timeout=50\n"
