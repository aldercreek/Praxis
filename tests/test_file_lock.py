"""Deterministic interleavings for file transactions and whole-run cancellation."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from queue import Queue
from threading import Condition, Event

import pytest

from praxis_agent.execution import ExecutionStopped, Run, run_operation
from praxis_agent.model import FunctionCall
from praxis_agent.planning import RunState
from praxis_agent.tool_execution import ToolCallExecutor
from praxis_agent.tools import ToolContext, create_default_registry, edit_file, write_file
from praxis_agent.tools import file_lock
from praxis_agent.tools.code import CodeRuntime
from praxis_agent.tools.text_file import TextFile


@pytest.fixture
def waiting(monkeypatch):
    """Signal only after a waiter owns the condition, just before releasing it."""
    arrivals = Queue()

    class ObservedCondition(Condition):
        def wait(self, timeout=None):
            assert timeout is None  # Cancellation must notify, not poll.
            arrivals.put(self)
            return super().wait(timeout)

    original = file_lock._FileState

    def observed_state():
        return original(condition=ObservedCondition())

    monkeypatch.setattr(file_lock, "_FileState", observed_state)
    return arrivals


@pytest.fixture
def paused_save(monkeypatch):
    entered, release = Event(), Event()
    original = TextFile.write

    def write(self, path, content):
        if content == "a=2\nb=1\n":
            entered.set()
            assert release.wait(5), "test did not release the saving writer"
        return original(self, path, content)

    monkeypatch.setattr(TextFile, "write", write)
    yield entered, release
    release.set()


def edit(workspace, name="data.txt", old="a=1", new="a=2"):
    return edit_file(workspace, name, [{"old_text": old, "new_text": new}])


@pytest.mark.parametrize("second_path", [
    "data.txt", "folder/../data.txt",
    pytest.param("DATA.TXT", marks=pytest.mark.skipif(os.name != "nt", reason="Windows alias")),
])
def test_same_file_edits_read_latest_content(tmp_path, waiting, paused_save, second_path):
    path = tmp_path / "data.txt"
    path.write_bytes(b"a=1\nb=1\n")
    (tmp_path / "folder").mkdir()
    entered, release = paused_save
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(edit, tmp_path)
        try:
            assert entered.wait(2)
            second = pool.submit(edit, tmp_path, second_path, "b=1", "b=2")
            waiting.get(timeout=2)
            assert not second.done()
        finally:
            release.set()
        first.result(timeout=2)
        change = second.result(timeout=2)
    assert path.read_bytes() == b"a=2\nb=2\n"
    assert " a=2" in change.diff


def test_conflicting_edit_fails_and_releases_lock(tmp_path, waiting, paused_save):
    path = tmp_path / "data.txt"
    path.write_bytes(b"a=1\nb=1\n")
    entered, release = paused_save
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(edit, tmp_path)
        try:
            assert entered.wait(2)
            second = pool.submit(edit, tmp_path, "data.txt", "a=1", "a=3")
            waiting.get(timeout=2)
        finally:
            release.set()
        first.result(timeout=2)
        with pytest.raises(ValueError, match="old_text was not found"):
            second.result(timeout=2)
    assert path.read_bytes() == b"a=2\nb=1\n"
    edit(tmp_path, old="a=2", new="a=4")
    assert path.read_bytes() == b"a=4\nb=1\n"
    assert not file_lock._states


def test_write_and_edit_share_lock_but_other_files_do_not(tmp_path, waiting, paused_save):
    path = tmp_path / "data.txt"
    path.write_bytes(b"a=1\nb=1\n")
    entered, release = paused_save
    with ThreadPoolExecutor(3) as pool:
        first = pool.submit(edit, tmp_path)
        try:
            assert entered.wait(2)
            second = pool.submit(write_file, tmp_path, "data.txt", "replacement\n")
            waiting.get(timeout=2)
            pool.submit(write_file, tmp_path, "other.txt", "independent").result(timeout=2)
            assert not second.done()
        finally:
            release.set()
        first.result(timeout=2)
        change = second.result(timeout=2)
    assert "-a=2" in change.diff  # Full overwrite used the latest before-image.
    assert path.read_bytes() == b"replacement\n"


@pytest.mark.parametrize("with_handles", [False, True])
def test_whole_run_cancel_wakes_all_waiters_before_holder_releases(tmp_path, waiting, with_handles):
    path = tmp_path / "data.txt"
    path.write_bytes(b"a=1\nb=1\n")
    scope = Run()

    def operation():
        if with_handles:
            return run_operation("edit", lambda handle: edit(tmp_path))
        return edit(tmp_path)

    def exercise():
        # Keep the holder outside this Run; cancellation must release both
        # waiters without depending on a notification from the holder.
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(copy_context().run, operation) for _ in range(2)]
            try:
                condition = waiting.get(timeout=2)
                assert waiting.get(timeout=2) is condition
                # An unrelated notification must not grant ownership.
                with condition:
                    condition.notify_all()
                waiting.get(timeout=2)
                waiting.get(timeout=2)
                scope.cancel()
                for future in futures:
                    with pytest.raises(ExecutionStopped):
                        future.result(timeout=2)
                assert all(state.occupied for state in file_lock._states.values())
                assert not scope._callbacks
                assert all(not handle._callbacks for handle in scope.handles)
            finally:
                scope.cancel()

    with file_lock.file_transaction(path):
        with pytest.raises(ExecutionStopped):
            scope.execute(exercise)
        assert path.read_bytes() == b"a=1\nb=1\n"
    # A cancelled waiter must not leave occupancy or cancellation on the file.
    Run().execute(lambda: edit(tmp_path))
    assert path.read_bytes() == b"a=2\nb=1\n"
    assert not file_lock._states


def test_cancellation_before_registration_never_enters_transaction(tmp_path):
    scope = Run()

    def exercise():
        scope.cancel()
        write_file(tmp_path, "new.txt", "must not be saved")

    with pytest.raises(ExecutionStopped):
        scope.execute(exercise)
    assert not (tmp_path / "new.txt").exists()
    assert not scope._callbacks
    assert not file_lock._states


@pytest.mark.parametrize("operation", [edit, lambda root: write_file(root, "data.txt", "replacement")])
def test_cancel_after_read_prevents_save(tmp_path, monkeypatch, operation):
    path = tmp_path / "data.txt"
    path.write_bytes(b"a=1\nb=1\n")
    scope = Run()
    original = TextFile.read

    def read(cls, path):
        source = original(path)
        scope.cancel()
        return source

    monkeypatch.setattr(TextFile, "read", classmethod(read))
    with pytest.raises(ExecutionStopped):
        scope.execute(lambda: operation(tmp_path))
    assert path.read_bytes() == b"a=1\nb=1\n"
    assert not file_lock._states


def test_save_failure_releases_lock(tmp_path, monkeypatch):
    path = tmp_path / "data.txt"
    path.write_bytes(b"a=1\nb=1\n")

    def fail(*args):
        raise OSError("disk failure")

    with monkeypatch.context() as patch:
        patch.setattr(TextFile, "write", fail)
        with pytest.raises(OSError, match="disk failure"):
            edit(tmp_path)
    edit(tmp_path)
    assert path.read_bytes() == b"a=2\nb=1\n"
    assert not file_lock._states


def test_cancellation_during_save_allows_save_to_finish(tmp_path, paused_save):
    path = tmp_path / "data.txt"
    path.write_bytes(b"a=1\nb=1\n")
    scope = Run()
    entered, release = paused_save
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(scope.execute, lambda: edit(tmp_path))
        try:
            assert entered.wait(2)
            scope.cancel()
            assert not future.done()
            assert any(state.occupied for state in file_lock._states.values())
        finally:
            release.set()
        with pytest.raises(ExecutionStopped):
            future.result(timeout=2)
    assert path.read_bytes() == b"a=2\nb=1\n"
    assert not file_lock._states


@pytest.mark.parametrize("cancel", [False, True])
def test_real_run_code_parallel_edits_and_whole_turn_cancel(tmp_path, waiting, cancel):
    path = tmp_path / "data.txt"
    path.write_bytes(b"a=1\nb=1\n")
    runtime = CodeRuntime(output_dir=tmp_path / "logs")
    registry = create_default_registry()
    registry.register(runtime.tool())
    executor = ToolCallExecutor(
        registry, ToolContext(tmp_path, RunState("file edits")),
        publish_event=lambda event: None, bypass_approvals=True,
    )
    requests = [{"name": "edit_file", "arguments": {
        "relative_path": "data.txt", "edits": [{"old_text": f"{key}=1", "new_text": f"{key}=2"}],
    }} for key in ("a", "b")]
    call = FunctionCall("code", "run_code", json.dumps({"code": f"call_parallel_tools({requests!r})"}))
    scope = Run()
    try:
        with ThreadPoolExecutor(1) as pool:
            with file_lock.file_transaction(path):
                future = pool.submit(scope.execute, lambda: executor.execute("turn", call))
                try:
                    condition = waiting.get(timeout=5)
                    assert waiting.get(timeout=5) is condition
                    if cancel:
                        scope.cancel()
                        # The entire Monty/tool tree must drain while the file
                        # is STILL occupied by this unrelated holder.
                        with pytest.raises(ExecutionStopped):
                            future.result(timeout=5)
                        assert path.read_bytes() == b"a=1\nb=1\n"
                except BaseException:
                    scope.cancel()
                    raise
            if not cancel:
                result = future.result(timeout=5)
                assert result.success, result.output
                payload = json.loads(result.output)
                assert all(item["success"] for item in payload["result"])
                assert path.read_bytes() == b"a=2\nb=2\n"
    finally:
        runtime.close()
    assert not file_lock._states
