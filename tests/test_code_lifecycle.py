"""Resident workers: real process ownership, isolation, concurrency and teardown."""
import ctypes
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from praxis_agent.planning import RunState
from praxis_agent.tools import ToolContext, ToolResult
from praxis_agent.tools.code import CodeConfig, CodeRuntime


def alive(pid):
    if os.name == "nt":
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def run(runtime, context, code, agent="parent", *, success=True):
    result = runtime.run(context, code, agent_id=agent)
    assert result.success is success, result.output
    return json.loads(result.output)


@pytest.fixture
def context(tmp_path):
    return ToolContext(tmp_path, RunState("worker lifecycle"))


def test_lazy_worker_reuse_isolation_and_process_reaping(context, tmp_path):
    runtime = CodeRuntime(output_dir=tmp_path / "logs")
    pids = []
    try:
        runtime.tool("unused")
        assert runtime._workers == {}
        assert runtime._thread is None
        first = run(runtime, context, "rows = [1, 2]\ndef total():\n    return sum(rows)\ntotal()")
        pid = runtime._workers["parent"].session.worker_pid
        pids.append(pid)
        assert pid and alive(pid) and first["worker_created"]
        second = run(runtime, context, "rows.append(3)\ntotal()")
        assert second["result"] == 6
        assert not second["worker_created"]
        assert runtime._workers["parent"].session.worker_pid == pid
        assert run(runtime, context, "'rows' in locals()", "child")["result"] is False
        child_pid = runtime._workers["child"].session.worker_pid
        pids.append(child_pid)
        assert child_pid != pid

        runtime.release_agent("parent")
        assert not alive(pid)
        assert alive(child_pid)
        empty = run(runtime, context, "'rows' in locals()")
        assert empty["worker_created"] and not empty["result"]
        pids.append(runtime._workers["parent"].session.worker_pid)
    finally:
        runtime.close()
    assert runtime._workers == {}
    assert all(not alive(pid) for pid in pids)


def test_ten_agents_can_enter_tools_concurrently_and_retain_state(context, tmp_path):
    runtime = CodeRuntime(CodeConfig(queue_seconds=2), output_dir=tmp_path / "logs")
    entered = threading.Barrier(11)
    release = threading.Event()

    def hold(name, arguments):
        entered.wait(10)
        assert release.wait(10)
        return ToolResult(True, "ok")

    context = replace(context, invoke_programmatic_tool=hold)
    try:
        with ThreadPoolExecutor(max_workers=10) as threads:
            jobs = [threads.submit(run, runtime, context,
                                   f"value: int = {i}\ncall_tool('hold', {{}})\nvalue", f"agent-{i}")
                    for i in range(10)]
            try:
                entered.wait(10)
            finally:
                release.set()
            assert [job.result(timeout=10)["result"] for job in jobs] == list(range(10))
        assert len({worker.session.worker_pid for worker in runtime._workers.values()}) == 10
        for i in range(10):
            result = run(runtime, context, "value += 10\nvalue", f"agent-{i}")
            assert result["result"] == i + 10
            assert result["worker_created"] is False
    finally:
        release.set()
        runtime.close()


def test_calls_within_one_agent_are_serialized(context, tmp_path):
    runtime = CodeRuntime(output_dir=tmp_path / "logs")
    entered, release, queued = threading.Event(), threading.Event(), threading.Event()

    def hold(name, arguments):
        entered.set()
        assert release.wait(5)
        return ToolResult(True, "ok")

    def followup():
        queued.set()
        return run(runtime, context, "value += 1\nvalue")

    context = replace(context, invoke_programmatic_tool=hold)
    try:
        with ThreadPoolExecutor(max_workers=2) as threads:
            first = threads.submit(run, runtime, context, "value: int = 10\ncall_tool('hold', {})\nvalue")
            try:
                assert entered.wait(5)
                second = threads.submit(followup)
                assert queued.wait(5)
                assert not second.done()
            finally:
                release.set()
            assert first.result(timeout=5)["result"] == 10
            assert second.result(timeout=5)["result"] == 11
    finally:
        release.set()
        runtime.close()


def test_compute_timeout_discards_only_affected_worker(context, tmp_path):
    runtime = CodeRuntime(CodeConfig(compute_seconds=0.1), output_dir=tmp_path / "logs")
    try:
        run(runtime, context, "value = 1")
        run(runtime, context, "value = 99", "other")
        old_pid = runtime._workers["parent"].session.worker_pid
        failed = run(runtime, context, "while True:\n    pass", success=False)
        assert failed["error_kind"] in ("resource", "timeout")
        assert failed["state_lost"] is True
        assert not alive(old_pid)
        assert run(runtime, context, "value", "other")["result"] == 99
        rebuilt = run(runtime, context, "'value' in locals()")
        assert rebuilt["worker_created"] is True
        assert rebuilt["result"] is False
    finally:
        runtime.close()


def test_release_during_execution_waits_for_callbacks_and_reaps_worker(context, tmp_path):
    from praxis_agent.execution import current_run, run_operation

    runtime = CodeRuntime(output_dir=tmp_path / "logs")
    entered, cancelled, finish = threading.Event(), threading.Event(), threading.Event()

    def hold(name, arguments):
        unsubscribe = current_run().on_cancel(cancelled.set)
        try:
            entered.set()
            assert cancelled.wait(5)
            assert finish.wait(5)
            return ToolResult(True, "done")
        finally:
            unsubscribe()

    context = replace(context, invoke_programmatic_tool=lambda name, arguments:
                      run_operation(name, lambda handle: hold(name, arguments)))
    run(runtime, context, "x = 1")
    pid = runtime._workers["parent"].session.worker_pid
    try:
        with ThreadPoolExecutor(max_workers=2) as threads:
            execution = threads.submit(run, runtime, context, "call_tool('hold', {})", success=False)
            try:
                assert entered.wait(5)
                closing = threads.submit(runtime.release_agent, "parent")
                assert cancelled.wait(5)
                assert not closing.done()
            finally:
                finish.set()
            assert execution.result(timeout=5)["state_lost"] is True
            closing.result(timeout=5)
        assert not alive(pid)
        assert "parent" not in runtime._workers
    finally:
        finish.set()
        runtime.close()
