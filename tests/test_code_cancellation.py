"""Cancellation through the real tool executor and Monty subprocess pool."""
import json
import threading
import time
from concurrent.futures import Future

import pytest

from praxis_agent.execution import ExecutionStopped, Run, current_handle, current_run
from praxis_agent.model import FunctionCall
from praxis_agent.planning import RunState
from praxis_agent.tool_execution import ToolCallExecutor
from praxis_agent.tools import Tool, ToolContext, ToolResult, create_default_registry
from praxis_agent.tools.code import CodeConfig, CodeRuntime


@pytest.fixture
def code_runtime():
    runtime = CodeRuntime(CodeConfig(max_processes=2, compute_seconds=8))
    try:
        yield runtime
    finally:
        runtime.close()


def executor_for(runtime, tmp_path, agent_id="parent"):
    registry = create_default_registry()
    registry.register(runtime.tool(agent_id))
    return ToolCallExecutor(
        registry, ToolContext(tmp_path, RunState("cancellation test")),
        publish_event=lambda event: None,
    )


def execute(executor, code):
    return executor.execute("turn", FunctionCall(
        "code", "run_code", json.dumps({"session_id": "test", "code": code}),
    ))


def start(executor, code):
    scope = Run("test turn")
    finished = Future()
    def worker():
        try:
            finished.set_result(scope.execute(lambda: execute(executor, code)))
        except BaseException as error:
            finished.set_exception(error)
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return scope, finished, thread


def add_signal(executor, name="ready"):
    entered = threading.Event()
    executor.tool_registry.register(Tool(
        name, "Signal code execution", {"type": "object"},
        lambda context: entered.set() or ToolResult(True, "ready"), programmatic=True,
    ))
    return entered


@pytest.mark.parametrize("body", ["while True:\n    pass", "import time\ntime.sleep(20)"])
def test_cancel_running_code_restores_snapshot_and_keeps_other_agent(code_runtime, tmp_path, body):
    executor = executor_for(code_runtime, tmp_path)
    other = executor_for(code_runtime, tmp_path, "other")
    assert execute(executor, "x: int = 1").success
    assert execute(other, "x = 99").success
    ready = add_signal(executor)
    other_ready = add_signal(other)
    scope, finished, thread = start(executor, "x = 2\ncall_tool('ready', {})\n" + body)
    other_scope, other_finished, other_thread = start(
        other, "call_tool('ready', {})\nimport time\ntime.sleep(0.5)\nx",
    )
    try:
        assert ready.wait(5), finished.result(timeout=1)
        assert other_ready.wait(5)
        time.sleep(0.05)  # Let the worker enter the loop/sleep after its callback.
        # Cancel only the run_code tool handle, not the enclosing turn. Its
        # private execution scope must still receive the request.
        scope.handles[0].cancel()
        result = finished.result(timeout=2)
        payload = json.loads(result.output)
        assert not result.success
        assert payload["error_kind"] == "cancelled"
        assert payload["state_saved"] is False
        assert json.loads(other_finished.result(timeout=2).output)["result"] == 99
        assert json.loads(execute(executor, "x").output)["result"] == 1
    finally:
        scope.cancel()
        other_scope.cancel()
        thread.join(10)
        other_thread.join(10)
    assert not thread.is_alive()
    assert not other_thread.is_alive()


def test_cancel_waits_for_nested_tool_cleanup(code_runtime, tmp_path):
    executor = executor_for(code_runtime, tmp_path)
    entered = threading.Event()
    cancelled = threading.Event()
    release_cleanup = threading.Event()
    observed = []

    def waiting_tool(context):
        handle = current_handle()
        observed.append(current_run().name)
        unsubscribe = handle.on_cancel(cancelled.set)
        try:
            entered.set()
            assert cancelled.wait(5)
            assert release_cleanup.wait(5)
            return ToolResult(True, "cleaned")
        finally:
            unsubscribe()

    executor.tool_registry.register(Tool(
        "waiting_tool", "Cancellable host operation", {"type": "object"},
        waiting_tool, programmatic=True,
    ))
    scope, finished, thread = start(executor, "call_tool('waiting_tool', {})")
    try:
        assert entered.wait(5)
        scope.handles[0].cancel()
        assert cancelled.wait(2)
        assert not finished.done(), "run_code returned before host tool cleanup"
        release_cleanup.set()
        payload = json.loads(finished.result(timeout=2).output)
        assert payload["error_kind"] == "cancelled"
        assert payload["state_saved"] is False
        assert observed == ["run_code"]
        assert execute(executor, "42").success
    finally:
        release_cleanup.set()
        scope.cancel()
        thread.join(10)


def test_cancel_queued_call_does_not_execute(code_runtime, tmp_path):
    executor = executor_for(code_runtime, tmp_path)
    ready = add_signal(executor)
    first, first_finished, first_thread = start(
        executor, "call_tool('ready', {})\nimport time\ntime.sleep(20)",
    )
    queued = queued_thread = None
    try:
        assert ready.wait(5)
        queued, queued_finished, queued_thread = start(executor, "x = 123")
        queued.cancel()
        with pytest.raises(ExecutionStopped):
            queued_finished.result(timeout=2)
        assert not first_finished.done()
        first.handles[0].cancel()
        assert json.loads(first_finished.result(timeout=2).output)["error_kind"] == "cancelled"
        assert json.loads(execute(executor, "'x' in locals()").output)["result"] is False
    finally:
        first.cancel()
        first_thread.join(10)
        if queued is not None:
            queued.cancel()
            queued_thread.join(10)


def test_close_cancels_active_code_and_joins_event_loop(code_runtime, tmp_path):
    executor = executor_for(code_runtime, tmp_path)
    ready = add_signal(executor)
    scope, finished, thread = start(executor, "call_tool('ready', {})\nwhile True:\n    pass")
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (code_runtime.close(), closed.set()), daemon=True)
    try:
        assert ready.wait(5)
        time.sleep(0.05)
        closer.start()
        assert closed.wait(2)
        assert json.loads(finished.result(timeout=1).output)["error_kind"] == "cancelled"
        assert not code_runtime._thread.is_alive()
        assert code_runtime._sessions == {}
        assert code_runtime._active == {}
    finally:
        scope.cancel()
        thread.join(10)
        if closer.ident is not None:
            closer.join(10)
