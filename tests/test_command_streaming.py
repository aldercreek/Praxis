import asyncio
import json
import sys
from io import StringIO
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from rich.console import Console

from praxis_agent.agent import Agent
from praxis_agent.execution import ExecutionStopped
from praxis_agent.events import (
    ToolCallPrepared,
    ToolFinished,
    ToolOutputDelta,
    ToolStarted,
    TurnCompleted,
    TurnStarted,
)
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.multi_agent.models import MultiAgentConfig
from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.terminal.app import PraxisApp
from praxis_agent.terminal.views import ToolBlock, ViewEventSink, ViewStore
from praxis_agent.tools import ToolResult, create_default_registry, run_command


def test_output_arrives_before_exit_and_decodes_split_utf8(tmp_path):
    # The child cannot finish until the live callback acknowledges its output.
    # Split a Chinese character across writes to exercise incremental decoding.
    code = """
import os, time
from pathlib import Path
data = bytes.fromhex('e4b8ade69687')
os.write(1, data[:2])
time.sleep(.25)
os.write(1, data[2:] + b' ready\\n')
deadline = time.monotonic() + 4
while not Path('ack').exists():
    if time.monotonic() > deadline:
        raise RuntimeError('no live output received')
    time.sleep(.02)
os.write(2, b'warning\\n')
os.write(1, b'done\\n')
"""
    chunks = {"stdout": [], "stderr": []}

    def receive(stream, text):
        chunks[stream].append(text)
        if "ready" in "".join(chunks["stdout"]):
            (tmp_path / "ack").touch()

    result = run_command(tmp_path, [sys.executable, "-u", "-c", code],
                         yield_time_ms=8000, on_output=receive)
    assert result["exit_code"] == 0
    assert "".join(chunks["stdout"]) == "中文 ready\ndone\n"
    assert "".join(chunks["stderr"]) == "warning\n"
    assert result["stdout"] == "中文 ready\ndone\n"


def test_large_output_does_not_move_child_write_cursor_or_lose_final_bytes(tmp_path):
    chunks = []
    code = "import os; [os.write(1, b'x' * 10000) for _ in range(30)]; os.write(1, b'END')"
    result = run_command(tmp_path, [sys.executable, "-c", code],
                         on_output=lambda stream, text: chunks.append(text))
    expected = "x" * 300000 + "END"
    assert "".join(chunks) == expected
    assert result["stdout"] == expected
    assert result["stderr"] == ""


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancellation_preserve_partial_output(tmp_path, cancel):
    seen = []

    def receive(stream, text):
        seen.append(text)

    def check():
        if cancel and seen:
            raise ExecutionStopped("cancel requested")

    code = "import os,time; os.write(1,b'partial out\\n'); os.write(2,b'partial error\\n'); time.sleep(20)"
    result = run_command(tmp_path, [sys.executable, "-u", "-c", code], yield_time_ms=1000,
                         on_output=receive, check_execution=check)
    assert result["status"] == ("cancelled" if cancel else "running")
    assert "partial out" in result["stdout"]
    assert "partial error" in result["stderr"]
    assert bool(result["error"]) == cancel
    assert "partial out" in "".join(seen)


def test_streaming_events_do_not_add_model_requests_or_history_items(tmp_path):
    calls, events = [], []
    arguments = json.dumps({"argv": [sys.executable, "-u", "-c", "print('visible output')"],
                            "workdir": ".", "yield_time_ms": 5000})
    call = FunctionCall("cmd", "run_command", arguments)

    def respond(items, tools, **kwargs):
        calls.append(items)
        if len(calls) == 1:
            return ModelResponse(({"type": "function_call", "call_id": "cmd", "name": "run_command",
                                   "arguments": arguments},), "", (call,))
        return ModelResponse(({"role": "assistant", "content": "done"},), "done", ())

    agent = Agent(model_client=SimpleNamespace(create_response=respond),
                  tool_registry=create_default_registry(), workspace=tmp_path,
                  bypass_approvals=True, event_sink=SimpleNamespace(emit=events.append))
    try:
        assert agent.run("Run the command") == "done"
        assert len(calls) == 2
        results = [item for item in calls[1] if isinstance(item, dict)
                   and item.get("type") == "function_call_output"]
        assert len(results) == 1 and "visible output" in results[0]["output"]
        assert any(isinstance(e, ToolOutputDelta) and "visible output" in e.text for e in events)
        assert not any(isinstance(item, ToolOutputDelta) for items in calls for item in items)
    finally:
        agent.close()


def test_worker_output_reaches_display_without_parent_pumping(tmp_path):
    arrived, release = Event(), Event()
    events = []

    def receive(event):
        events.append(event)
        arrived.set()

    def factory(agent_id, control, sink):
        def run(task):
            sink.emit(ToolOutputDelta("child-turn", "cmd", "stdout", "child progress"))
            release.wait(5)
            return "done"
        return SimpleNamespace(run=run, close=lambda: None)

    runtime = AgentRuntime(factory, MultiAgentConfig(), workspace=tmp_path,
                           live_output_sink=SimpleNamespace(emit_live=lambda e: receive(e) if isinstance(getattr(e, "event", None), ToolOutputDelta) else None))
    try:
        info = runtime.spawn("work", working_directory=".")
        # No pump_events call: the parent's thread may be busy executing a command.
        assert arrived.wait(2)
        assert events[0].agent_id == info["agent_id"]
        assert events[0].event.text == "child progress"
    finally:
        release.set()
        runtime.shutdown(lambda: runtime.pump_events(None))


def test_preview_is_bounded_and_worker_updates_do_not_write_terminal():
    sink, store = ViewEventSink(), ViewStore()
    worker = Thread(target=lambda: [sink.emit(ToolOutputDelta(
        "turn", "cmd", "stdout", f"line {i}\n", agent_id="worker")) for i in range(1000)])
    worker.start()
    worker.join()
    sink.drain(store)
    preview = store.get("worker")._command_output
    output = StringIO()
    console = Console(file=output, color_system=None, width=100)
    from praxis_agent.terminal.cells import tool_preview, tool_title
    block, = store.get("worker").committed_blocks
    console.print(tool_title((block,)))
    console.print(tool_preview((block,)))
    text = output.getvalue()
    assert "line 999" in text and "line 0\n" in text
    assert "omitted" in text
    assert len(text.splitlines()) == 6  # Header plus a bounded head/tail preview.
    preview.clear()
    assert not preview.outputs


def test_page_refreshes_before_command_ends_and_keeps_failure_logs(tmp_path):
    async def exercise():
        sink = ViewEventSink()
        app = PraxisApp(sink)
        code = """
import time
from pathlib import Path
print('LIVE_PROGRESS', flush=True)
deadline = time.monotonic() + 5
while not Path('ui-ack').exists():
    if time.monotonic() > deadline:
        raise RuntimeError('UI did not display progress')
    time.sleep(.02)
print('last diagnostic', flush=True)
"""
        arguments = {"argv": [sys.executable, "-u", "-c", code], "workdir": ".", "yield_time_ms": 8000}

        def execute():
            sink.emit(TurnStarted("demo", "stream output"))
            sink.emit(ToolCallPrepared("demo", "cmd", "run_command", arguments, False))
            sink.emit(ToolStarted("demo", "cmd", "run_command"))
            try:
                result = run_command(tmp_path, **arguments, on_output=lambda stream, text:
                                     sink.emit(ToolOutputDelta("demo", "cmd", stream, text)))
                sink.emit(ToolFinished("demo", "cmd", "run_command", ToolResult(
                    result["status"] == "exited", json.dumps(result), structured_output=result)))
            except Exception as error:
                sink.emit(ToolFinished("demo", "cmd", "run_command", ToolResult(False, str(error))))
            sink.emit(TurnCompleted("demo"))

        async with app.run_test(size=(100, 30)) as pilot:
            worker = Thread(target=execute)
            worker.start()
            try:
                visible = ""
                for _ in range(30):
                    await pilot.pause(0.1)
                    capture = StringIO()
                    for widget in app.query("#blocks Static"):
                        Console(file=capture, width=100).print(widget.renderable)
                    visible = capture.getvalue()
                    if "LIVE_PROGRESS" in visible:
                        break
                assert "LIVE_PROGRESS" in visible
                assert worker.is_alive()
            finally:
                (tmp_path / "ui-ack").touch()
                await asyncio.to_thread(worker.join, 10)
            assert not worker.is_alive()
            await pilot.pause(0.2)
            blocks = [b for b in app.store.views["main"].committed_blocks if isinstance(b, ToolBlock)]
            assert len(blocks) == 1
            assert "last diagnostic" in blocks[0].output
            assert not blocks[0].failed
            timed_out = {"status": "timed_out", "exit_code": 1, "stdout": "partial progress",
                         "stderr": "last error", "error": "执行超时，命令已停止。"}
            sink.emit(ToolFinished("demo", "failed", "run_command", ToolResult(
                False, json.dumps(timed_out, ensure_ascii=False), structured_output=timed_out)))
            await pilot.pause(0.2)
            failure = app.store.views["main"].committed_blocks[-1]
            assert failure.failed
            assert all(text in failure.output for text in ("执行超时", "partial progress", "last error"))
            assert failure.state == "timed_out"
    asyncio.run(exercise())
