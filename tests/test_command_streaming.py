import json
import sys
from io import StringIO
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from rich.console import Console

from praxis_agent.agent import Agent
from praxis_agent.events import ToolOutputDelta, ToolCallPrepared, ToolStarted, ToolFinished, TurnStarted, TurnCompleted
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.multi_agent.models import MultiAgentConfig
from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.terminal.command_output import CommandOutputPreview
from praxis_agent.terminal.renderer import ConsoleRenderer
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
                         timeout_seconds=8, on_output=receive)
    assert "exit_code: 0" in result
    assert "".join(chunks["stdout"]) == "中文 ready\ndone\n"
    assert "".join(chunks["stderr"]) == "warning\n"
    assert "中文 ready\ndone\n" in result


def test_large_output_does_not_move_child_write_cursor_or_lose_final_bytes(tmp_path):
    chunks = []
    code = "import os; [os.write(1, b'x' * 10000) for _ in range(30)]; os.write(1, b'END')"
    result = run_command(tmp_path, [sys.executable, "-c", code],
                         on_output=lambda stream, text: chunks.append(text))
    expected = "x" * 300000 + "END"
    assert "".join(chunks) == expected
    assert f"stdout:\n{expected}\nstderr:\n<empty>" in result


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancellation_preserve_partial_output(tmp_path, cancel):
    seen = []

    def receive(stream, text):
        seen.append(text)

    def check():
        if cancel and seen:
            raise RuntimeError("cancel requested")

    code = "import os,time; os.write(1,b'partial out\\n'); os.write(2,b'partial error\\n'); time.sleep(20)"
    with pytest.raises(RuntimeError if cancel else TimeoutError) as failure:
        run_command(tmp_path, [sys.executable, "-u", "-c", code], timeout_seconds=1,
                    on_output=receive, check_execution=check)
    assert "partial out" in str(failure.value)
    assert "partial error" in str(failure.value)
    assert "partial out" in "".join(seen)


def test_streaming_events_do_not_add_model_requests_or_history_items(tmp_path):
    calls, events = [], []
    arguments = json.dumps({"argv": [sys.executable, "-u", "-c", "print('visible output')"],
                            "relative_path": ".", "timeout_seconds": 5})
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
                           live_output_sink=SimpleNamespace(emit=receive))
    try:
        info = runtime.spawn("work", working_directory=".")
        # No pump_events call: the parent's thread may be busy executing a command.
        assert arrived.wait(2)
        assert events[0].agent_id == info["agent_id"]
        assert events[0].text == "child progress"
    finally:
        release.set()
        runtime.shutdown(lambda: runtime.pump_events(None))


def test_preview_is_bounded_and_worker_updates_do_not_write_terminal():
    preview = CommandOutputPreview()
    worker = Thread(target=lambda: [preview.update(ToolOutputDelta(
        "turn", "cmd", "stdout", f"line {i}\n", agent_id="worker")) for i in range(1000)])
    worker.start()
    worker.join()
    output = StringIO()
    console = Console(file=output, color_system=None, width=100)
    for block in preview.render():
        console.print(block)
    text = output.getvalue()
    assert "line 999" in text and "line 0\n" not in text
    assert len(text.splitlines()) == 7  # Header plus six tail lines.
    preview.clear()
    assert preview.render() == []


def test_terminal_refreshes_before_command_ends_and_shows_timeout_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    capture = StringIO()
    renderer = ConsoleRenderer(Console(file=capture, force_terminal=True, width=90, color_system=None))
    code = "import os,time; os.write(1,b'LIVE_PROGRESS\\n'); time.sleep(1.2); os.write(2,b'last diagnostic\\n')"
    arguments = {"argv": [sys.executable, "-u", "-c", code], "relative_path": ".", "timeout_seconds": 5}
    visible_before_end = []

    def receive(stream, text):
        if "last diagnostic" in text:
            visible_before_end.append("LIVE_PROGRESS" in capture.getvalue())
        renderer.emit(ToolOutputDelta("demo", "cmd", stream, text))

    renderer.emit(TurnStarted("demo", "stream output"))
    try:
        renderer.emit(ToolCallPrepared("demo", "cmd", "run_command", arguments, False))
        renderer.emit(ToolStarted("demo", "cmd", "run_command"))
        result = run_command(tmp_path, **arguments, on_output=receive)
        renderer.emit(ToolOutputDelta("demo", "cmd", "", "", finished=True))
        renderer.emit(ToolFinished("demo", "cmd", "run_command", ToolResult(True, result)))
    finally:
        renderer.emit(TurnCompleted("demo"))
    assert visible_before_end == [True]
    assert "last diagnostic" in capture.getvalue()

    failure = StringIO()
    renderer = ConsoleRenderer(Console(file=failure, width=100, color_system=None))
    renderer.emit(ToolCallPrepared("x", "cmd", "run_command", arguments, False))
    renderer.emit(ToolFinished("x", "cmd", "run_command", ToolResult(
        False, "TimeoutError: command timed out\nstdout:\npartial progress\nstderr:\nlast error")))
    assert all(text in failure.getvalue() for text in ("TimeoutError", "partial progress", "last error"))
