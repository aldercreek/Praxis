"""Native Python capabilities, dependency environments and lifecycle boundaries."""
import time
import zipfile
from pathlib import Path

from praxis_agent.planning import RunState
from praxis_agent.tools import ToolContext, ToolResult
from praxis_agent.tools.code import CodeConfig, CodeRuntime
from test_code_lifecycle import alive, run


def test_native_imports_absolute_paths_and_session_environment(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "code.py").write_text("raise RuntimeError('project code.py must not shadow bootstrap stdlib')\n", encoding="utf-8")
    (workspace / "pricing.py").write_text("def price(x): return x * 3\n", encoding="utf-8")
    external = tmp_path / "outside.txt"
    external.write_text("7", encoding="utf-8")
    context = ToolContext(workspace, RunState("native Python"))
    runtime = CodeRuntime(environment_dir=tmp_path / "environment")
    wheel = workspace / "session_package-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("session_package.py", "VALUE = 42\n")
        archive.writestr("session_package-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: session-package\nVersion: 1.0\n")
        archive.writestr("session_package-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr("session_package-1.0.dist-info/RECORD", "")
    try:
        result = run(runtime, context, f"""
import sys, jsonschema
from pathlib import Path
from pricing import price
value: int = 'annotations are not runtime checks'
answer = price(int(Path({str(external)!r}).read_text()))
Path({str(external)!r}).write_text(str(answer))
{{'answer': answer, 'python': sys.executable, 'prefix': sys.prefix}}
""")
        assert result["result"]["answer"] == 21
        assert Path(result["result"]["prefix"]) == tmp_path / "environment"
        assert external.read_text() == "21"
        # Model-installed packages live in the shared environment, separate memory.
        run(runtime, context, "import subprocess\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', 'session_package-1.0-py3-none-any.whl'], check=True, capture_output=True).returncode")
        other = run(runtime, context, "import sys, session_package\n{'value': session_package.VALUE, 'python': sys.executable, 'shared': 'answer' in globals()}", "other")
        assert other["result"] == {"value": 42, "python": result["result"]["python"], "shared": False}
        installed = run(runtime, context, "from pathlib import Path\nstr(Path(session_package.__file__).resolve())", "other")["result"]
        assert Path(installed).is_relative_to(tmp_path / "environment")
    finally:
        runtime.close()


def test_idle_reaper_preserves_whole_turn_then_reclaims(tmp_path):
    runtime = CodeRuntime(CodeConfig(idle_seconds=0.2))
    context = ToolContext(tmp_path, RunState("idle"))
    try:
        with runtime.agent_turn("parent"):
            run(runtime, context, "saved = 42")
            pid = runtime._workers["parent"].session.worker_pid
            time.sleep(0.5)  # Simulate model thinking between code calls.
            assert alive(pid)
            assert run(runtime, context, "saved")["result"] == 42
        deadline = time.monotonic() + 5
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not alive(pid)
        with runtime.agent_turn("parent"):
            rebuilt = run(runtime, context, "'saved' in globals()")
            assert rebuilt["worker_created"] and rebuilt["result"] is False
    finally:
        runtime.close()


def test_timeout_kills_spawned_descendants(tmp_path):
    runtime = CodeRuntime(CodeConfig(execution_seconds=1))
    context = ToolContext(tmp_path, RunState("process tree"))
    try:
        run(runtime, context, "saved = 1")
        pid = runtime._workers["parent"].session.worker_pid
        failed = run(runtime, context, """
import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path('child.pid').write_text(str(child.pid))
time.sleep(60)
""", success=False)
        assert failed["error_kind"] == "timeout" and failed["state_lost"]
        assert not alive(pid)
        assert not alive(int((tmp_path / "child.pid").read_text()))
    finally:
        runtime.close()


def test_unexpected_exit_restarts_without_replay(tmp_path):
    runtime = CodeRuntime()
    context = ToolContext(tmp_path, RunState("crash"))
    try:
        failed = run(runtime, context, "import os\nos._exit(7)", success=False)
        assert failed["state_lost"] and failed["error_kind"] == "worker_crash"
        assert run(runtime, context, "42")["result"] == 42
    finally:
        runtime.close()


def test_late_thread_callback_cannot_attach_to_next_execution(tmp_path):
    invoked = []
    context = ToolContext(tmp_path, RunState("late callback"),
                          invoke_programmatic_tool=lambda name, args: invoked.append(name) or ToolResult(True, "ok"))
    runtime = CodeRuntime()
    try:
        run(runtime, context, """
import threading, queue
release = threading.Event()
outcome = queue.Queue()
def background():
    release.wait()
    try:
        call_tool('must_not_run', {})
    except Exception as error:
        outcome.put(str(error))
    else:
        outcome.put('incorrectly dispatched')
thread = threading.Thread(target=background)
thread.start()
""")
        result = run(runtime, context, "release.set()\noutcome.get(timeout=5)")
        assert "no longer active" in result["result"]
        assert invoked == []
    finally:
        runtime.close()


def test_stream_log_is_capped_and_preview_keeps_tail(tmp_path, monkeypatch):
    from praxis_agent.tools import code_output

    monkeypatch.setattr(code_output, "_LOG_CHARS", 40_000)
    output = code_output.CodeOutput(tmp_path)
    output.append("stdout", "a" * 35_000)
    output.append("stdout", "b" * 35_000)
    output.close()
    fields = output.fields()
    reference = fields["output_files"]["stdout"]
    assert reference["complete"] is False
    assert reference["saved_chars"] == 40_000
    assert reference["total_chars"] == 70_000
    assert len(Path(reference["path"]).read_text()) == 40_000
    assert fields["stdout"].endswith("b" * 24_000)


def test_timeout_cleanup_does_not_wait_for_callback_executor_slot(tmp_path):
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from praxis_agent.execution import current_handle

    entered, cancelled = threading.Event(), threading.Event()

    def waiting_tool(name, arguments):
        unsubscribe = current_handle().on_cancel(cancelled.set)
        try:
            entered.set()
            assert cancelled.wait(5)
            return ToolResult(True, "cancelled")
        finally:
            unsubscribe()

    runtime = CodeRuntime(CodeConfig(execution_seconds=0.5))
    context = ToolContext(tmp_path, RunState("callback timeout"), invoke_programmatic_tool=waiting_tool)

    async def configure():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))

    try:
        asyncio.run_coroutine_threadsafe(configure(), runtime._ensure_loop()).result(timeout=5)
        failed = run(runtime, context, "call_tool('waiting', {})", success=False)
        assert entered.is_set() and cancelled.is_set()
        assert failed["error_kind"] == "timeout" and failed["state_lost"]
    finally:
        runtime.close()
