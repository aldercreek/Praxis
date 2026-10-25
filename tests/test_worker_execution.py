import json
import os
import sys
from pathlib import Path
from threading import Event, Thread, get_ident
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from praxis_agent.factory import create_default_agent
from praxis_agent.mcp import MCPConnections
from praxis_agent.config import ModelConfig
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.multi_agent.models import (
    AgentStatus,
    ExecutionControl,
    MultiAgentConfig,
)
from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.planning import RunState
from praxis_agent.tools import ToolContext
from praxis_agent.tools.command import run_command


def wait_status(runtime, agent_id, state=AgentStatus.IDLE):
    with runtime.condition:
        assert runtime.condition.wait_for(lambda: runtime.handles[agent_id].status == state, timeout=10)


def response(text="done", tool=None, arguments=None):
    if tool:
        call = FunctionCall("call-" + tool, tool, json.dumps(arguments))
        return ModelResponse(({"type": "function_call", "call_id": call.call_id,
                               "name": tool, "arguments": call.arguments},), "", (call,))
    return ModelResponse(({"role": "assistant", "content": text},), text, ())


def test_default_agent_binds_directory_delivers_resumes_and_retains_files(tmp_path, monkeypatch):
    clients = []

    class Client:
        def __init__(self, config, instructions, **kwargs):
            self.instructions = instructions
            self.calls = 0
            self.inputs = []
            clients.append(self)

        def close(self):
            pass

        def create_response(self, input_items, tools, **kwargs):
            self.inputs.append(str(input_items))
            self.calls += 1
            if "Complete the work assigned by the main agent" in self.instructions and "run_command" in self.instructions:
                if self.calls == 1:
                    return response(tool="write_file", arguments={"relative_path": "results/report.txt", "content": "initial report\n"})
                if self.calls == 2:
                    return response(tool="run_command", arguments={"argv": [sys.executable, "-c",
                        "from pathlib import Path; p=Path('results/report.txt'); assert p.read_text() == 'initial report\\n'; print('verified')"],
                        "workdir": ".", "yield_time_ms": 10000})
                return response("Report complete: results/report.txt; command verified.")
            return response("Reviewed worker deliveries.")

    monkeypatch.setattr("praxis_agent.factory.OpenAIModelClient", Client)
    monkeypatch.setattr("praxis_agent.factory.MCPConnections", lambda: MCPConnections(tmp_path / "absent-mcp.json"))
    monkeypatch.setattr(ModelConfig, "from_env", classmethod(lambda cls, **kwargs: ModelConfig(api_key="test", model="test")))
    parent_path = tmp_path / "project"
    parent_path.mkdir()
    parent = create_default_agent(parent_path, session_database_path=tmp_path / "state.sqlite3", bypass_approvals=True)
    context = ToolContext(parent_path, RunState("test"))
    supplied = tmp_path / "worker"
    supplied.mkdir()
    runtime = parent.collaboration
    try:
        spawned = parent.tool_registry.execute("spawn_agent", {"task": "Generate and verify report", "working_directory": str(supplied)}, context=context)
        assert spawned.success, spawned.output
        info = json.loads(spawned.output)
        assert info["can_run_commands"] is True
        assert info["can_write_files"] is True
        agent_id, root = info["agent_id"], Path(info["workspace"])
        wait_status(runtime, agent_id)
        worker = runtime.handles[agent_id].agent
        assert "run_command" in worker.tool_registry.names
        assert "submit_result" not in worker.tool_registry.names
        assert worker.workspace == root
        assert str(root).replace("\\", "\\\\") in worker.model_client.instructions
        assert "verified" in worker.model_client.inputs[-1]
        assert not (parent_path / "results/report.txt").exists()
        parent.run("Review deliveries")
        saved = str(parent.session_store.load_conversation_items(parent.session_id))
        assert "Working directory" in saved and "Report complete" in saved
        runtime.send_many([{"agent_id": agent_id, "content": "Double-check the report"}])
        wait_status(runtime, agent_id)
        assert runtime.handles[agent_id].agent is worker
        assert "initial report" in worker.model_client.inputs[-1]
        assert "apply_agent_changes" not in parent.tool_registry.names
        assert "inspect_agent_changes" not in parent.tool_registry.names
        read = parent.tool_registry.execute("read_file", {"relative_path": str(root / "results/report.txt")}, context=context)
        assert read.success and "initial report" in read.output
        second_root = tmp_path / "second-worker"
        second_root.mkdir()
        second_info = runtime.spawn("Review report", working_directory=str(second_root))
        assert second_info["can_run_commands"] is True
        assert second_info["can_write_files"] is True
        assert "mode" not in second_info
        second = second_info["agent_id"]
        wait_status(runtime, second)
        names = runtime.handles[second].agent.tool_registry.names
        assert {"run_command", "write_file", "edit_file", "run_code", "send_message"} <= set(names)
        assert "spawn_agent" not in names
        runtime.close_agent(agent_id)
        assert (root / "results/report.txt").exists()
        old_session = parent.session_id
        parent.clear_history()
        next_root = tmp_path / "next-worker"
        next_root.mkdir()
        next_info = runtime.spawn("Generate another report", working_directory=str(next_root))
        wait_status(runtime, next_info["agent_id"])
        assert parent.session_id != old_session
        assert Path(next_info["workspace"]) == next_root
        assert (root / "results/report.txt").exists()
    finally:
        parent.close()
    assert (root / "results/report.txt").exists()


def test_close_running_worker_cancels_and_preserves_files(tmp_path):
    parent = tmp_path / "project"
    parent.mkdir()
    worker_path = tmp_path / "worker"
    worker_path.mkdir()
    started = Event()

    def factory(agent_id, control, sink):
        def run(task):
            root = runtime.handles[agent_id].workspace
            (root / "partial.txt").write_text("unfinished")
            started.set()
            while True:
                control.check()
                control.cancelled.wait(0.05)
        return SimpleNamespace(run=run, close=lambda: None)

    runtime = AgentRuntime(factory, MultiAgentConfig(), workspace=parent)
    try:
        info = runtime.spawn("long task", working_directory=str(worker_path))
        assert started.wait(5)
        assert runtime.close_agent(info["agent_id"])["status"] == "closing"
        wait_status(runtime, info["agent_id"], AgentStatus.CLOSED)
        assert (Path(info["workspace"]) / "partial.txt").read_text() == "unfinished"
    finally:
        runtime.shutdown(lambda: None)


def test_worker_approval_reaches_ui_callback_without_parent_pump(tmp_path):
    parent_thread = get_ident()
    observed = []

    def approve(name, arguments):
        observed.append((get_ident(), name, arguments))
        return True

    def factory(agent_id, control, sink):
        def run(task):
            assert runtime.request_approval(agent_id, "run_command", {"argv": ["test"]})
            return "approved"
        return SimpleNamespace(run=run, close=lambda: None)

    runtime = AgentRuntime(factory, MultiAgentConfig(), approve_tool_call=approve)
    try:
        agent_id = runtime.spawn("approval", working_directory=".")["agent_id"]
        wait_status(runtime, agent_id)
        assert observed[0][0] != parent_thread
        assert observed[0][2]["agent_id"] == agent_id
    finally:
        runtime.shutdown(lambda: runtime.pump_events(None))


def test_parent_shutdown_cancels_running_command_and_queued_worker(tmp_path):
    parent = tmp_path / "project"
    parent.mkdir()
    worker_path = tmp_path / "worker"
    worker_path.mkdir()
    started = Event()

    def factory(agent_id, control, sink):
        def run(task):
            root = runtime.handles[agent_id].workspace
            started.set()
            return run_command(root, [sys.executable, "-c",
                "import time; from pathlib import Path; Path('partial.txt').write_text('saved'); time.sleep(120)"],
                yield_time_ms=120000, check_execution=control.check)
        return SimpleNamespace(run=run, close=lambda: None)

    runtime = AgentRuntime(factory, MultiAgentConfig(max_workers=1), workspace=parent)
    first = runtime.spawn("long command", working_directory=str(worker_path))
    try:
        assert started.wait(5)
        queued_path = tmp_path / "queued"
        queued_path.mkdir()
        second = runtime.spawn("queued task", working_directory=str(queued_path))
        deadline = monotonic() + 5
        partial = Path(first["workspace"]) / "partial.txt"
        while not partial.exists() and monotonic() < deadline:
            sleep(0.02)
        assert partial.exists()
        before = monotonic()
        runtime.shutdown(lambda: None)
        assert monotonic() - before < 5
        assert all(h.status == AgentStatus.CLOSED for h in runtime.handles.values())
        assert partial.read_text() == "saved"
        assert not (Path(second["workspace"]) / "partial.txt").exists()
    finally:
        runtime.shutdown(lambda: None)


def process_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.parametrize("ending", ["cancel", "timeout", "return"])
@pytest.mark.parametrize("mode", ["argv", "command"])
def test_command_stops_descendants_on_every_exit(tmp_path, ending, mode):
    control = ExecutionControl()
    errors = []
    results = []
    # Publish readiness only after the PID is complete; creation alone races readers.
    child = ("import os,time; from pathlib import Path; "
             "Path('child.pid.tmp').write_text(str(os.getpid())); "
             "os.replace('child.pid.tmp', 'child.pid'); time.sleep(120)")
    launcher = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
        "\nwhile not Path('child.pid').exists(): time.sleep(0.01)\n"
        + ("" if ending == "return" else "time.sleep(120)")
    )
    arguments = {"argv": [sys.executable, "-c", launcher]}
    if mode == "command":
        from praxis_agent.runtime import select_command_shell
        shell = select_command_shell()
        if shell is None:
            pytest.skip("no supported shell installed")
        (tmp_path / "launcher.py").write_text(launcher, encoding="utf-8")
        if shell.name in {"pwsh", "powershell"}:
            executable = sys.executable.replace("'", "''")
            arguments = {"command": f"& '{executable}' launcher.py\nexit $LASTEXITCODE"}
        else:
            import shlex
            arguments = {"command": f"{shlex.quote(sys.executable)} launcher.py"}

    def run():
        try:
            results.append(run_command(tmp_path, **arguments,
                                       yield_time_ms=1000 if ending == "timeout" else 10000,
                                       check_execution=control.check))
        except Exception as error:
            errors.append(error)

    thread = Thread(target=run)
    thread.start()
    try:
        deadline = monotonic() + 5
        while not (tmp_path / "child.pid").exists() and monotonic() < deadline:
            sleep(0.02)
        assert (tmp_path / "child.pid").exists(), errors
        pid = int((tmp_path / "child.pid").read_text())
        if ending == "cancel":
            control.cancelled.set()
        thread.join(5)
        assert not thread.is_alive()
        if ending == "timeout":
            assert results[0]["status"] == "running"
            assert process_alive(pid)
            from praxis_agent.tools.command_sessions import default_manager
            default_manager().stop(results[0]["session_id"])
        deadline = monotonic() + 2
        while process_alive(pid) and monotonic() < deadline:
            sleep(0.02)
        assert not process_alive(pid)
        assert not errors
        assert results[0]["status"] == {"cancel": "cancelled", "timeout": "running", "return": "exited"}[ending]
    finally:
        control.cancelled.set()
        thread.join(5)
