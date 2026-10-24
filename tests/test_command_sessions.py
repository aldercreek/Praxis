import json
import sys
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from praxis_agent.agent import Agent
from praxis_agent.execution import ExecutionStopped, Run
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.tools import ToolContext, create_default_registry, run_command, wait_command
from praxis_agent.tools.command_sessions import CommandSessionManager
from praxis_agent.planning import RunState


@pytest.fixture
def manager():
    instance = CommandSessionManager()
    try:
        yield instance
    finally:
        instance.close()


def until(predicate):
    deadline = monotonic() + 5
    while not predicate():
        assert monotonic() < deadline, "condition did not become true"
        sleep(.01)


def launch(tmp_path, manager, code, **kwargs):
    return run_command(tmp_path, [sys.executable, "-u", "-c", code],
                       manager=manager, **kwargs)


def test_yield_keeps_process_alive_and_streams_are_independent(tmp_path, manager):
    chunks = []
    code = """
import sys,time
from pathlib import Path
print('first', flush=True)
while not Path('go').exists(): time.sleep(.01)
print('second', flush=True)
print('warning', file=sys.stderr, flush=True)
"""
    first = launch(tmp_path, manager, code, yield_time_ms=200,
                   on_output=lambda stream, text: chunks.append((stream, text)))
    assert first["status"] == "running"
    session = manager._get(first["session_id"])
    until(lambda: session.outputs["stdout"].size() > 0)
    # Do not assume interpreter startup finishes inside the first wait on a busy host.
    if not first["stdout"]:
        first = wait_command(first["session_id"], 0, manager=manager,
                             on_output=lambda stream, text: chunks.append((stream, text)))
    assert first["stdout"] == "first\n"
    assert "".join(text for stream, text in chunks if stream == "stdout").replace("\r\n", "\n") == "first\n"
    assert session.process.process.poll() is None
    paths = [Path(output.writer.name) for output in session.outputs.values()]
    (tmp_path / "go").touch()
    second = wait_command(first["session_id"], 5000, manager=manager,
                          on_output=lambda stream, text: chunks.append((stream, text)))
    assert second["status"] == "exited" and second["exit_code"] == 0
    assert second["stdout"] == "second\n" and second["stderr"] == "warning\n"
    assert "".join(text for stream, text in chunks if stream == "stdout").replace("\r\n", "\n") == "first\nsecond\n"
    assert all(not path.exists() for path in paths)
    assert wait_command(first["session_id"], 0, manager=manager)["stdout"] == ""


def test_maintenance_finishes_without_a_waiter_and_retains_tail(tmp_path, manager):
    first = launch(tmp_path, manager, "import time; time.sleep(.2); print('tail')", yield_time_ms=0)
    session = manager._get(first["session_id"])
    until(lambda: session.state == "exited")
    assert session.cleaned
    assert Path(session.outputs["stdout"].writer.name).exists()
    final = wait_command(first["session_id"], 600000, manager=manager)
    assert final["stdout"] == "tail\n" and final["exit_code"] == 0


def test_normal_run_settlement_does_not_wait_for_background_command(tmp_path, manager):
    scope = Run()
    result = scope.execute(lambda: launch(tmp_path, manager, "import time; time.sleep(60)", yield_time_ms=0))
    assert scope.finished.done()
    assert manager._get(result["session_id"]).process.process.poll() is None
    manager.close()
    with pytest.raises(ValueError, match="unavailable"):
        wait_command(result["session_id"], 0, manager=manager)


def test_turn_cancel_stops_created_commands_but_preserves_older_ones(tmp_path, manager):
    older = Run().execute(lambda: launch(tmp_path, manager, "import time; time.sleep(60)", yield_time_ms=0))
    current = Run()
    created = []

    def operation():
        created.append(launch(tmp_path, manager, "import time; time.sleep(60)", yield_time_ms=0))
        current.cancel()

    with pytest.raises(ExecutionStopped):
        current.execute(operation)
    assert wait_command(created[0]["session_id"], 0, manager=manager)["status"] == "cancelled"
    assert manager._get(older["session_id"]).process.process.poll() is None


def test_stop_does_not_wait_for_the_wait_operation_lock(tmp_path, manager):
    first = launch(tmp_path, manager, "import time; time.sleep(60)", yield_time_ms=0)
    started, results = Event(), []

    def wait():
        results.append(wait_command(first["session_id"], 600000, manager=manager,
                                    check_execution=started.set))

    worker = Thread(target=wait)
    worker.start()
    try:
        assert started.wait(3)
        with pytest.raises(RuntimeError, match="busy"):
            wait_command(first["session_id"], 0, manager=manager)
        manager.stop(first["session_id"])
        worker.join(3)
        assert not worker.is_alive()
        assert results[0]["status"] == "cancelled"
    finally:
        manager.close()
        worker.join(3)


def test_unacknowledged_output_replays_until_saved(tmp_path, manager):
    result = launch(tmp_path, manager, "print('preserve me')", defer_ack=True)
    session = manager._get(result["session_id"])
    path = Path(session.outputs["stdout"].writer.name)
    assert path.exists()
    replay = wait_command(result["session_id"], 0, manager=manager, defer_ack=True)
    assert replay == result
    manager.acknowledge(result["session_id"], result)
    assert not path.exists()
    assert wait_command(result["session_id"], 0, manager=manager)["stdout"] == ""


def test_split_utf8_is_preserved_across_tool_calls(tmp_path, manager):
    code = """
import os,time
from pathlib import Path
os.write(1, bytes.fromhex('e4b8'))
Path('ready').touch()
while not Path('go').exists(): time.sleep(.01)
os.write(1, bytes.fromhex('ad') + b'\\n')
"""
    first = launch(tmp_path, manager, code, yield_time_ms=0)
    until(lambda: (tmp_path / "ready").exists())
    partial = wait_command(first["session_id"], 0, manager=manager)
    assert first["stdout"] + partial["stdout"] == ""
    (tmp_path / "go").touch()
    final = wait_command(first["session_id"], 5000, manager=manager)
    assert final["stdout"] == "中\n"


def test_output_quota_stops_unread_background_command(tmp_path):
    manager = CommandSessionManager(max_output_bytes=1024)
    try:
        result = launch(tmp_path, manager, "import os,time; os.write(1,b'x'*2000); time.sleep(60)", yield_time_ms=0)
        session = manager._get(result["session_id"])
        until(lambda: session.state == "failed")
        final = wait_command(result["session_id"], 0, manager=manager)
        assert "storage limit" in final["error"]
        assert len(final["stdout"]) <= 1024
        assert session.process.process.poll() is not None
    finally:
        manager.close()


def test_expiration_removes_unread_files_and_rejects_old_ids(tmp_path):
    manager = CommandSessionManager(retention_seconds=.2)
    try:
        result = launch(tmp_path, manager, "print('unread')", yield_time_ms=0)
        session = manager._get(result["session_id"])
        paths = [Path(output.writer.name) for output in session.outputs.values()]
        until(lambda: not manager.summaries())
        assert all(not path.exists() for path in paths)
        with pytest.raises(ValueError, match="expired"):
            wait_command(result["session_id"], 0, manager=manager)
    finally:
        manager.close()


def test_sessions_are_isolated_between_agents(tmp_path, manager):
    result = launch(tmp_path, manager, "import time; time.sleep(60)", yield_time_ms=0)
    other = CommandSessionManager()
    try:
        with pytest.raises(ValueError, match="another agent"):
            wait_command(result["session_id"], 0, manager=other)
    finally:
        other.close()


def test_cancel_returns_before_background_cleanup_finishes(tmp_path, manager, monkeypatch):
    ready, proceed, cleaning, release, cancelled = (Event() for _ in range(5))
    scope = Run()
    failures = []

    def operation():
        result = launch(tmp_path, manager, "import time; time.sleep(60)", yield_time_ms=0)
        session = manager._get(result["session_id"])
        close = session.process.close

        def slow_close():
            cleaning.set()
            assert release.wait(5)
            close()

        monkeypatch.setattr(session.process, "close", slow_close)
        ready.set()
        assert proceed.wait(5)

    def execute():
        try:
            scope.execute(operation)
        except ExecutionStopped:
            pass
        except Exception as error:
            failures.append(error)

    worker = Thread(target=execute)
    worker.start()
    stopper = Thread(target=lambda: (scope.cancel(), cancelled.set()))
    try:
        assert ready.wait(3)
        stopper.start()
        assert cancelled.wait(1), "cancel callback blocked on process cleanup"
        proceed.set()
        assert cleaning.wait(3)
        assert not scope.finished.done()
        release.set()
        worker.join(3)
        assert not worker.is_alive() and not failures
        assert isinstance(scope.finished.exception(), ExecutionStopped)
    finally:
        proceed.set()
        release.set()
        worker.join(3)
        if stopper.ident is not None:
            stopper.join(3)


def test_finished_result_ids_expire_even_without_live_sessions(tmp_path):
    manager = CommandSessionManager(retention_seconds=.2)
    try:
        result = launch(tmp_path, manager, "print('done')")
        until(lambda: result["session_id"] not in manager._finished)
        with pytest.raises(ValueError, match="expired"):
            wait_command(result["session_id"], 0, manager=manager)
    finally:
        manager.close()


@pytest.mark.parametrize("value", [-1, 600001, True, 1.5])
def test_invalid_wait_is_rejected_before_start(tmp_path, manager, value):
    with pytest.raises((ValueError, TypeError)):
        launch(tmp_path, manager, "raise RuntimeError('should not start')", yield_time_ms=value)
    assert not manager.summaries()


def test_registry_rejects_old_timeout_and_registers_wait_without_approval(tmp_path):
    registry = create_default_registry()
    context = ToolContext(tmp_path, RunState("test"))
    try:
        result = registry.execute("run_command", {"argv": [sys.executable], "timeout_seconds": 1}, context=context)
        assert not result.success and "timeout_seconds" in result.output
        assert not registry.requires_approval("wait_command")
        assert not registry.allows_programmatic("wait_command")
    finally:
        context.command_sessions.close()


def test_agent_retains_command_across_turns_and_saves_each_increment_once(tmp_path):
    from praxis_agent.sessions import SQLiteSessionStore

    requests = []
    session_id = None

    def respond(items, tools, **kwargs):
        nonlocal session_id
        requests.append(items)
        if len(requests) == 1:
            name = "run_command"
            args = {"argv": [sys.executable, "-u", "-c",
                     "import time; from pathlib import Path; print('first'); "
                     "\nwhile not Path('go').exists(): time.sleep(.01)\nprint('last')"], "yield_time_ms": 200}
        elif len(requests) == 3:
            name, args = "wait_command", {"session_id": session_id, "yield_time_ms": 5000}
        else:
            if len(requests) == 2:
                results = [item for item in items if isinstance(item, dict) and item.get("type") == "function_call_output"]
                session_id = json.loads(results[-1]["output"].split("\n", 1)[-1])["output"]["session_id"]
            return ModelResponse(({"role": "assistant", "content": "done"},), "done", ())
        call = FunctionCall(str(len(requests)), name, json.dumps(args))
        return ModelResponse(({"type": "function_call", "call_id": call.call_id,
                               "name": name, "arguments": call.arguments},), "", (call,))

    store = SQLiteSessionStore(tmp_path / "history.sqlite3")
    record = store.create_session(tmp_path, "test")
    agent = Agent(SimpleNamespace(create_response=respond), create_default_registry(), tmp_path,
                  bypass_approvals=True, session_store=store, session_id=record.id, session_model="test")
    try:
        assert agent.run("start") == "done"
        assert agent.command_sessions._get(session_id).process.process.poll() is None
        (tmp_path / "go").touch()
        assert agent.run("finish") == "done"
        results = [json.loads(item["output"])["output"] for item in store.load_conversation_items(record.id)
                   if item.get("type") == "function_call_output"]
        assert isinstance(results[1], dict), results
        assert results[0]["status"] == "running" and results[1]["status"] == "exited"
        assert "".join(result["stdout"] for result in results) == "first\nlast\n"
        assert any("Command sessions owned" in str(item) for item in requests[2])
        assert not agent.command_sessions.summaries()
    finally:
        agent.close()
