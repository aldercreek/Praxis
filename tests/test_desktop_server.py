import json
import subprocess
import sys
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic

import pytest

from praxis_agent.agent import Agent
from praxis_agent.desktop.event_adapter import DesktopEventSink, Transcript
from praxis_agent.desktop.protocol import OutputChannel
from praxis_agent.desktop.session_manager import SessionManager, SessionRuntime
from praxis_agent.desktop.server import DesktopServer
from praxis_agent.events import AssistantTextDelta, ChildAgentEvent, ModelRequestCompleted, ModelRequestStarted
from praxis_agent.execution import current_handle
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.sessions import SQLiteSessionStore
from praxis_agent.tools import create_default_registry


def answer(text):
    return ModelResponse(({"role": "assistant", "content": text},), text, ())


class Lines:
    def __init__(self):
        self.queue = Queue()

    def __iter__(self):
        while (line := self.queue.get()) is not None:
            yield line


class Capture:
    def __init__(self):
        self.queue = Queue()

    def write(self, line):
        self.queue.put(json.loads(line))

    def flush(self):
        pass


class Harness:
    def __init__(self, tmp_path, clients, *, bypass_approvals=False):
        self.input, self.output = Lines(), Capture()
        self.messages, self.created = [], []
        self.clients = iter(clients)

        def factory(workspace, *, event_sink, approve_tool_call, bypass_approvals, resume_session_id, session_database_path):
            store = SQLiteSessionStore(session_database_path)
            record = store.get_session(resume_session_id)
            store.recover_interrupted_runs(record.id)
            agent = Agent(next(self.clients), create_default_registry(), workspace,
                          event_sink=event_sink, approve_tool_call=approve_tool_call, bypass_approvals=bypass_approvals,
                          session_store=store, session_id=record.id, session_model=record.model)
            agent.restore_history(store.load_conversation_items(record.id))
            self.created.append(agent)
            return agent

        self.server = DesktopServer(self.input, self.output, database_path=tmp_path / "desktop.sqlite3",
                                    agent_factory=factory, bypass_approvals=bypass_approvals)
        self.thread = Thread(target=self.server.run, daemon=True)
        self.thread.start()
        self.next_id = 0
        self.workspace = str(tmp_path)

    def send(self, method, **params):
        self.next_id += 1
        request_id = str(self.next_id)
        self.input.queue.put(json.dumps({"id": request_id, "method": method, "params": params}) + "\n")
        return request_id

    def wait(self, predicate, timeout=5):
        deadline = monotonic() + timeout
        while True:
            found = next((message for message in self.messages if predicate(message)), None)
            if found is not None:
                return found
            try:
                message = self.output.queue.get(timeout=max(0.01, deadline - monotonic()))
            except Empty:
                pytest.fail(f"Message not received; last messages: {self.messages[-8:]}")
            self.messages.append(message)
            if monotonic() > deadline:
                pytest.fail(f"Timed out; last messages: {self.messages[-8:]}")

    def request(self, method, **params):
        request_id = self.send(method, **params)
        return self.wait(lambda message: message.get("id") == request_id)

    def create(self):
        response = self.request("session.create", workspace=self.workspace)
        assert "result" in response, response
        return response["result"]["session"]["id"]

    def settled(self, session_id, turn_id):
        return self.wait(lambda message: message.get("event") == "TurnSettled"
                         and message.get("session_id") == session_id and message.get("turn_id") == turn_id)

    def close(self):
        self.input.queue.put(None)
        self.thread.join(5)
        assert not self.thread.is_alive(), "EOF did not shut down the server"


class BlockingClient:
    def __init__(self, text):
        self.text = text
        self.started, self.release = Event(), Event()
        self.received = []

    def create_response(self, input_items, tools, *, on_text_delta=None):
        self.received.append(input_items)
        handle = current_handle()
        unsubscribe = handle.on_cancel(self.release.set)
        self.started.set()
        try:
            assert self.release.wait(5)
            handle.check()
            on_text_delta(self.text)
            return answer(self.text)
        finally:
            unsubscribe()


def test_background_command_management_is_scoped_to_its_chat(tmp_path):
    from praxis_agent.tools import run_command, wait_command

    harness = Harness(tmp_path, [BlockingClient("unused"), BlockingClient("unused")])
    try:
        first, other = harness.create(), harness.create()
        manager = harness.created[0].command_sessions
        command = run_command(tmp_path, [sys.executable, "-c", "import time; time.sleep(60)"],
                              yield_time_ms=0, manager=manager)
        command_id = command["session_id"]
        listed = harness.request("command.list", session_id=first)["result"]["commands"]
        assert listed[0]["session_id"] == command_id
        assert harness.request("command.list", session_id=other)["result"]["commands"] == []
        assert "error" in harness.request("command.stop", session_id=other, command_id=command_id)
        assert harness.request("command.stop", session_id=first, command_id=command_id)["result"]["stopped"]
        assert wait_command(command_id, 0, manager=manager)["status"] == "cancelled"
    finally:
        harness.close()


def test_concurrent_sessions_are_routed_and_live_open_reuses_runtime(tmp_path):
    a, b = BlockingClient("answer A"), BlockingClient("answer B")
    harness = Harness(tmp_path, [a, b])
    try:
        session_a, session_b = harness.create(), harness.create()
        turn_a = harness.request("turn.start", session_id=session_a, text="task A")["result"]["turn_id"]
        turn_b = harness.request("turn.start", session_id=session_b, text="task B")["result"]["turn_id"]
        assert a.started.wait(2) and b.started.wait(2)
        duplicate = harness.request("turn.start", session_id=session_a, text="invalid second run")
        assert duplicate["error"]["code"] == "session_busy"
        opened = harness.request("session.open", session_id=session_a)["result"]
        assert opened["session"]["status"] == "running"
        assert opened["messages"][0]["content"] == "task A"
        assert len(harness.created) == 2
        assert harness.request("turn.input", session_id=session_a, text="additional A")["result"] == {"queued": True}
        b.release.set()
        harness.settled(session_b, turn_b)
        assert not a.release.is_set()
        a.release.set()
        harness.settled(session_a, turn_a)
        snapshot = harness.request("session.open", session_id=session_a)["result"]
        assert [message["content"] for message in snapshot["messages"] if message["role"] == "user"] == ["task A", "additional A"]
        assert all(message["session_id"] == session_a for message in harness.messages
                   if message.get("event") == "AssistantTextDelta" and message["data"]["text"] == "answer A")
        sequence = [message["sequence"] for message in harness.messages if message.get("session_id") == session_a]
        assert sequence == sorted(set(sequence))
        accepted = next(index for index, message in enumerate(harness.messages)
                        if message.get("result", {}).get("turn_id") == turn_a)
        started = next(index for index, message in enumerate(harness.messages)
                       if message.get("event") == "TurnStarted" and message.get("turn_id") == turn_a)
        assert accepted < started
    finally:
        a.release.set()
        b.release.set()
        harness.close()


def test_stop_waits_for_cleanup_and_accepts_new_turn_after_settlement(tmp_path):
    cleaning, release = Event(), Event()

    class Client(BlockingClient):
        def create_response(self, *args, **kwargs):
            try:
                return super().create_response(*args, **kwargs)
            finally:
                cleaning.set()
                assert release.wait(5)

    client = Client("done")
    harness = Harness(tmp_path, [client])
    try:
        session = harness.create()
        turn = harness.request("turn.start", session_id=session, text="first")["result"]["turn_id"]
        assert client.started.wait(2)
        assert harness.request("turn.stop", session_id=session)["result"] == {"stopping": True}
        assert cleaning.wait(2)
        assert harness.request("session.open", session_id=session)["result"]["session"]["status"] == "stopping"
        assert harness.request("turn.start", session_id=session, text="too early")["error"]["code"] == "session_busy"
        assert harness.request("initialize")["result"]["protocol_version"] == 1
        release.set()
        assert harness.settled(session, turn)["data"]["status"] == "cancelled"
        next_turn = harness.request("turn.start", session_id=session, text="second")["result"]["turn_id"]
        assert harness.settled(session, next_turn)["data"]["status"] == "idle"
    finally:
        release.set()
        harness.close()


@pytest.mark.parametrize("decision", [True, False, "stop", "eof"])
def test_approval_response_stop_and_eof_are_responsive(tmp_path, decision):
    class Client:
        calls = 0

        def create_response(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                call = FunctionCall("write-1", "run_command", json.dumps({
                    "argv": [sys.executable, "-c", "from pathlib import Path; Path('result.txt').write_text('approved')"],
                }))
                return ModelResponse(({"type": "function_call", "call_id": call.call_id,
                                       "name": call.name, "arguments": call.arguments},), "", (call,))
            return answer("done")

    harness = Harness(tmp_path, [Client()])
    try:
        session = harness.create()
        turn = harness.request("turn.start", session_id=session, text="write file")["result"]["turn_id"]
        approval = harness.wait(lambda message: message.get("event") == "ApprovalPending")
        approval_id = approval["data"]["approval_id"]
        snapshot = harness.request("session.open", session_id=session)["result"]
        assert snapshot["approvals"][0]["approval_id"] == approval_id
        assert harness.request("initialize")["result"]["protocol_version"] == 1
        if decision == "eof":
            harness.close()
        elif decision == "stop":
            harness.request("turn.stop", session_id=session)
            assert harness.settled(session, turn)["data"]["status"] == "cancelled"
        else:
            assert harness.request("approval.respond", session_id=session,
                                   approval_id=approval_id, approved=decision)["result"] == {}
            assert harness.settled(session, turn)["data"]["status"] == "idle"
        assert (tmp_path / "result.txt").exists() is (decision is True)
        snapshot = harness.server.manager.sessions[session].snapshot()
        tool = next(message for message in snapshot["messages"] if message["role"] == "tool")
        expected = "completed" if decision is True else "denied" if decision is False else "cancelled"
        assert tool["status"] == expected
    finally:
        if harness.thread.is_alive():
            harness.close()


def test_desktop_bypass_executes_tools_without_approval_in_new_and_reopened_sessions(tmp_path):
    class Client:
        calls = 0

        def create_response(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                call = FunctionCall("write-bypass", "run_command", json.dumps({
                    "argv": [sys.executable, "-c", "from pathlib import Path; Path('bypass.txt').write_text('executed')"],
                }))
                return ModelResponse(({"type": "function_call", "call_id": call.call_id,
                                       "name": call.name, "arguments": call.arguments},), "", (call,))
            return answer("done")

    session = None
    for _ in range(2):
        harness = Harness(tmp_path, [Client()], bypass_approvals=True)
        try:
            assert harness.request("initialize")["result"]["bypass_approvals"] is True
            if session is None:
                session = harness.create()
            else:
                assert "result" in harness.request("session.open", session_id=session)
            turn = harness.request("turn.start", session_id=session, text="write file")["result"]["turn_id"]
            assert harness.settled(session, turn)["data"]["status"] == "idle"
            assert not any(message.get("event") == "ApprovalPending" for message in harness.messages)
            assert (tmp_path / "bypass.txt").read_text() == "executed"
            (tmp_path / "bypass.txt").unlink()
        finally:
            harness.close()


def test_malformed_input_and_unknown_methods_do_not_kill_receiver(tmp_path, monkeypatch):
    monkeypatch.delenv("ADA_API_KEY", raising=False)
    harness = Harness(tmp_path, [])
    try:
        harness.input.queue.put("{malformed\n")
        error = harness.wait(lambda message: message.get("error", {}).get("code") == "parse_error")
        assert error["id"] is None
        assert harness.request("not.a.method")["error"]["code"] == "method_not_found"
        assert harness.request("initialize")["result"]["configured"] is False
        assert harness.request("shutdown")["result"] == {}
        harness.thread.join(3)
        assert not harness.thread.is_alive()
    finally:
        if harness.thread.is_alive():
            harness.close()


def test_event_adapter_omits_prompts_and_does_not_duplicate_child_events():
    received = []

    class Runtime:
        def publish(self, *event):
            received.append(event)

    sink = DesktopEventSink(Runtime())
    child = ChildAgentEvent("parent-turn", "worker-1", AssistantTextDelta("child-turn", 1, "hello"))
    sink.emit_live(child)
    sink.emit_deferred(child)
    sink.emit(ModelRequestStarted("turn", 1, ({"role": "developer", "content": "private prompt"},)))
    sink.emit(ModelRequestCompleted("turn", 1, answer("public answer")))
    assert len(received) == 3
    assert received[0] == ("AssistantTextDelta", "parent-turn", "worker-1",
                           {"request_number": 1, "text": "hello", "child_turn_id": "child-turn"})
    assert "private prompt" not in json.dumps(received)
    assert "response" not in received[-1][-1]


def test_transcript_restores_saved_tools_and_text():
    history = [
        {"role": "user", "content": "read"},
        {"type": "function_call", "call_id": "c", "name": "read_file", "arguments": '{"path":"a.txt"}'},
        {"type": "function_call_output", "call_id": "c", "output": '{"success":true,"output":"hello"}'},
        {"role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
    ]
    messages = Transcript(history).messages
    assert [message["role"] for message in messages] == ["user", "tool", "assistant"]
    assert messages[1]["content"] == "hello"
    assert messages[1]["arguments"] == {"path": "a.txt"}
    assert messages[2]["content"] == "done"


def test_child_response_numbers_can_restart_within_a_parent_turn():
    transcript = Transcript()
    for child_turn, text in [("child-1", "first result"), ("child-2", "second result")]:
        transcript.consume("AssistantTextDelta", "parent-turn", "worker-1", {
            "request_number": 1, "text": text, "child_turn_id": child_turn,
        })
    assert [message["content"] for message in transcript.messages] == ["first result", "second result"]
    assert [message["id"] for message in transcript.messages] == [
        "assistant:parent-turn:worker-1:child-1:1", "assistant:parent-turn:worker-1:child-2:1",
    ]


def test_slow_session_initialization_does_not_block_stop(tmp_path):
    client = BlockingClient("first")
    harness = Harness(tmp_path, [client, BlockingClient("second")])
    setup_started, setup_release = Event(), Event()
    try:
        session = harness.create()
        turn = harness.request("turn.start", session_id=session, text="first")["result"]["turn_id"]
        assert client.started.wait(2)
        original_factory = harness.server.manager.agent_factory

        def slow_factory(*args, **kwargs):
            setup_started.set()
            assert setup_release.wait(5)
            return original_factory(*args, **kwargs)

        harness.server.manager.agent_factory = slow_factory
        create_id = harness.send("session.create", workspace=harness.workspace)
        assert setup_started.wait(2)
        assert harness.request("turn.stop", session_id=session)["result"]["stopping"] is True
        assert harness.settled(session, turn)["data"]["status"] == "cancelled"
        assert not setup_release.is_set()
        setup_release.set()
        harness.wait(lambda message: message.get("id") == create_id)
    finally:
        setup_release.set()
        harness.close()


def test_session_reopens_persisted_history_after_server_restart(tmp_path):
    client = BlockingClient("persisted answer")
    client.release.set()
    first = Harness(tmp_path, [client])
    session = first.create()
    turn = first.request("turn.start", session_id=session, text="saved question")["result"]["turn_id"]
    first.settled(session, turn)
    first.close()

    second = Harness(tmp_path, [BlockingClient("continued")])
    try:
        result = second.request("session.open", session_id=session)["result"]
        assert [message["content"] for message in result["messages"]] == ["saved question", "persisted answer"]
        assert result["session"]["status"] == "idle"
    finally:
        second.close()


@pytest.mark.parametrize("bypass", [False, True])
def test_module_entrypoint_emits_only_json_and_exits_on_eof(tmp_path, bypass):
    result = subprocess.run(
        [sys.executable, "-m", "praxis_agent.desktop", "--stdio", "--database", str(tmp_path / "stdio.sqlite3"),
         *(["--bypass"] if bypass else [])],
        input='{"id":"init","method":"initialize","params":{}}\n',
        capture_output=True, text=True, encoding="utf-8", timeout=15,
    )
    assert result.returncode == 0, result.stderr
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    assert messages[0]["id"] == "init"
    assert messages[0]["result"]["protocol_version"] == 1
    assert messages[0]["result"]["bypass_approvals"] is bypass


def test_cancelled_supplementary_input_stays_paused_until_explicitly_resubmitted(tmp_path):
    client = BlockingClient("done")
    harness = Harness(tmp_path, [client])
    try:
        session = harness.create()
        turn = harness.request("turn.start", session_id=session, text="first")["result"]["turn_id"]
        assert client.started.wait(2)
        harness.request("turn.input", session_id=session, text="do this later")
        harness.request("turn.stop", session_id=session)
        harness.settled(session, turn)
        paused = harness.wait(lambda message: message.get("event") == "InputsPaused")
        assert paused["data"] == {"texts": ["do this later"]}
        snapshot = harness.request("session.open", session_id=session)["result"]
        assert snapshot["pending_inputs"] == ["do this later"]
        next_turn = harness.request("turn.start", session_id=session, text="different task")["result"]["turn_id"]
        harness.settled(session, next_turn)
        assert not any(isinstance(item, dict) and item.get("content") == "do this later" for item in client.received[-1])
        assert harness.request("session.open", session_id=session)["result"]["pending_inputs"] == ["do this later"]
        resumed = harness.request("turn.start", session_id=session, text="do this later")["result"]["turn_id"]
        harness.settled(session, resumed)
        assert harness.request("session.open", session_id=session)["result"]["pending_inputs"] == []
    finally:
        harness.close()


class BlockedOutput:
    def __init__(self):
        self.started, self.release = Event(), Event()

    def write(self, text):
        self.started.set()
        self.release.wait(5)

    def flush(self):
        pass


def test_output_backpressure_fails_connection_and_close_is_bounded():
    stream = BlockedOutput()
    failed = Event()
    output = OutputChannel(stream, capacity=1, send_timeout=0.1, on_failure=failed.set)
    try:
        assert output.send({"event": "first"})
        assert stream.started.wait(1)
        assert output.send({"event": "second"})
        start = monotonic()
        assert output.send({"event": "terminal"}) is False
        assert failed.is_set()
        output.close()
        assert monotonic() - start < 0.8
    finally:
        stream.release.set()
        output.close()


def test_blocked_stdout_cancels_agent_and_shuts_down_without_stdin_eof(tmp_path):
    harness = Harness(tmp_path, [BlockingClient("answer")])
    blocked = BlockedOutput()
    try:
        session = harness.create()
        harness.server.output.stream = blocked
        harness.server.output.queue.maxsize = 2
        harness.server.output.send_timeout = 0.1
        harness.send("turn.start", session_id=session, text="must be cancelled")
        assert blocked.started.wait(1)
        harness.thread.join(2)
        assert harness.server.output.failed.is_set()
        assert not harness.thread.is_alive(), "Transport failure must wake the lifecycle owner even while stdin stays open"
        assert harness.created[0]._closed
        assert harness.server.manager.sessions[session].run is None
    finally:
        blocked.release.set()
        harness.input.queue.put(None)
        if harness.thread.is_alive():
            harness.close()


def test_failed_cleanup_still_releases_concurrent_session_openers(tmp_path, monkeypatch):
    factory_entered, factory_release, waiter_entered = Event(), Event(), Event()

    def failing_factory(*args, **kwargs):
        factory_entered.set()
        assert factory_release.wait(3)
        raise ValueError("initialization failed")

    def failing_cleanup(self):
        raise RuntimeError("cleanup failed")

    manager = SessionManager(lambda message: None, database_path=tmp_path / "failed.sqlite3",
                             agent_factory=failing_factory)
    record = manager.store.create_session(tmp_path, "test")
    errors = []

    def open_session():
        try:
            manager._load(record)
        except Exception as error:
            errors.append(str(error))

    monkeypatch.setattr(SessionRuntime, "close", failing_cleanup)
    owner = Thread(target=open_session, daemon=True)
    owner.start()
    assert factory_entered.wait(2)
    pending = manager.loading[record.id]
    original_result = pending.result

    def waiting_result(*args, **kwargs):
        waiter_entered.set()
        return original_result(*args, **kwargs)

    pending.result = waiting_result
    waiter = Thread(target=open_session, daemon=True)
    waiter.start()
    try:
        assert waiter_entered.wait(2)
        factory_release.set()
        owner.join(2)
        waiter.join(2)
        assert not owner.is_alive() and not waiter.is_alive()
        assert pending.done()
        assert errors == ["initialization failed", "initialization failed"]
    finally:
        factory_release.set()


@pytest.mark.parametrize("failure", ["multi_agent_config", "mcp_constructor"])
def test_factory_initialization_failures_do_not_leak_model_clients(tmp_path, monkeypatch, failure):
    import praxis_agent.agent as agent_module

    clients = []

    class Client:
        def __init__(self, *args, **kwargs):
            self.closed = False
            clients.append(self)

        def close(self):
            self.closed = True

    def invalid_mcp():
        raise ValueError("invalid MCP configuration")

    monkeypatch.setenv("ADA_API_KEY", "test-key")
    monkeypatch.setattr(agent_module, "OpenAIModelClient", Client)
    monkeypatch.setattr(agent_module, "discover_skills", lambda workspace: [])
    monkeypatch.setattr(agent_module, "MCPConnections", invalid_mcp)
    if failure == "multi_agent_config":
        monkeypatch.setenv("PRAXIS_AGENT_MAX_WORKERS", "invalid")
    else:
        monkeypatch.setenv("PRAXIS_AGENT_MAX_WORKERS", "2")
    with pytest.raises(ValueError):
        agent_module.create_default_agent(tmp_path, session_database_path=tmp_path / "factory.sqlite3")
    assert len(clients) == (0 if failure == "multi_agent_config" else 2)
    assert all(client.closed for client in clients)


def test_child_cancellation_finishes_only_its_execution_and_root_cancels_remaining_tools():
    transcript = Transcript()
    for agent, child_turn, call in [("worker-a", "child-a", "a"), ("worker-a", "child-b", "b"),
                                    ("worker-b", "child-c", "c")]:
        transcript.consume("ToolCallPrepared", "parent", agent, {
            "call_id": call, "tool_name": "run_command", "arguments": {},
            "requires_approval": True, "child_turn_id": child_turn,
        })
    transcript.consume("ApprovalResolved", "parent", "worker-a", {
        "call_id": "a", "tool_name": "run_command", "approved": False, "child_turn_id": "child-a",
    })
    transcript.consume("TurnCancelled", "parent", "worker-a", {"child_turn_id": "child-a"})
    assert [message["status"] for message in transcript.messages] == ["cancelled", "approval", "approval"]
    transcript.consume("TurnCancelled", "parent", "main", {})
    assert [message["status"] for message in transcript.messages] == ["cancelled"] * 3


def test_denied_finished_tool_remains_denied_after_root_cancellation():
    transcript = Transcript()
    transcript.consume("ToolCallPrepared", "turn", "main", {
        "call_id": "denied", "tool_name": "run_command", "arguments": {}, "requires_approval": True,
    })
    transcript.consume("ApprovalResolved", "turn", "main", {
        "call_id": "denied", "tool_name": "run_command", "approved": False,
    })
    transcript.consume("ToolFinished", "turn", "main", {
        "call_id": "denied", "tool_name": "run_command",
        "result": {"success": False, "output": "tool execution was declined by the user; do not retry"},
    })
    transcript.consume("TurnCancelled", "turn", "main", {})
    transcript.consume("TurnSettled", "turn", "main", {"status": "cancelled"})
    assert transcript.messages[0]["status"] == "denied"
    assert transcript.messages[0]["tool_finished"] is True


@pytest.mark.parametrize("output,expected", [
    ("tool execution was declined by the user; do not retry", "denied"),
    ({"status": "cancelled", "stdout": "partial"}, "cancelled"),
    ("ExecutionStopped: command cancelled", "cancelled"),
])
def test_restored_tool_keeps_denied_or_cancelled_result_status(output, expected):
    transcript = Transcript([
        {"type": "function_call", "call_id": "c", "name": "run_command", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c", "output": json.dumps({"success": False, "output": output})},
    ])
    assert transcript.messages[0]["status"] == expected
    assert transcript.messages[0]["tool_finished"] is True
