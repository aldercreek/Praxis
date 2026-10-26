import json
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest
from support.service import wait_until
from test_sessions import BlockingModel

from praxis_agent.client import APIError
from praxis_agent.service.errors import ServiceError
from praxis_agent.service.manager import SessionManager
from praxis_agent.service.store import ServiceStore
from praxis_agent.sessions import SQLiteSessionStore


def test_delete_purges_history_inputs_and_events_but_retains_creation_receipt(service):
    h = service()
    client = h.client()
    data = dict(command_id=uuid4().hex, workspace=str(h.path), model="test")
    session = client.request("POST", "/api/sessions", data)["session"]
    h.submit(session, "private input")
    h.settled(session)
    path = f"/api/sessions/{session['id']}"
    for _ in range(2):
        assert client.request("DELETE", path)["deleted"]
    assert client.request("GET", "/api/sessions")["sessions"] == []
    with h.store.db.connection() as db:
        for table in ("sessions", "runs", "conversation_items", "context_view_operations",
                      "inputs", "active_turns", "approvals", "events", "tool_calls"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        receipts = db.execute("SELECT kind, payload, result FROM commands").fetchall()
        assert len(receipts) == 1 and receipts[0]["kind"] == "create"
        assert json.loads(receipts[0]["result"])["deleted"]
        assert "private input" not in str(tuple(receipts[0]))
    with pytest.raises(APIError) as error:
        client.request("POST", "/api/sessions", data)
    assert error.value.status == 410
    assert session["id"] not in h.manager.runtimes


@pytest.mark.parametrize("already_stopping", [False, True])
def test_delete_closes_admission_waits_once_and_does_not_block_other_sessions(service, monkeypatch, already_stopping):
    model = BlockingModel()
    model.cleanup_release.clear()
    h = service(model)
    session = h.create()
    client, sid = h.client(), session["id"]
    h.submit(session)
    assert model.started.wait(2)
    h.submit(session, "must never run")
    runtime = h.manager.runtime(sid)
    turn = runtime.run.id
    if already_stopping:
        runtime.cancel(turn)
    calls = []
    original = runtime.close

    def close():
        calls.append(sid)
        original()

    monkeypatch.setattr(runtime, "close", close)
    frames = []
    subscription = client.subscribe(sid, frames.append)
    with ThreadPoolExecutor(3) as pool:
        first = pool.submit(client.request, "DELETE", f"/api/sessions/{sid}")
        try:
            assert model.cleaning.wait(3)
            second = pool.submit(h.client().request, "DELETE", f"/api/sessions/{sid}")
            wait_until(lambda: sid in h.store.deleting_sessions())
            assert not first.done()
            for operation in (
                lambda: runtime.submit(uuid4().hex, "late"),
                lambda: runtime.resume(uuid4().hex),
                lambda: runtime.delete_input("missing"),
                lambda: runtime.cancel(turn),
                lambda: runtime.approvals.decide("missing", True),
                lambda: h.manager.runtime(sid),
            ):
                with pytest.raises(ServiceError) as error:
                    operation()
                assert error.value.status == 410
            for suffix in ("snapshot", "events", "commands"):
                with pytest.raises(APIError) as error:
                    client.request("GET", f"/api/sessions/{sid}/{suffix}")
                assert error.value.status == 410
            assert client.request("GET", "/api/sessions")["sessions"] == []
            wait_until(lambda: any(frame["event"] == "SessionDeleted" for frame in frames))
            wait_until(lambda: not subscription.thread.is_alive())
            other = h.create()
            h.submit(other)
            assert h.settled(other)["session"]["latest_turn"]["status"] == "completed"
        finally:
            model.cleanup_release.set()
        assert first.result(5) == second.result(5)
    assert calls == [sid]
    assert len(model.calls) == 1


def test_failed_cleanup_keeps_marker_and_can_be_retried(service, monkeypatch):
    h = service()
    session = h.create()
    runtime = h.manager.runtime(session["id"])
    close = runtime.close
    monkeypatch.setattr(runtime, "close", lambda: (_ for _ in ()).throw(OSError("cleanup failed")))
    with pytest.raises(OSError):
        h.manager.delete(session["id"])
    assert h.store.deleting_sessions() == [session["id"]]
    assert runtime.closed
    assert h.manager.list()["sessions"] == []
    assert h.manager.deletions.pending == {}
    monkeypatch.setattr(runtime, "close", close)
    assert h.manager.delete(session["id"])["deleted"]
    assert h.store.deleting_sessions() == []


def test_failed_marker_leaves_session_available(service, monkeypatch):
    h = service()
    session = h.create()
    runtime = h.manager.runtime(session["id"])
    monkeypatch.setattr(h.store, "mark_for_deletion", lambda _: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        h.manager.delete(session["id"])
    assert not runtime.closed
    assert h.manager.deletions.pending == {}
    assert h.manager.runtime(session["id"]) is runtime


def test_startup_loads_markers_before_background_cleanup_without_creating_agents(tmp_path, monkeypatch):
    store = ServiceStore(SQLiteSessionStore(tmp_path / "state.sqlite3"))
    sid = store.sessions.create_session(tmp_path, "test").id
    store.mark_for_deletion(sid)
    # Simulate a crash before the active turn was settled.
    with store.db.transaction() as db:
        db.execute("INSERT INTO active_turns(session_id, turn_id) VALUES (?, 'unfinished')", (sid,))
    store = ServiceStore(SQLiteSessionStore(tmp_path / "state.sqlite3"))
    store.recover()
    assert store.events(sid) == []
    entered, release = Event(), Event()
    delete = store.delete_session

    def delayed(session_id):
        entered.set()
        assert release.wait(5)
        delete(session_id)

    monkeypatch.setattr(store, "delete_session", delayed)
    manager = SessionManager(store, agent_factory=lambda *a, **k: pytest.fail("restored deleted Agent"))
    try:
        assert entered.wait(2)
        assert manager.runtimes == {}
        with pytest.raises(ServiceError) as error:
            manager.runtime(sid)
        assert error.value.status == 410
        assert manager.list()["sessions"] == []
    finally:
        release.set()
        manager.deletions.wait()
        manager.close()
    assert store.deleting_sessions() == []
    with pytest.raises(ServiceError) as error:
        store.require_available(sid)
    assert error.value.status == 404


def test_reconnecting_to_deleted_stream_is_terminal(service):
    h = service()
    session = h.create()
    h.manager.delete(session["id"])
    frames = []
    subscription = h.client().subscribe(session["id"], frames.append)
    wait_until(lambda: not subscription.thread.is_alive())
    assert [frame["event"] for frame in frames] == ["SessionDeleted"]


def test_idle_chat_deletion_stops_background_process_and_preserves_workspace(service):
    h = service()
    session = h.create()
    h.submit(session)
    h.settled(session)
    keep = h.path / "keep.txt"
    keep.write_text("user file")
    commands = h.manager.runtime(session["id"]).agent.command_sessions
    command_id = commands.start([sys.executable, "-c", "import time; time.sleep(60)"], h.path)
    process = commands._get(command_id).process.process
    assert process.poll() is None
    assert h.manager.delete(session["id"])["deleted"]
    assert process.poll() is not None
    assert keep.read_text() == "user file"


def test_database_cleanup_rolls_back_and_can_retry(service):
    h = service()
    session = h.create()
    h.submit(session)
    h.settled(session)
    before = h.store.events(session["id"])
    with h.store.db.transaction() as db:
        db.execute("CREATE TRIGGER fail_delete BEFORE DELETE ON sessions BEGIN SELECT RAISE(ABORT, 'busy'); END")
    with pytest.raises(Exception, match="busy"):
        h.manager.delete(session["id"])
    assert h.store.events(session["id"]) == before
    assert h.store.deleting_sessions() == [session["id"]]
    with h.store.db.transaction() as db:
        db.execute("DROP TRIGGER fail_delete")
    assert h.manager.delete(session["id"])["deleted"]


def test_failed_turn_retains_agent_until_resource_cleanup_can_be_retried(service, monkeypatch):
    h = service()
    session = h.create()
    h.submit(session)
    h.settled(session)
    runtime = h.manager.runtime(session["id"])
    agent = runtime.agent
    close_commands = agent.command_sessions.close
    attempts = []

    def close():
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("temporary cleanup failure")
        close_commands()

    monkeypatch.setattr(agent.command_sessions, "close", close)
    monkeypatch.setattr(type(agent), "run", lambda *_: (_ for _ in ()).throw(RuntimeError("model failed")))
    h.submit(session, "failure")
    assert h.settled(session)["session"]["latest_turn"]["status"] == "failed"
    assert runtime.agent is agent and runtime.closed
    assert not agent._resources_closed
    assert h.manager.delete(session["id"])["deleted"]
    assert agent._resources_closed and len(attempts) == 2
