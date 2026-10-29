from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Event

import pytest

from praxis_agent.planning import RunState
from praxis_agent.service.store import ServiceStore
from praxis_agent.service.transcript import Transcript
from praxis_agent.sessions import SQLiteSessionStore


def test_replay_live_handoff_is_ordered_while_events_are_added(service):
    h = service()
    session = h.create()
    sid = session["id"]
    for n in range(600):
        h.store.append_event(sid, "TestEvent", data={"number": n})
    received = Queue()
    client = h.client()
    sub = client.subscribe(sid, received.put, after=100)
    for n in range(600, 640):
        h.store.append_event(sid, "TestEvent", data={"number": n})
    frames = [received.get(timeout=3) for _ in range(541)]
    assert [frame["sequence"] for frame in frames] == list(range(101, 642))
    sub.acknowledge(641)
    sub.close()
    h.store.append_event(sid, "TestEvent", data={"number": 640})
    sub2 = client.subscribe(sid, received.put, after=641)
    assert received.get(timeout=3)["sequence"] == 642
    sub2.close()


def test_connection_drop_replays_only_after_client_acknowledgement(service):
    h = service()
    sid = h.create()["id"]
    queue = Queue()
    client = h.client()
    sub = client.subscribe(sid, queue.put)
    first = queue.get(timeout=3)
    sub.acknowledge(first["sequence"])
    second = h.store.append_event(sid, "TestEvent", data={"text": "中文"})
    assert queue.get(timeout=3) == second
    # Deliberately do not acknowledge second: receiving bytes is not applying UI state.
    connections = list(h.server.server_state.connections)
    for connection in connections:
        connection.loop.call_soon_threadsafe(connection.transport.abort)
    third = h.store.append_event(sid, "TestEvent", data={"text": "after drop"})
    replay = [queue.get(timeout=5), queue.get(timeout=5)]
    assert replay == [second, third]
    sub.acknowledge(third["sequence"])


def test_snapshot_boundary_discards_buffered_old_events(service):
    h = service()
    sid = h.create()["id"]
    queue = Queue()
    sub = h.client().subscribe(sid, queue.put)
    h.store.append_event(
        sid, "AssistantTextDelta", "T", data={"request_number": 1, "text": "before"}
    )
    snapshot = h.store.snapshot(sid)
    frame = h.store.append_event(
        sid, "AssistantTextDelta", "T", data={"request_number": 1, "text": "after"}
    )
    buffered = [queue.get(timeout=3) for _ in range(3)]
    assert snapshot["messages"][0]["content"] == "before"
    assert [item for item in buffered if item["sequence"] > snapshot["sequence"]] == [
        frame
    ]
    sub.acknowledge(frame["sequence"])


def test_snapshot_projection_does_not_block_event_writers(service, monkeypatch):
    h = service()
    sid = h.create()["id"]
    projecting, release = Event(), Event()
    consume = Transcript.consume

    def slow_projection(self, *args):
        projecting.set()
        assert release.wait(5)
        return consume(self, *args)

    monkeypatch.setattr(Transcript, "consume", slow_projection)
    with ThreadPoolExecutor(2) as pool:
        snapshot = pool.submit(h.store.snapshot, sid)
        try:
            assert projecting.wait(3)
            appended = pool.submit(h.store.append_event, sid, "TestEvent").result(timeout=3)
        finally:
            release.set()
        assert snapshot.result(timeout=3)["sequence"] == appended["sequence"] - 1


def test_recovery_fails_unfinished_turn_preserves_events_and_pending_input(tmp_path):
    sessions = SQLiteSessionStore(tmp_path / "state.sqlite3")
    store = ServiceStore(sessions)
    sid = sessions.create_session(tmp_path, "test").id
    sessions.start_run(
        sid, "old-turn", RunState("task"), {"role": "user", "content": "task"}
    )
    sessions.append_response_items(
        sid,
        "old-turn",
        [
            {
                "type": "function_call",
                "call_id": "unknown",
                "name": "run_command",
                "arguments": "{}",
            }
        ],
    )
    store.append_event(sid, "TurnStarted", "old-turn", data={"task": "task"})
    store.append_event(
        sid,
        "ToolStarted",
        "old-turn",
        data={"call_id": "unknown", "tool_name": "run_command"},
    )
    with store.db.transaction() as connection:
        connection.execute("INSERT INTO active_turns(session_id, turn_id, cancel_requested) VALUES (?, 'old-turn', 0)", (sid,))
        connection.execute(
            "INSERT INTO approvals VALUES ('A1', ?, 'old-turn', 'main', 'run_command', '{}', 'pending')",
            (sid,),
        )

    def pending(connection):
        connection.execute(
            "INSERT INTO inputs(command_id, session_id, content) VALUES ('C1', ?, 'keep me')",
            (sid,),
        )
        return {"accepted": True}

    store.command("C1", "input", {}, pending)
    reopened = ServiceStore(SQLiteSessionStore(tmp_path / "state.sqlite3"))
    reopened.recover()
    snapshot = reopened.snapshot(sid)
    assert snapshot["session"]["latest_turn"]["status"] == "failed"
    assert snapshot["session"]["active_turn"] is None
    assert snapshot["pending_inputs"] == [{"command_id": "C1", "content": "keep me"}]
    assert snapshot["approvals"] == []
    with reopened.db.connection() as connection:
        tool = connection.execute(
            "SELECT status, result FROM tool_calls WHERE call_id = 'unknown'"
        ).fetchone()
        assert tool["status"] == "failed" and "unknown" in tool["result"]
    history = sessions.load_conversation_items(sid)
    assert "unknown" in history[-1]["output"]
    assert (
        reopened.append_event(sid, "TestEvent")["sequence"] == snapshot["sequence"] + 1
    )
    reopened.recover()
    assert reopened.snapshot(sid)["sequence"] == snapshot["sequence"] + 1


@pytest.mark.parametrize("status", ["completed", "cancelled", "failed"])
def test_recovery_preserves_result_committed_before_runtime_settled(tmp_path, status):
    sessions = SQLiteSessionStore(tmp_path / "state.sqlite3")
    store = ServiceStore(sessions)
    sid = sessions.create_session(tmp_path, "test").id
    state = RunState("task")
    sessions.start_run(sid, "turn", state, {"role": "user", "content": "task"})
    if status == "completed":
        state.complete()
    elif status == "cancelled":
        state.cancel()
    else:
        state.fail(ValueError("original error"))
    sessions.finish_run(sid, "turn", state)
    with store.db.transaction() as connection:
        connection.execute("INSERT INTO active_turns(session_id, turn_id, cancel_requested) VALUES (?, 'turn', 0)", (sid,))
    store.recover()
    snapshot = store.snapshot(sid)
    assert snapshot["session"]["latest_turn"]["status"] == status
    assert snapshot["session"]["active_turn"] is None
    assert not any(event["event"] == "TurnFailed" for event in snapshot["view_events"])
    assert (
        next(
            event
            for event in snapshot["view_events"]
            if event["event"] == "TurnSettled"
        )["data"]["status"]
        == status
    )
