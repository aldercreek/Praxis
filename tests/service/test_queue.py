from uuid import uuid4

import pytest
from support.service import wait_until

from praxis_agent.client import APIError
from praxis_agent.service.store import ServiceStore
from praxis_agent.sessions import SQLiteSessionStore
from test_sessions import BlockingModel


def action(h, session, name, **params):
    return h.client().request("POST", f"/api/sessions/{session['id']}/queue/{name}", {
        "command_id": uuid4().hex, **params,
    })


def test_edit_reorder_delete_persist_and_reject_stale_queue(service):
    model = BlockingModel()
    h = service(model)
    session = h.create()
    h.submit(session, "first", "first")
    assert model.started.wait(2)
    try:
        for name in ("one", "two", "three"):
            h.submit(session, name, name)
        snapshot = h.store.snapshot(session["id"])
        assert snapshot["sending_input_id"] == "first"
        assert [item["command_id"] for item in snapshot["pending_inputs"]] == ["one", "two", "three"]
        action(h, session, "reorder", input_ids=["three", "one", "two"])
        reopened = ServiceStore(SQLiteSessionStore(h.path / "state.sqlite3"))
        assert [item["command_id"] for item in reopened.pending(session["id"])] == ["three", "one", "two"]
        with pytest.raises(APIError) as error:
            action(h, session, "reorder", input_ids=["one", "two", "two"])
        assert error.value.code == "queue_changed"
        edit = dict(command_id="edit-two", input_id="two")
        assert action(h, session, "edit", **edit) == {"content": "two"}
        assert action(h, session, "edit", **edit) == {"content": "two"}
        h.client().request("DELETE", f"/api/sessions/{session['id']}/inputs/one")
        with pytest.raises(APIError) as error:
            action(h, session, "edit", input_id="first")
        assert error.value.code == "input_consumed"
    finally:
        model.release.set()
    snapshot = h.settled(session)
    assert [m["content"] for m in snapshot["messages"] if m["role"] == "user"] == ["first", "three"]
    assert snapshot["pending_inputs"] == []
    assert snapshot["sending_input_id"] is None


@pytest.mark.parametrize("stop_handover", [False, True])
def test_send_now_waits_for_cleanup_and_stop_can_cancel_handover(service, stop_handover):
    model = BlockingModel()
    model.cleanup_release.clear()
    h = service(model)
    session = h.create()
    h.submit(session, "first", "first")
    assert model.started.wait(2)
    h.submit(session, "older", "older")
    h.submit(session, "priority", "priority")
    old_turn = h.store.summary(session["id"])["active_turn"]["id"]
    try:
        send = dict(command_id="send-priority", input_id="priority")
        assert action(h, session, "send", **send) == {"accepted": True}
        assert action(h, session, "send", **send) == {"accepted": True}
        assert model.cleaning.wait(2)
        assert len(model.calls) == 1
        assert h.store.snapshot(session["id"])["sending_input_id"] == "priority"
        for operation in ("edit", "send"):
            with pytest.raises(APIError) as error:
                action(h, session, operation, input_id="priority")
            assert error.value.code == "input_consumed"
        # An invalid/old cancellation must not clear the scheduled handover.
        with pytest.raises(APIError):
            h.client().request("POST", f"/api/sessions/{session['id']}/turns/unknown-turn/cancel")
        if stop_handover:
            h.client().request("POST", f"/api/sessions/{session['id']}/turns/{old_turn}/cancel")
    finally:
        model.cleanup_release.set()
    if not stop_handover:
        wait_until(lambda: len(model.calls) > 1)
    snapshot = h.settled(session)
    if stop_handover:
        assert len(model.calls) == 1
        assert [item["content"] for item in snapshot["pending_inputs"]] == ["priority", "older"]
        action(h, session, "send", input_id="priority")
        snapshot = h.settled(session)
    assert snapshot["pending_inputs"] == []
    assert [m["content"] for m in snapshot["messages"] if m["role"] == "user"] == ["first", "priority", "older"]
    assert len({e["turn_id"] for e in snapshot["view_events"] if e["event"] == "TurnStarted"}) == 2
