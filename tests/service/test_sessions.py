import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import httpx
import pytest
from support.service import answer, wait_until

from praxis_agent.client import APIError
from praxis_agent.execution import ExecutionStopped, current_handle
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.service.inputs import TurnInputs


class BlockingModel:
    def __init__(self):
        self.started, self.release, self.cleaning, self.cleanup_release = (
            Event() for _ in range(4)
        )
        self.cleanup_release.set()
        self.calls = []

    def create_response(self, items, tools, **kwargs):
        self.calls.append(list(items))
        if len(self.calls) == 1:
            handle = current_handle()
            handle.on_cancel(self.release.set)
            self.started.set()
            try:
                assert self.release.wait(5)
                handle.check()
            finally:
                self.cleaning.set()
                assert self.cleanup_release.wait(5)
        return answer()


def test_creation_and_viewing_are_lazy_and_idempotent(service):
    h = service()
    c = h.client()
    data = dict(command_id=uuid4().hex, workspace=str(h.path), model="test")
    first = c.request("POST", "/api/sessions", data)
    assert c.request("POST", "/api/sessions", data) == first
    session = first["session"]
    assert c.request("GET", f"/api/sessions/{session['id']}/snapshot")["messages"] == []
    assert h.agents == []
    assert h.manager.runtimes == {}


def test_two_clients_duplicate_input_once_and_conflicting_content_rejected(service):
    model = BlockingModel()
    h = service(model)
    session, key = h.create(), uuid4().hex
    try:
        with ThreadPoolExecutor(8) as pool:
            results = list(
                pool.map(lambda _: h.submit(session, "same input", key), range(16))
            )
        assert all(
            result == {"command_id": key, "accepted": True} for result in results
        )
        assert model.started.wait(2)
        assert len(h.agents) == 1
        with pytest.raises(APIError) as error:
            h.submit(session, "different input", key)
        assert error.value.code == "command_conflict"
    finally:
        model.release.set()
    snapshot = h.settled(session)
    assert [m["content"] for m in snapshot["messages"] if m["role"] == "user"] == [
        "same input"
    ]


def test_input_at_model_completion_joins_context_and_other_sessions_run(service):
    model = BlockingModel()
    h = service(model)
    first = h.create()
    h.submit(first, "first")
    assert model.started.wait(2)
    second = h.create()
    h.submit(second, "independent")
    assert h.settled(second)["session"]["latest_turn"]["status"] == "completed"
    h.submit(first, "follow up")
    model.release.set()
    snapshot = h.settled(first)
    assert len(model.calls) == 2
    assert "follow up" in str(model.calls[1])
    assert (
        len(
            {
                frame["turn_id"]
                for frame in snapshot["view_events"]
                if frame["event"] == "TurnStarted"
            }
        )
        == 1
    )
    assert snapshot["pending_inputs"] == []


def test_cancel_waits_cleanup_retains_inputs_and_new_input_resumes_in_order(service):
    model = BlockingModel()
    model.cleanup_release.clear()
    h = service(model)
    c, session = h.client(), h.create()
    h.submit(session, "first")
    assert model.started.wait(2)
    h.submit(session, "retained")
    turn = h.store.summary(session["id"])["active_turn"]["id"]
    url = f"/api/sessions/{session['id']}/turns/{turn}/cancel"
    try:
        assert c.request("POST", url)["cancel_requested"]
        assert c.request("POST", url)["cancel_requested"]
        assert model.cleaning.wait(2)
        h.submit(session, "during cleanup")
        assert h.store.summary(session["id"])["active_turn"]["cancel_requested"]
    finally:
        model.cleanup_release.set()
    snapshot = h.settled(session)
    assert snapshot["session"]["latest_turn"]["status"] == "cancelled"
    assert [item["content"] for item in snapshot["pending_inputs"]] == [
        "retained",
        "during cleanup",
    ]
    assert len(model.calls) == 1
    h.submit(session, "new instruction")
    h.settled(session)
    assert "retained" in str(model.calls[-1]) and "during cleanup" in str(
        model.calls[-1]
    )
    assert "new instruction" in str(model.calls[-1])
    assert c.request("POST", url)["status"] == "cancelled"


@pytest.mark.parametrize("approved", [True, False])
def test_first_approval_wins_and_duplicates_never_repeat_tool(service, approved):
    class Model:
        calls = 0

        def create_response(self, *args, **kwargs):
            self.calls += 1
            if self.calls > 1:
                return answer()
            call = FunctionCall(
                "write-1",
                "run_command",
                json.dumps(
                    {
                        "argv": [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; Path('result.txt').write_text('written')",
                        ]
                    }
                ),
            )
            return ModelResponse(
                (
                    {
                        "type": "function_call",
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                ),
                "",
                (call,),
            )

    h = service(Model())
    c, session = h.client(), h.create()
    h.submit(session)
    approval = wait_until(lambda: h.store.snapshot(session["id"])["approvals"])[0]
    url = f"/api/approvals/{approval['approval_id']}/decision"
    first = c.request("POST", url, {"approved": approved})
    assert c.request("POST", url, {"approved": not approved}) == first
    h.settled(session)
    assert (h.path / "result.txt").exists() == approved


def test_cancel_invalidates_approval_and_does_not_execute_late_approval(service):
    class Model:
        def create_response(self, *args, **kwargs):
            call = FunctionCall(
                "write-1",
                "run_command",
                json.dumps(
                    {
                        "argv": [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; Path('unexpected.txt').touch()",
                        ]
                    }
                ),
            )
            return ModelResponse(
                (
                    {
                        "type": "function_call",
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                ),
                "",
                (call,),
            )

    h = service(Model())
    c, session = h.client(), h.create()
    h.submit(session)
    approval = wait_until(lambda: h.store.snapshot(session["id"])["approvals"])[0]
    c.request(
        "POST", f"/api/sessions/{session['id']}/turns/{approval['turn_id']}/cancel"
    )
    h.settled(session)
    result = c.request(
        "POST", f"/api/approvals/{approval['approval_id']}/decision", {"approved": True}
    )
    assert result["status"] == "cancelled"
    assert not (h.path / "unexpected.txt").exists()


def test_closing_client_does_not_cancel_active_turn(service):
    model = BlockingModel()
    h = service(model)
    c, session = h.client(), h.create()
    h.submit(session)
    assert model.started.wait(2)
    c.subscribe(session["id"], lambda _: None)
    c.close()
    assert h.store.summary(session["id"])["active_turn"] is not None
    model.release.set()
    assert h.settled(session)["session"]["latest_turn"]["status"] == "completed"


def test_business_mutation_and_event_roll_back_together(service):
    h = service()
    session = h.create()
    before = h.store.snapshot(session["id"])["sequence"]

    def fail(connection):
        connection.execute(
            "INSERT INTO inputs(command_id, session_id, content) VALUES ('rollback', ?, 'x')",
            (session["id"],),
        )
        h.store.pending_changed(session["id"])
        raise RuntimeError("test failure")

    with pytest.raises(RuntimeError):
        h.store.command("rollback", "input", {}, fail)
    assert h.store.pending(session["id"]) == []
    assert h.store.snapshot(session["id"])["sequence"] == before


def test_authentication_and_invalid_requests_do_not_disrupt_service(service):
    h = service()
    with httpx.Client(trust_env=False) as c:
        assert c.get(h.endpoint["url"] + "/api/health").status_code == 401
        headers = {"Authorization": "Bearer test-token"}
        assert (
            c.get(
                h.endpoint["url"] + "/api/health",
                headers={**headers, "Origin": "https://example.com"},
            ).status_code
            == 403
        )
        assert (
            c.post(
                h.endpoint["url"] + "/api/sessions", content="bad", headers=headers
            ).status_code
            == 400
        )
        assert c.get(h.endpoint["url"] + "/api/health", headers=headers).json()["ready"]


def test_late_input_starts_next_turn_and_stale_cancel_cannot_stop_it(
    service, monkeypatch
):
    checked, finish_first, second_started, finish_second = (Event() for _ in range(4))
    receive = TurnInputs.receive

    def boundary(self, turn_id, accept, *, final=False):
        received = receive(self, turn_id, accept, final=final)
        if final and not received and not checked.is_set():
            checked.set()
            assert finish_first.wait(5)
        return received

    class Model:
        calls = 0

        def create_response(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                second_started.set()
                assert finish_second.wait(5)
                current_handle().check()
            return answer()

    monkeypatch.setattr(TurnInputs, "receive", boundary)
    h = service(Model())
    session, client = h.create(), h.client()
    try:
        h.submit(session, "first")
        assert checked.wait(3)
        first_id = h.store.summary(session["id"])["active_turn"]["id"]
        h.submit(session, "after final check")
        finish_first.set()
        assert second_started.wait(3)
        second_id = h.store.summary(session["id"])["active_turn"]["id"]
        assert first_id != second_id
        reply = client.request(
            "POST", f"/api/sessions/{session['id']}/turns/{first_id}/cancel"
        )
        assert reply["status"] == "completed"
        assert not h.store.summary(session["id"])["active_turn"]["cancel_requested"]
    finally:
        finish_first.set()
        finish_second.set()
    snapshot = h.settled(session)
    assert snapshot["session"]["latest_turn"]["status"] == "completed"
    assert [m["content"] for m in snapshot["messages"] if m["role"] == "user"] == [
        "first",
        "after final check",
    ]
    assert snapshot["pending_inputs"] == []


def test_pending_delete_and_resume_are_idempotent(service):
    model = BlockingModel()
    h = service(model)
    session, client = h.create(), h.client()
    base = f"/api/sessions/{session['id']}"
    initial = h.submit(session)
    assert model.started.wait(3)
    with pytest.raises(APIError) as error:
        client.request("DELETE", base + "/inputs/" + initial["command_id"])
    assert error.value.code == "input_consumed"
    removed = h.submit(session, "delete this")
    retained = h.submit(session, "keep this")
    turn = h.store.summary(session["id"])["active_turn"]["id"]
    client.request("POST", base + f"/turns/{turn}/cancel")
    h.settled(session)
    for _ in range(2):
        assert client.request("DELETE", base + "/inputs/" + removed["command_id"]) == {
            "deleted": True
        }
    assert h.store.pending(session["id"]) == [
        {"command_id": retained["command_id"], "content": "keep this"}
    ]
    data = {"command_id": uuid4().hex}
    first = client.request("POST", base + "/inputs/resume", data)
    assert client.request("POST", base + "/inputs/resume", data) == first
    snapshot = h.settled(session)
    assert len(model.calls) == 2
    assert snapshot["pending_inputs"] == []
    assert [m["content"] for m in snapshot["messages"] if m["role"] == "user"] == [
        "task",
        "keep this",
    ]


@pytest.mark.parametrize(
    "failure,status,event",
    [
        (ValueError("model failed"), "failed", "TurnFailed"),
        (ExecutionStopped("cancelled"), "cancelled", "TurnCancelled"),
    ],
)
def test_terminal_result_and_event_commit_together(
    service, monkeypatch, failure, status, event
):
    class Model:
        def create_response(self, *args, **kwargs):
            raise failure

    h = service(Model())
    session = h.create()
    append = h.store.append_event
    observed = []

    def observe(session_id, kind, turn_id="", agent_id="main", data=None):
        if kind == event:
            # A separate reader must still see running until both writes commit.
            with sqlite3.connect(h.store.db.path) as reader:
                observed.append(
                    reader.execute(
                        "SELECT status FROM runs WHERE id = ?", (turn_id,)
                    ).fetchone()[0]
                )
        return append(session_id, kind, turn_id, agent_id, data)

    monkeypatch.setattr(h.store, "append_event", observe)
    h.submit(session)
    snapshot = h.settled(session)
    assert observed == ["running"]
    assert snapshot["session"]["latest_turn"]["status"] == status
    assert sum(frame["event"] == event for frame in snapshot["view_events"]) == 1
