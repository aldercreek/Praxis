import json

import pytest

from praxis_agent.planning import RunState
from praxis_agent.terminal.cells import MessageBlock
from praxis_agent.terminal.connection import TerminalSession
from praxis_agent.terminal.views import ToolBlock, ViewStore


def test_cold_history_keeps_collaboration_in_its_agent_without_rewriting_model_history(service):
    h = service()
    sid = h.store.sessions.create_session(h.path, "test").id
    state = RunState("review this")
    h.store.sessions.start_run(sid, "saved", state, {"role": "user", "content": state.original_task})
    envelope = "[Agent collaboration messages; source-labelled task data]\n" + json.dumps([
        dict(sender_id="reviewer", recipient_id="parent", content="Review complete.")
    ])
    h.store.sessions.append_response_items(sid, "saved", [{"role": "user", "content": envelope}])
    state.complete()
    h.store.sessions.finish_run(sid, "saved", state)
    h.store.append_event(sid, "TurnStarted", "saved", data={"task": "review this"})
    h.store.append_event(sid, "MessagesReceived", "saved", data={"source": "agent", "items": [{"role": "user", "content": envelope}]})
    snapshot = h.store.snapshot(sid)
    assert [m["content"] for m in snapshot["messages"]] == ["review this", "Review complete."]
    assert snapshot["messages"][1]["agent_id"] == "reviewer"
    assert snapshot["messages"][1]["source"] == "agent"
    assert h.store.sessions.load_conversation_items(sid)[-1]["content"] == envelope


def saved_turn(h, sid, turn_id, *, finish=True, journal=True):
    state = RunState("saved question")
    h.store.sessions.start_run(
        sid, turn_id, state, {"role": "user", "content": "saved question"}
    )
    h.store.sessions.append_response_items(
        sid,
        turn_id,
        [
            {"role": "developer", "content": "private instructions"},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "checking the file"}],
            },
            {
                "type": "function_call",
                "call_id": "read",
                "name": "read_file",
                "arguments": '{"relative_path":"notes.md"}',
            },
            {
                "type": "function_call_output",
                "call_id": "read",
                "output": json.dumps({"success": True, "output": "saved tool output"}),
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "saved supplement"}],
            },
            {"role": "assistant", "content": "saved answer"},
        ],
    )
    if journal:
        events = [
            ("TurnStarted", {"task": "saved question"}),
            ("ModelRequestStarted", {"request_number": 1}),
            ("ModelRequestCompleted", {"request_number": 1, "output_text": "checking the file"}),
            ("ToolCallPrepared", {"call_id": "read", "tool_name": "read_file", "arguments": {"relative_path": "notes.md"}, "requires_approval": False}),
            ("ToolFinished", {"tool_name": "read_file", "call_id": "read", "result": {"success": True, "output": "saved tool output"}}),
            ("MessagesReceived", {"source": "user", "items": [{"role": "user", "content": "saved supplement"}]}),
            ("ModelRequestStarted", {"request_number": 2}),
            ("ModelRequestCompleted", {"request_number": 2, "output_text": "saved answer"}),
        ]
        if finish:
            events += [("AssistantTextCompleted", {"text": "saved answer"}), ("TurnCompleted", {})]
        for event, data in events:
            h.store.append_event(sid, event, turn_id, data=data)
    if finish:
        state.complete()
        h.store.sessions.finish_run(sid, turn_id, state)


@pytest.mark.parametrize("recover", [False, True])
def test_saved_messages_and_tools_open_in_desktop_and_terminal_without_running_an_agent(
    service, recover
):
    h = service()
    sid = h.store.sessions.create_session(h.path, "test").id
    saved_turn(h, sid, "saved", finish=not recover)
    if recover:
        h.store.recover()
    journal = h.store.events(sid)
    snapshot = h.client().request("GET", f"/api/sessions/{sid}/snapshot")
    assert [message["content"] for message in snapshot["messages"][:5]] == [
        "saved question",
        "checking the file",
        "saved tool output",
        "saved supplement",
        "saved answer",
    ]
    assert all(message["turn_id"] == "saved" for message in snapshot["messages"])
    assert snapshot["messages"][2]["arguments"] == {"relative_path": "notes.md"}
    assert snapshot["messages"][2]["tool_finished"] is True
    assert "private instructions" not in json.dumps(snapshot)
    assert snapshot["sequence"] == (journal[-1]["sequence"] if journal else 0)
    assert h.store.events(sid) == journal

    terminal = TerminalSession(h.client(), snapshot["session"])
    view = ViewStore()
    terminal.drain(view)
    blocks = view.views["main"].committed_blocks
    assert [block.text for block in blocks if isinstance(block, MessageBlock)] == [
        "saved question",
        "checking the file",
        "saved supplement",
        "saved answer",
    ]
    assert [block.output for block in blocks if isinstance(block, ToolBlock)] == [
        "saved tool output"
    ]
    assert h.agents == []


def test_saved_and_journaled_turns_keep_order_and_do_not_duplicate_after_continuing(
    service,
):
    h = service()
    sid = h.store.sessions.create_session(h.path, "test").id
    saved_turn(h, sid, "saved")
    session = h.store.summary(sid)
    h.submit(session, "new question")
    snapshot = h.settled(session)
    assert [message["content"] for message in snapshot["messages"]] == [
        "saved question",
        "checking the file",
        "saved tool output",
        "saved supplement",
        "saved answer",
        "new question",
        "done",
    ]
    assert len({message["id"] for message in snapshot["messages"]}) == len(
        snapshot["messages"]
    )
    assert h.store.snapshot(sid) == snapshot
    assert [
        frame["sequence"] for frame in snapshot["view_events"] if "sequence" in frame
    ] == [frame["sequence"] for frame in h.store.events(sid)]


def test_running_turn_without_start_event_is_not_restored_ahead_of_its_journal(service):
    h = service()
    session = h.create()
    sid = session["id"]
    saved_turn(h, sid, "starting", finish=False, journal=False)
    assert h.store.snapshot(sid)["messages"] == []
    h.store.append_event(
        sid, "TurnStarted", "starting", data={"task": "saved question"}
    )
    assert len(h.store.snapshot(sid)["messages"]) == 1
