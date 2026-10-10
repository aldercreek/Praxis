from copy import deepcopy

import pytest

from praxis_agent.context.operations import ContextOperation, ContextOperations


def call(cid):
    return {"type": "function_call", "call_id": cid, "name": "read_file", "arguments": "{}"}


def result(cid):
    return {"type": "function_call_output", "call_id": cid, "output": cid}


def operation(action, ids, number=1):
    return ContextOperation(
        f"op{number}", action, tuple(ids),
        replacement={"message_id": f"replacement-{number}",
                     "payload": {"role": "assistant", "content": "retained findings"}},
    )


def assert_tool_sequence(messages):
    pending = set()
    for message in messages:
        payload = message["payload"]
        if payload.get("type") == "function_call":
            pending.add(payload["call_id"])
        elif payload.get("type") == "function_call_output":
            assert payload["call_id"] in pending
            pending.remove(payload["call_id"])
        elif payload.get("role"):
            assert not pending
    assert not pending


@pytest.mark.parametrize("action", ["compress", "delete"])
@pytest.mark.parametrize("selected", [("m4",), ("m5",), ("m6",), ("m4", "m6"), ("m4", "m5", "m6")])
def test_partial_tool_results_keep_replacement_after_remaining_outputs(action, selected):
    history = [call("a"), call("b"), call("c"), result("a"), result("b"), result("c"),
               {"role": "user", "content": "next task"}]
    original = deepcopy(history)
    view = ContextOperations([operation(action, selected)]).project(history)
    assert_tool_sequence(view)
    assert [m["message_id"] for m in view][-2:] == ["replacement-1", "m7"]
    remaining = {cid for mid, cid in [("m4", "a"), ("m5", "b"), ("m6", "c")] if mid not in selected}
    assert {m["payload"]["call_id"] for m in view if m["payload"].get("type") == "function_call_output"} == remaining
    assert history == original


def test_replayed_operations_can_target_deferred_replacement_across_batches():
    history = [call("a"), call("b"), result("a"), result("b"),
               {"role": "user", "content": "continue"},
               call("c"), call("d"), result("c"), result("d")]
    records = [operation("delete", ["m3"]),
               operation("compress", ["replacement-1", "m8"], 2)]
    replayed = ContextOperations()
    replayed.replay([ContextOperation.from_dict({**r.to_dict(), "message_ids": list(r.message_ids)})
                     for r in records])
    view = replayed.project(history)
    assert_tool_sequence(view)
    assert [m["message_id"] for m in view] == ["m2", "m4", "m5", "m7", "m9", "replacement-2"]
