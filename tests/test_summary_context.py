import json
from copy import deepcopy
from queue import SimpleQueue
from types import SimpleNamespace

import pytest

from praxis_agent.agent import Agent
from praxis_agent.context.capacity import ContextPolicy
from praxis_agent.context.compaction import trim_tool_messages
from praxis_agent.context.manager import ContextManager
from praxis_agent.context.model_metadata import ModelLimits
from praxis_agent.context.models import ConversationHistory
from praxis_agent.context.operations import ContextOperation
from praxis_agent.context.retrieval import read_history
from praxis_agent.context.summarizer import ModelContextSummarizer
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.model_errors import PromptTooLongError
from praxis_agent.planning.plan import PlanStep, PlanStepStatus
from praxis_agent.tools.registry import Tool, ToolRegistry


def test_agent_keeps_current_task_corrections_and_progress_in_each_summary_chunk(tmp_path):
    requests, summaries = [], []
    user_input = SimpleQueue()

    def inspect(context):
        context.run_state.plan = [PlanStep("Inspect expiry", PlanStepStatus.COMPLETED),
                                  PlanStep("Run tests", PlanStepStatus.PENDING)]
        context.run_state.record_changed_file("auth.py")
        user_input.put("Correction: keep the public API unchanged.")
        return "Verified expiry condition is reversed. " + "trace " * 8000

    def main(items, tools, **options):
        requests.append(deepcopy(items))
        if len(requests) == 1:
            call = FunctionCall("inspect-1", "inspect", "{}")
            return ModelResponse((
                {"role": "assistant", "content": "Observed progress. " * 400},
                {"type": "function_call", "call_id": call.call_id, "name": call.name, "arguments": call.arguments},
            ), "", (call,))
        if len(requests) == 3:
            raise PromptTooLongError("input too long")
        return ModelResponse(({"role": "assistant", "content": "done"},), "done", ())

    def summarize(items, **options):
        summaries.append(json.loads(items[0]["content"]))
        return ModelResponse((), "Expiry condition reversed; tests pending.", ())

    registry = ToolRegistry()
    registry.register(Tool("inspect", "Inspect expiry handling", {"type": "object", "properties": {}}, inspect))
    manager = ContextManager(
        model_limits=ModelLimits(12000),
        policy=ContextPolicy(output_reserve_tokens=1000, summary_output_tokens=1000, max_tool_output_chars=32000),
        summarizer=ModelContextSummarizer(SimpleNamespace(create_response=summarize),
                                         ModelLimits(128000), input_limit_tokens=3000),
    )
    agent = Agent(SimpleNamespace(create_response=main), registry, tmp_path, context_manager=manager,
                  user_input=user_input, bypass_approvals=True, mcp_catalog=lambda: "UNRELATED_MCP_CATALOG")
    agent.restore_history([{"role": "user", "content": "PRIOR_TURN_REQUEST"}])
    try:
        assert agent.run("Fix expiry handling and verify it.") == "done"
        first_turn_summaries = len(summaries)
        assert first_turn_summaries > 1
        assert [op.action for op in manager.operations.records] == ["compress"]
        for data in summaries:
            reference = data["reference_context"]
            assert reference[:2] == [
                {"role": "user", "content": "Fix expiry handling and verify it."},
                {"role": "user", "content": "Correction: keep the public API unchanged."},
            ]
            state = json.loads(reference[2]["content"].splitlines()[-2])
            assert state["plan"] == [{"step": "Inspect expiry", "status": "completed"},
                                     {"step": "Run tests", "status": "pending"}]
            assert state["changed_files"] == ["auth.py"]
            assert "PRIOR_TURN_REQUEST" not in json.dumps(reference)
            assert "UNRELATED_MCP_CATALOG" not in json.dumps(reference)
        assert "Correction: keep the public API unchanged." in json.dumps(requests[-1])
        history = agent._history.snapshot()
        assert sum(item.get("content") == "Fix expiry handling and verify it." for item in history) == 1
        assert "praxis_run_state" not in json.dumps(history)

        assert agent.run("Now review the deployment plan.") == "done"
        assert len(summaries) > first_turn_summaries
        for data in summaries[first_turn_summaries:]:
            reference = data["reference_context"]
            assert len(reference) == 2
            assert reference[0] == {"role": "user", "content": "Now review the deployment plan."}
            state = json.loads(reference[1]["content"].splitlines()[-2])
            assert state["plan"] == [] and state["changed_files"] == []
    finally:
        agent.close()


@pytest.mark.parametrize("previous_action", ["trim", "delete", "compress"])
def test_summary_uses_current_tool_previews_without_restoring_prior_removed_bodies(previous_action):
    calls = []

    def summarize(messages, instructions, **options):
        calls.append(deepcopy(messages))
        return "Verified findings with historical references."

    manager = ContextManager(
        model_limits=ModelLimits(32000), policy=ContextPolicy(max_tool_output_chars=100),
        summarizer=SimpleNamespace(summarize=summarize),
    )
    history = ConversationHistory([
        {"type": "function_call", "call_id": "old", "name": "read_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "old", "output": "OLD_REMOVED_BODY"},
    ])
    manager.build_view(history)
    if previous_action == "trim":
        trimmed = trim_tool_messages(manager._current_messages)
        manager.operations.record(ContextOperation("op1", "trim", ("m2",), updates=(trimmed[1],)))
    else:
        options = {"summary": "PREVIOUS_VERIFIED_CONCLUSION (m2)"} if previous_action == "compress" else {}
        manager.change_messages(previous_action, ["m2"], **options)
    raw_body = "HEAD_FINDING " + "noise " * 200 + "UNSHOWN_MIDDLE" + "noise " * 200 + " TAIL_FINDING"
    raw_output = json.dumps({"success": True, "output": raw_body})
    history.extend([
        {"type": "function_call", "call_id": "fresh", "name": "read_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "fresh", "output": raw_output},
        {"role": "user", "content": "Review the current evidence"},
    ])
    original = deepcopy(history.snapshot())
    main_view = manager.build_view(history)
    manager.recover_from_ptl(PromptTooLongError("input too long"))
    source = next(m for m in calls[0] if m["message_id"] == "m4")
    preview = source["payload"]["output"]
    main_output = next(item["output"] for item in main_view.items if item.get("call_id") == "fresh"
                       and item.get("type") == "function_call_output")
    assert main_output.split("\n", 1)[1] == preview
    assert "HEAD_FINDING" in preview and "TAIL_FINDING" in preview
    assert "UNSHOWN_MIDDLE" not in preview and "characters omitted" in preview
    assert "OLD_REMOVED_BODY" not in json.dumps(calls)
    if previous_action == "compress":
        assert "PREVIOUS_VERIFIED_CONCLUSION (m2)" in json.dumps(calls)
    assert history.snapshot() == original
    assert read_history(history, manager.operations.records, "m4", character_limit=None)["text"] == raw_output


def test_model_assisted_compress_receives_task_context_without_expanding_replaced_history():
    requests = []

    def summarize(items, **options):
        requests.append(json.loads(items[0]["content"]))
        return ModelResponse((), "Expiry condition is reversed (m1).", ())

    history = ConversationHistory([
        {"role": "assistant", "content": "The expiry condition is reversed."},
        {"role": "user", "content": "Fix expiry handling without changing the API."},
    ])
    manager = ContextManager(summarizer=ModelContextSummarizer(
        SimpleNamespace(create_response=summarize), ModelLimits(32000)))
    state = {"role": "developer", "content": "Inspection completed; tests pending."}
    manager.build_view(history, transient_items=(state,))
    original = deepcopy(history.snapshot())
    manager.change_messages("compress", ["m1"], instructions="Preserve the finding")
    assert requests[0]["reference_context"] == [original[-1], state]
    assert [message["message_id"] for message in requests[0]["selected_messages"]] == ["m1"]
    assert manager.operations.records[0].message_ids == ("m1",)
    assert history.snapshot() == original
    view = manager.build_view(history, transient_items=(state,))
    assert sum(item == state for item in view.items) == 1
    assert "Fix expiry handling without changing the API." in json.dumps(view.items)
