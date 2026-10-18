import json
from copy import deepcopy

import pytest
from jsonschema import ValidationError, validate

from praxis_agent.context.capacity import ContextPolicy, HeuristicTokenEstimator
from praxis_agent.context.manager import ContextManager
from praxis_agent.context.model_catalog import ModelLimits
from praxis_agent.context.models import ConversationHistory
from praxis_agent.context.operations import ContextOperation, ContextOperations
from praxis_agent.context.retrieval import read_history
from praxis_agent.context.tool_output import preview_tool_outputs
from praxis_agent.context.tools import create_context_tools


class SummaryModel:
    def __init__(self, text="Verified evidence and unfinished work."):
        self.calls = []
        self.text = text

    def summarize(self, messages, instructions, **kwargs):
        self.calls.append((deepcopy(messages), instructions))
        return self.text


def test_direct_summary_needs_no_model_preserves_history_and_has_one_visible_copy():
    original = "Long original requirement; tolerance must be 0.001."
    summary = "Tolerance is 0.001; implementation pending."
    history = ConversationHistory([{"role": "user", "content": original}])
    manager = ContextManager()
    manager.build_view(history)
    receipt = manager.change_messages("compress", ["m1"], summary=summary)
    history.extend([
        {"type": "function_call", "call_id": "c", "name": "compress",
         "arguments": json.dumps({"message_ids": ["m1"], "summary": summary})},
        {"type": "function_call_output", "call_id": "c",
         "output": json.dumps({"success": True, "output": receipt})},
    ])
    saved = deepcopy(history.snapshot())
    view = manager.build_view(history)
    assert json.dumps(view.items).count(summary) == 1
    assert history.snapshot() == saved
    assert read_history(history, manager.operations.records, "m1")["text"] == original
    replay = ContextOperations()
    replay.replay([ContextOperation.from_dict({**op.to_dict(), "message_ids": list(op.message_ids)})
                   for op in manager.operations.records])
    assert json.dumps(replay.project(history.snapshot())).count(summary) == 1
    # Original function-call arguments are retained in the full history too.
    assert json.loads(history.snapshot()[1]["arguments"])["summary"] == summary


@pytest.mark.parametrize("options", [{}, {"summary": "a", "instructions": "b"},
                                    {"summary": " "}, {"instructions": ""}])
def test_compress_rejects_invalid_modes_without_recording(options):
    manager = ContextManager()
    manager.build_view(ConversationHistory([{"role": "user", "content": "task"}]))
    with pytest.raises(ValueError):
        manager.change_messages("compress", ["m1"], **options)
    assert not manager.operations.records


def test_compress_schema_and_existing_summary_model_mode():
    schema = next(t.parameters for t in create_context_tools() if t.name == "compress")
    for key in ("summary", "instructions"):
        validate({"message_ids": ["m1"], key: "keep requirement"}, schema)
    for options in ({}, {"summary": "a", "instructions": "b"}):
        with pytest.raises(ValidationError):
            validate({"message_ids": ["m1"], **options}, schema)
    model = SummaryModel()
    manager = ContextManager(summarizer=model)
    manager.build_view(ConversationHistory([{"role": "user", "content": "task"}]))
    manager.change_messages("compress", ["m1"], instructions="keep requirement")
    assert len(model.calls) == 1


def test_failed_compress_call_keeps_original_arguments_visible():
    items = [
        {"type": "function_call", "call_id": "c", "name": "compress",
         "arguments": '{"message_ids":["bad"],"summary":"failed draft"}'},
        {"type": "function_call_output", "call_id": "c", "output": '{"success":false,"output":"bad id"}'},
    ]
    assert ContextOperations().project(items)[0]["payload"] == items[0]


def test_preview_notice_is_first_and_history_remains_fully_readable():
    body = "begin\n" + "abc\n" * 1000
    raw = json.dumps({"success": True, "output": body})
    history = ConversationHistory([{"type": "function_call_output", "call_id": "c", "output": raw}])
    messages = ContextOperations().project(history.snapshot())
    preview = preview_tool_outputs(messages, 100)[0]["payload"]["output"]
    assert preview.startswith("[Tool result preview: success=true;")
    assert "saved as m1." in preview
    assert body[:50] in preview
    assert preview.endswith(body[-50:])
    assert "characters omitted" in preview
    assert "Use read_history" not in preview and "Manage context" not in preview
    assert read_history(history, [], "m1", character_limit=None)["text"] == raw


def test_usage_is_only_percentage_and_counts_all_request_parts():
    limits = ModelLimits(128000, max_output_tokens=16384)
    manager = ContextManager(model_limits=limits, instructions="system instructions " * 30)
    history = ConversationHistory([{"role": "user", "content": "task " * 90}])
    tools = [{"type": "function", "name": "example", "description": "tool information " * 40}]
    transient = [{"role": "developer", "content": "run state " * 20}]
    view = manager.build_view(history, transient_items=transient, tool_definitions=tools)
    assert view.items[-1]["content"] == f"Context usage: {view.context_status.usage_percent:g}%"
    estimate = HeuristicTokenEstimator().estimate_request(
        instructions=manager._instructions, input_items=view.items, tools=tools)
    assert abs(view.context_status.estimated_input_tokens - estimate) <= 2
    assert view.context_status.usage_percent == round(estimate / 105600 * 100, 1)


@pytest.mark.parametrize("ratio", [0.15, 0.25, 0.4])
def test_automatic_compaction_retains_token_budgeted_complete_tool_groups(ratio):
    model = SummaryModel()
    policy = ContextPolicy(desired_output_tokens=1000, recent_history_ratio=ratio)
    manager = ContextManager(model_limits=ModelLimits(12000), policy=policy,
                             instructions="fixed " * 200, summarizer=model)
    history = ConversationHistory()
    for i in range(50):
        history.extend([
            {"type": "function_call", "call_id": str(i), "name": "read_file", "arguments": "{}"},
            {"type": "function_call_output", "call_id": str(i), "output": "evidence " * 100},
        ])
    original = deepcopy(history.snapshot())
    transient = [{"role": "developer", "content": "state " * 60}]
    view = manager.build_view(history, transient_items=transient)
    assert len(model.calls) == len(manager.operations.records) == 1
    assert history.snapshot() == original
    assert view.context_status.estimated_input_tokens < view.context_status.effective_input_limit_tokens
    retained = [m for m in manager._current_messages if m["message_id"].startswith("m")
                and m["payload"].get("type") in ("function_call", "function_call_output")]
    assert retained
    assert [m["payload"]["call_id"] for m in retained[::2]] == [m["payload"]["call_id"] for m in retained[1::2]]
    fixed = manager._estimate_view(tuple(transient), ())[0].context_status
    target = int((fixed.effective_input_limit_tokens - fixed.estimated_input_tokens) * ratio)
    from praxis_agent.context.operations import render_message
    tail = tuple(render_message(m) for m in retained) + tuple(transient)
    size = manager._estimate_view(tail, ())[0].context_status.estimated_input_tokens - fixed.estimated_input_tokens
    assert 0 < size <= target
    assert target - size < 450  # Less than one additional complete interaction.


def test_oversized_summary_is_not_committed():
    manager = ContextManager(model_limits=ModelLimits(8000), summarizer=SummaryModel("x" * 30000))
    history = ConversationHistory([{"role": "user", "content": "x" * 30000}])
    with pytest.raises(ValueError, match="compacted context"):
        manager.build_view(history)
    assert not manager.operations.records
