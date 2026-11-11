import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from praxis_agent.context.capacity import ContextPolicy, HeuristicTokenEstimator
from praxis_agent.context.manager import ContextManager
from praxis_agent.context.model_metadata import ModelLimits
from praxis_agent.context.models import ConversationHistory
from praxis_agent.context.summarizer import ModelContextSummarizer
from praxis_agent.model import ModelResponse
from praxis_agent.model_errors import SummaryError, SummaryTooLongError


@pytest.mark.parametrize("fixed_repeats,output_limit,long_summary", [
    (20, 1000, False),
    (2100, 1000, False),
    (20, 4000, True),
])
def test_summary_prompt_target_uses_window_and_remaining_complete_request_space(
    fixed_repeats, output_limit, long_summary,
):
    calls, fixed_estimates = [], []
    summary = "Verified finding (m1). " * (400 if long_summary else 1)

    class Estimator(HeuristicTokenEstimator):
        def estimate_request(self, **kwargs):
            tokens = super().estimate_request(**kwargs)
            if any(item.get("content", "").endswith("Use search_history/read_history for original details.\n")
                   for item in kwargs["input_items"]):
                fixed_estimates.append(tokens)
            return tokens

    def respond(items, **options):
        calls.append((json.loads(items[0]["content"]), options))
        return ModelResponse((), summary, ())

    limits = ModelLimits(12000)
    policy = ContextPolicy(output_reserve_tokens=output_limit, summary_output_tokens=output_limit)
    manager = ContextManager(
        model_limits=limits, policy=policy, instructions="Fixed rule. " * fixed_repeats,
        token_estimator=Estimator(),
        summarizer=ModelContextSummarizer(SimpleNamespace(create_response=respond), ModelLimits(128000)),
    )
    runtime = {"role": "developer", "content": "Validation is pending."}
    tools = ({"type": "function", "name": "inspect", "description": "Inspect task evidence."},)
    history = ConversationHistory([
        {"role": "assistant", "content": "Original evidence. " * 2000},
        {"role": "user", "content": "Continue the current task."},
    ])
    original = deepcopy(history.snapshot())
    view = manager.build_view(history, transient_items=(runtime,), tool_definitions=tools)
    assert len(calls) == 1
    data, options = calls[0]
    room = int(policy.effective_input_limit(limits) * 0.95) - max(fixed_estimates) - 1
    assert data["summary_token_budget"] == min(2400, room)
    assert (room < 2400) == (fixed_repeats == 2100)
    assert options == {"tools": (), "max_output_tokens": output_limit}
    if fixed_repeats == 20 and not long_summary:
        assert data["summary_token_budget"] > options["max_output_tokens"]
    if long_summary:
        assert HeuristicTokenEstimator().estimate_request(
            instructions="", input_items=({"role": "assistant", "content": summary},), tools=(),
        ) > data["summary_token_budget"]
    assert len(manager.operations.records) == 1
    assert manager.operations.records[0].replacement["payload"]["content"].endswith(summary.strip())
    assert view.context_status.estimated_input_tokens < int(policy.effective_input_limit(limits) * 0.95)
    assert data["reference_context"] == [original[-1], runtime]
    assert runtime in view.items
    assert "Continue the current task." in json.dumps(view.items)
    assert history.snapshot() == original


@pytest.mark.parametrize("safe_trim", [True, False])
@pytest.mark.parametrize("refinement", ["success", "oversized", "failure"])
def test_automatic_summary_refines_once_then_saves_only_a_usable_result(safe_trim, refinement):
    calls, events = [], []
    candidate = "Verified checkpoint. " * 500

    def summarize(messages, instructions, **options):
        calls.append((deepcopy(messages), options))
        if len(calls) == 1 or refinement == "oversized":
            return candidate
        if refinement == "failure":
            raise SummaryError("summary generation exhausted its retries")
        return "Verified checkpoint (m1); validation is pending."

    manager = ContextManager(
        model_limits=ModelLimits(12000),
        policy=ContextPolicy(output_reserve_tokens=4000, summary_output_tokens=4000, max_tool_output_chars=32000),
        instructions="Fixed rule. " * 900,
        summarizer=SimpleNamespace(summarize=summarize),
    )
    manager.event_sink = SimpleNamespace(emit=events.append)
    runtime = {"role": "developer", "content": "Validation is pending."}
    history = ConversationHistory([
        {"role": "assistant", "content": "Original evidence. " * (400 if safe_trim else 700)},
        {"type": "function_call", "call_id": "c", "name": "inspect", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c",
         "output": json.dumps({"success": True, "output": "ORIGINAL_TOOL_EVIDENCE " * 1600})},
        {"role": "user", "content": "Continue the current task."},
    ])
    original = deepcopy(history.snapshot())
    if refinement == "success" or safe_trim:
        view = manager.build_view(history, transient_items=(runtime,))
        assert view.context_status.usage_percent < 95
        expected_action = "compress" if refinement == "success" else "trim"
        assert [operation.action for operation in manager.operations.records] == [expected_action]
        assert events[-1].stage == "completed"
        assert sum(event.stage == "fallback" for event in events) == int(refinement != "success")
        assert candidate.strip() not in json.dumps(view.items)
        if refinement == "success":
            assert "Verified checkpoint (m1)" in json.dumps(view.items)
    else:
        error = SummaryTooLongError if refinement == "oversized" else SummaryError
        with pytest.raises(error):
            manager.build_view(history, transient_items=(runtime,))
        assert not manager.operations.records
        assert events[-1].stage == "failed"
    assert len(calls) == 2
    assert "ORIGINAL_TOOL_EVIDENCE" in json.dumps(calls[0][0])
    assert "context_omitted" not in json.dumps(calls[0][0])
    assert calls[1][0] == [{"role": "assistant", "content": candidate.strip()}]
    first, second = (options for _, options in calls)
    assert 0 < second["target_tokens"] <= first["target_tokens"] // 2
    assert first["max_output_tokens"] == second["max_output_tokens"] == 4000
    assert first["reference_context"] == second["reference_context"] == (original[-1], runtime)
    assert history.snapshot() == original


def test_fixed_context_with_no_summary_space_fails_without_calling_model():
    calls = []
    manager = ContextManager(
        model_limits=ModelLimits(12000), policy=ContextPolicy(output_reserve_tokens=1000, summary_output_tokens=1000),
        instructions="Fixed rule. " * 3000,
        summarizer=SimpleNamespace(summarize=lambda *args, **kwargs: calls.append(1) or "summary"),
    )
    with pytest.raises(SummaryError, match="fixed context leaves insufficient summary space"):
        manager.build_view(ConversationHistory([
            {"role": "assistant", "content": "Verified finding."},
            {"role": "user", "content": "Continue."},
        ]))
    assert not calls and not manager.operations.records
