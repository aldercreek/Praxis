import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from praxis_agent.agent import Agent
from praxis_agent.context.capacity import ContextPolicy
from praxis_agent.context.manager import ContextManager
from praxis_agent.context.model_metadata import ModelLimits
from praxis_agent.context.models import ConversationHistory
from praxis_agent.context.summarizer import ModelContextSummarizer
from praxis_agent.events import ContextCompaction
from praxis_agent.execution import ExecutionStopped
from praxis_agent.model import ModelResponse
from praxis_agent.model_errors import IncompleteResponseError, ModelClientError, OutputTruncatedError, SummaryError
from praxis_agent.tools.registry import ToolRegistry


def oversized_history():
    return [
        {"role": "assistant", "content": "Verified evidence. " * 400},
        {"type": "function_call", "call_id": "c", "name": "read_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c",
         "output": json.dumps({"success": True, "output": "trace " * 8000})},
    ]


@pytest.mark.parametrize("failure", ["empty", "truncated", "interrupted"])
@pytest.mark.parametrize("succeed_on", [2, 3, None])
def test_automatic_summary_recovers_or_saves_safe_trim_and_continues(tmp_path, failure, succeed_on):
    summary_calls, main_calls, events = [], [], []

    def summarize(items, **options):
        summary_calls.append((deepcopy(items), options))
        if len(summary_calls) == succeed_on:
            return ModelResponse((), "Verified evidence retained.", ())
        if failure == "empty":
            return ModelResponse((), " \n ", ())
        error = OutputTruncatedError if failure == "truncated" else IncompleteResponseError
        raise error("incomplete summary")

    def main(items, tools, **options):
        main_calls.append(deepcopy(items))
        return ModelResponse(({"role": "assistant", "content": "done"},), "done", ())

    limits = ModelLimits(12000)
    manager = ContextManager(
        model_limits=limits, policy=ContextPolicy(output_reserve_tokens=1000, summary_output_tokens=1000, max_tool_output_chars=32000),
        summarizer=ModelContextSummarizer(SimpleNamespace(create_response=summarize), ModelLimits(128000)),
    )
    agent = Agent(SimpleNamespace(create_response=main), ToolRegistry(), tmp_path,
                  context_manager=manager, event_sink=SimpleNamespace(emit=events.append))
    original = oversized_history()
    agent.restore_history(original)
    try:
        assert agent.run("Continue the current request") == "done"
        assert len(summary_calls) == (succeed_on or 3)
        assert len(main_calls) == 1
        assert agent._history.snapshot()[:len(original)] == tuple(original)
        assert agent.context_status.usage_percent < 95
        assert [op.action for op in manager.operations.records] == ["compress" if succeed_on else "trim"]
        assert "trace trace" not in json.dumps(main_calls[0])
        assert "Continue the current request" in json.dumps(main_calls[0])
        compactions = [event for event in events if isinstance(event, ContextCompaction)]
        assert len([event for event in compactions if event.stage == "fallback"]) == int(succeed_on is None)
        assert compactions[-1].stage == "completed"
        assert compactions[-1].request_count == len(summary_calls)
        if succeed_on is None:
            assert agent.context_status.usage_percent > 20
            result = next(message["payload"] for message in manager.operations.project(agent._history.snapshot())
                          if message["payload"].get("type") == "function_call_output")
            assert json.loads(result["output"])["success"] is True
        targets = [json.loads(items[0]["content"])["summary_token_budget"] for items, _ in summary_calls]
        assert all(options["max_output_tokens"] == 1000 for _, options in summary_calls)
        assert targets == ([2400 // 2 ** attempt for attempt in range(len(targets))]
                           if failure == "truncated" else [2400] * len(targets))
    finally:
        agent.close()


@pytest.mark.parametrize("trimmed_tokens", [9499, 9500, 9501])
def test_automatic_fallback_requires_complete_request_strictly_below_95_percent(trimmed_tokens):
    calls = []
    instructions = "FIXED_SYSTEM"
    tools = ({"type": "function", "name": "inspect", "description": "FIXED_TOOLS"},)
    runtime = {"role": "developer", "content": "FIXED_RUNTIME"}

    def estimate_request(*, instructions, input_items, tools):
        if not instructions:
            assert not tools and runtime not in input_items
            assert all(item.get("role") != "user" for item in input_items)
            return 5000  # Compressible history exceeds 20% of the model window.
        assert instructions == "FIXED_SYSTEM" and tools[0]["description"] == "FIXED_TOOLS"
        assert runtime in input_items
        results = [item for item in input_items if item.get("type") == "function_call_output"]
        if not results:
            return 1000  # Fixed context in the prospective summary request.
        return trimmed_tokens if '"context_omitted": true' in results[0]["output"] else 15000

    def summarize(*args, **kwargs):
        calls.append(1)
        return ModelResponse((), "", ())

    limits = ModelLimits(20000, max_input_tokens=10000)
    manager = ContextManager(
        model_limits=limits, instructions=instructions,
        token_estimator=SimpleNamespace(estimate_request=estimate_request),
        summarizer=ModelContextSummarizer(SimpleNamespace(create_response=summarize), limits),
    )
    history = ConversationHistory(oversized_history() + [{"role": "user", "content": "current request"}])
    original = deepcopy(history.snapshot())
    if trimmed_tokens < 9500:
        view = manager.build_view(history, transient_items=(runtime,), tool_definitions=tools)
        assert view.context_status.estimated_input_tokens == 9499
        assert [op.action for op in manager.operations.records] == ["trim"]
    else:
        with pytest.raises(SummaryError, match="after two retries"):
            manager.build_view(history, transient_items=(runtime,), tool_definitions=tools)
        assert not manager.operations.records
    assert len(calls) == 3
    assert history.snapshot() == original


@pytest.mark.parametrize("failure", [ExecutionStopped("cancelled"), RuntimeError("program error")])
def test_automatic_compaction_never_swallows_cancellation_or_internal_errors(failure):
    calls = []

    def summarize(*args, **kwargs):
        calls.append(1)
        raise failure

    limits = ModelLimits(12000)
    manager = ContextManager(
        model_limits=limits, policy=ContextPolicy(output_reserve_tokens=1000, summary_output_tokens=1000, max_tool_output_chars=32000),
        summarizer=ModelContextSummarizer(SimpleNamespace(create_response=summarize), limits),
    )
    history = ConversationHistory(oversized_history() + [{"role": "user", "content": "continue"}])
    original = deepcopy(history.snapshot())
    with pytest.raises(type(failure)):
        manager.build_view(history)
    assert len(calls) == 1
    assert not manager.operations.records
    assert history.snapshot() == original


def test_exhausted_sdk_failure_is_not_retried_by_summary_but_safe_trim_can_continue():
    calls = []

    def summarize(*args, **kwargs):
        calls.append(1)
        raise ModelClientError("service unavailable after SDK retries")

    limits = ModelLimits(12000)
    manager = ContextManager(
        model_limits=limits, policy=ContextPolicy(output_reserve_tokens=1000, summary_output_tokens=1000, max_tool_output_chars=32000),
        summarizer=ModelContextSummarizer(SimpleNamespace(create_response=summarize), limits),
    )
    view = manager.build_view(ConversationHistory(
        oversized_history() + [{"role": "user", "content": "continue"}]))
    assert len(calls) == 1
    assert 20 < view.context_status.usage_percent < 95
    assert [op.action for op in manager.operations.records] == ["trim"]


@pytest.mark.parametrize("actual,expected", [(14936, 30040), (25064, 35104)])
def test_usage_calibration_can_only_raise_the_current_raw_estimate(actual, expected):
    raw = 20000
    manager = ContextManager(model_limits=ModelLimits(128000), token_estimator=SimpleNamespace(
        estimate_request=lambda **kwargs: raw))
    history = ConversationHistory([{"role": "user", "content": "task"}])
    assert manager.build_view(history).context_status.estimated_input_tokens == raw
    manager.record_actual_input_tokens(actual)
    raw = 30040
    assert manager.build_view(history).context_status.estimated_input_tokens == expected
    # The next anchor uses raw 30,040, not its previously calibrated value.
    manager.record_actual_input_tokens(32040)
    raw = 31040
    assert manager.build_view(history).context_status.estimated_input_tokens == 33040
    manager.reset()
    assert manager.build_view(history).context_status.estimated_input_tokens == raw


def test_previous_overestimate_cannot_prevent_automatic_compaction():
    raw = 20000
    calls = []

    def estimate_request(**kwargs):
        if "Conversation summary of" in json.dumps(kwargs["input_items"]):
            return 1000
        return raw

    def summarize(*args, **kwargs):
        calls.append(1)
        return "Verified checkpoint"

    manager = ContextManager(
        model_limits=ModelLimits(32000), policy=ContextPolicy(output_reserve_tokens=4096, summary_output_tokens=4096),
        token_estimator=SimpleNamespace(estimate_request=estimate_request),
        summarizer=SimpleNamespace(summarize=summarize),
    )
    history = ConversationHistory([{"role": "assistant", "content": "old evidence"},
                                   {"role": "user", "content": "current task"}])
    manager.build_view(history)
    manager.record_actual_input_tokens(14936)
    raw = 30040  # Previously reduced to 24,976, missing the 24,988 trigger.
    assert manager.build_view(history).context_status.estimated_input_tokens == 1000
    assert len(calls) == 1
    assert [op.action for op in manager.operations.records] == ["compress"]
    assert manager._input_token_anchor is None
