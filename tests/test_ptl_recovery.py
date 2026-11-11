import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from praxis_agent.agent import Agent
from praxis_agent.context.capacity import ContextPolicy, HeuristicTokenEstimator
from praxis_agent.context.manager import ContextManager
from praxis_agent.context.model_metadata import ModelLimits
from praxis_agent.context.summarizer import ModelContextSummarizer
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.model_errors import (
    IncompleteResponseError, OutputTruncatedError, PromptTooLongError, SummaryError, SummaryTooLongError,
    model_request_error,
)
from praxis_agent.tools.registry import Tool, ToolRegistry


@pytest.mark.parametrize("local_overflow,still_too_long", [(True, False), (False, False), (True, True)])
def test_main_recovery_refines_only_complete_candidate(tmp_path, local_overflow, still_too_long):
    summaries, requests = [], []
    candidate = "Verified evidence; task pending. " * 180
    final = candidate if still_too_long else "Verified evidence retained; validation is next."

    def summarize(messages, instructions, **options):
        summaries.append((deepcopy(messages), options))
        return candidate if len(summaries) == 1 else final

    def request(items, tools, **options):
        requests.append(deepcopy(items))
        if len(requests) == 1:
            assert not summaries  # This test enters through a provider PTL, not automatic compaction.
            raise PromptTooLongError("input too long")
        if not local_overflow and len(requests) == 2:
            raise PromptTooLongError("input too long")
        return ModelResponse((), "done", ())

    manager = ContextManager(
        model_limits=ModelLimits(8000), policy=ContextPolicy(output_reserve_tokens=3000, summary_output_tokens=3000),
        instructions="Fixed rule. " * (750 if local_overflow else 300),
        summarizer=SimpleNamespace(summarize=summarize),
    )
    agent = Agent(SimpleNamespace(create_response=request), ToolRegistry(), tmp_path, context_manager=manager)
    history = [
        {"role": "system", "content": "SYSTEM_FIXED"},
        {"role": "developer", "content": "DEVELOPER_FIXED"},
        {"role": "user", "content": "OLD_REQUEST"},
        {"role": "assistant", "content": "ORIGINAL_HISTORY"},
        {"type": "function_call", "call_id": "c", "name": "inspect", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c", "output": json.dumps({
            "success": False, "output": {"exit_code": 1, "stderr": "permission denied"},
        })},
        {"type": "function_call", "call_id": "d", "name": "inspect", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "d",
         "output": json.dumps({"success": True, "output": "ORIGINAL_TOOL_EVIDENCE"})},
    ]
    agent.restore_history(history)
    before = deepcopy(agent._history.snapshot())
    try:
        if still_too_long:
            with pytest.raises(SummaryTooLongError):
                agent.run("CURRENT_REQUEST")
            assert len(requests) == 1
            assert not manager.operations.records
        else:
            assert agent.run("CURRENT_REQUEST") == "done"
            assert len(requests) == (2 if local_overflow else 3)
            assert len(manager.operations.records) == (1 if local_overflow else 2)
            assert "CURRENT_REQUEST" in json.dumps(requests[-1])
        assert len(summaries) == 2
        first_sources = json.dumps(summaries[0][0])
        assert "ORIGINAL_HISTORY" in first_sources and "OLD_REQUEST" in first_sources
        assert "permission denied" in first_sources and '"message_id": "m6"' in first_sources
        assert "ORIGINAL_TOOL_EVIDENCE" in first_sources
        for omitted in ("CURRENT_REQUEST", "SYSTEM_FIXED", "DEVELOPER_FIXED"):
            assert omitted not in first_sources
        assert summaries[1][0] == [{"role": "assistant", "content": candidate.strip()}]
        first, second = (options for _, options in summaries)
        assert first["max_output_tokens"] == second["max_output_tokens"] == 3000
        assert 0 < second["target_tokens"] <= first["target_tokens"] // 2
        assert first["reference_context"] == second["reference_context"]
        reference = json.dumps(first["reference_context"])
        assert "CURRENT_REQUEST" in reference and "praxis_run_state" in reference
        assert "OLD_REQUEST" not in reference
        assert agent._history.snapshot()[:len(before)] == before
    finally:
        agent.close()


def test_provider_excess_can_rule_out_further_summary_reduction(tmp_path):
    summaries, requests = [], []

    def summarize(messages, instructions, **options):
        summaries.append(messages)
        return "This complete summary is already very small."

    def request(*args, **kwargs):
        requests.append(1)
        if len(requests) == 1:
            raise PromptTooLongError("input too long")
        # Local estimates fit, but the provider says more must be removed than the whole summary.
        raise PromptTooLongError("input too long", requested_tokens=15000, limit_tokens=12000)

    manager = ContextManager(model_limits=ModelLimits(32000), policy=ContextPolicy(output_reserve_tokens=4096, summary_output_tokens=4096),
                             summarizer=SimpleNamespace(summarize=summarize))
    agent = Agent(SimpleNamespace(create_response=request), ToolRegistry(), tmp_path, context_manager=manager)
    agent.restore_history([{"role": "assistant", "content": "Original task evidence."}])
    try:
        with pytest.raises(SummaryError, match="fixed context is too large"):
            agent.run("continue")
        assert len(requests) == 2 and len(summaries) == 1
        assert len(manager.operations.records) == 1
        assert "already very small" in manager.operations.records[0].replacement["payload"]["content"]
    finally:
        agent.close()


@pytest.mark.parametrize("message,details,expected", [
    ("maximum context length is 12,000 tokens. However, you requested 12,345 tokens "
     "(11000 in the messages, 1345 in the completion)", None, 346),
    ("prompt is too long: 12345 tokens > 12000 maximum", None, 346),
    ("input too long", {"error": {"details": {"input_tokens": 12345, "max_input_tokens": 12000}}}, 346),
    ("input too long", {"requested_tokens": 12345, "max_context_tokens": 12000}, 346),
    ("maximum context length is 12000", None, None),
    ("input too long", {"input_tokens": 12345, "max_context_tokens": 12000}, None),
    ("input too long", {"input_tokens": 100, "max_input_tokens": 12000}, None),
])
def test_provider_usage_is_paired_without_guessing(message, details, expected):
    error = model_request_error(message, details=details)
    assert isinstance(error, PromptTooLongError)
    assert error.required_reduction == expected


def test_summary_ptl_restarts_from_first_material_and_keeps_generation_budget():
    calls, starts = [], []
    failures = 0

    def progress(attempt, chunk, count, response):
        if response is None:
            starts.append((attempt, chunk))

    def request(items, **options):
        nonlocal failures
        data = json.loads(items[0]["content"])
        calls.append((data, options))
        if data["previous_summary"] and failures < 2:
            failures += 1
            raise PromptTooLongError("input too long")
        return ModelResponse((), "Checkpoint", ())

    summarizer = ModelContextSummarizer(SimpleNamespace(create_response=request), ModelLimits(32000),
                                       input_limit_tokens=8000)
    material = {"message_id": "m1", "role": "assistant", "content": "evidence " * 6000}
    reference = [{"role": "user", "content": "Audit payments; do not modify files."},
                 {"role": "developer", "content": "Investigation complete; reconciliation pending."}]
    original = deepcopy(material)
    assert summarizer.summarize([material], "preserve evidence", target_tokens=500,
                                max_output_tokens=4000, on_progress=progress,
                                reference_context=reference) == "Checkpoint"
    assert failures == 2 and {attempt for attempt, _ in starts} == {1, 2, 3}
    final_fragments = []
    estimator = HeuristicTokenEstimator()
    first_sizes = []
    for (attempt, chunk), (data, options) in zip(starts, calls, strict=True):
        assert data["reference_context"] == reference
        assert data["summary_token_budget"] == 500
        assert options == {"tools": (), "max_output_tokens": 4000}
        if chunk == 1:
            assert data["previous_summary"] == ""
            assert data["selected_messages"][0]["serialized_offset"] == 0
            first_sizes.append(estimator.estimate_request(
                instructions="", input_items=({"role": "user", "content": json.dumps(data)},), tools=(),
            ))
        if attempt == 3:
            final_fragments.extend(data["selected_messages"])
    assert first_sizes[0] > first_sizes[1] > first_sizes[2]
    offset = 0
    for fragment in final_fragments:
        assert fragment["serialized_offset"] == offset
        offset += len(fragment["text"])
    assert json.loads("".join(fragment["text"] for fragment in final_fragments)) == original
    assert material == original


def test_summary_retry_counter_resets_only_for_a_new_task():
    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        raise PromptTooLongError("input too long")

    summarizer = ModelContextSummarizer(SimpleNamespace(create_response=request), ModelLimits(32000),
                                       input_limit_tokens=8000)
    for task in range(2):
        with pytest.raises(SummaryError) as error:
            summarizer.summarize([{"role": "assistant", "content": "evidence"}], "")
        assert not isinstance(error.value, PromptTooLongError)
        assert isinstance(error.value.__cause__, PromptTooLongError)
        assert len(calls) == (task + 1) * 3


@pytest.mark.parametrize("failure", [PromptTooLongError, OutputTruncatedError, IncompleteResponseError])
def test_summary_failure_does_not_trigger_another_main_recovery(tmp_path, failure):
    main_calls, summary_calls = [], []

    def main(*args, **kwargs):
        main_calls.append(1)
        raise PromptTooLongError("input too long")

    def summary(*args, **kwargs):
        summary_calls.append(1)
        raise failure("summary request failed")

    manager = ContextManager(model_limits=ModelLimits(32000),
        summarizer=ModelContextSummarizer(SimpleNamespace(create_response=summary), ModelLimits(32000)))
    agent = Agent(SimpleNamespace(create_response=main), ToolRegistry(), tmp_path, context_manager=manager)
    agent.restore_history([{"role": "assistant", "content": "Original evidence"}])
    try:
        with pytest.raises(SummaryError) as error:
            agent.run("continue")
        assert isinstance(error.value.__cause__, failure)
        assert len(main_calls) == 1
        assert len(summary_calls) == 3
        assert not manager.operations.records
    finally:
        agent.close()


@pytest.mark.parametrize("last_failure", [PromptTooLongError, OutputTruncatedError, IncompleteResponseError])
def test_summary_mixed_failures_share_two_retries_across_chunks_and_restarts(last_failure):
    calls, starts = [], []

    def progress(attempt, chunk, count, response):
        if response is None:
            starts.append((attempt, chunk))

    def request(items, **options):
        data = json.loads(items[0]["content"])
        calls.append((data, options))
        if data["previous_summary"]:
            failure = [PromptTooLongError, OutputTruncatedError, last_failure][starts[-1][0] - 1]
            raise failure("retry this summary pass")
        return ModelResponse((), "Intermediate checkpoint", ())

    summarizer = ModelContextSummarizer(SimpleNamespace(create_response=request), ModelLimits(32000),
                                       input_limit_tokens=8000)
    material = [{"role": "assistant", "content": "evidence " * 6000}]
    for task in range(2):
        calls.clear()
        starts.clear()
        with pytest.raises(SummaryError) as error:
            summarizer.summarize(material, "", target_tokens=500, max_output_tokens=4000,
                                 on_progress=progress)
        assert type(error.value.__cause__) is last_failure
        assert starts == [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
        first_chunks = [data for (attempt, chunk), (data, _) in zip(starts, calls) if chunk == 1]
        assert [data["summary_token_budget"] for data in first_chunks] == [500, 500, 250]
        assert all(data["previous_summary"] == "" for data in first_chunks)
        assert all(data["selected_messages"][0]["serialized_offset"] == 0 for data in first_chunks)
        assert len(first_chunks[0]["selected_messages"][0]["text"]) > len(
            first_chunks[1]["selected_messages"][0]["text"])
        assert all(options == {"tools": (), "max_output_tokens": 4000} for _, options in calls)


def test_main_recovery_counter_resets_after_a_successful_request(tmp_path):
    main_calls, summary_calls, tool_runs = [], [], []

    def request(*args, **kwargs):
        main_calls.append(1)
        if len(main_calls) in {1, 2, 4, 5}:
            raise PromptTooLongError("input too long")
        if len(main_calls) == 3:
            item = {"type": "function_call", "call_id": "c", "name": "inspect", "arguments": "{}"}
            return ModelResponse((item,), "", (FunctionCall("c", "inspect", "{}"),))
        return ModelResponse((), "done", ())

    def summarize(*args, **kwargs):
        summary_calls.append(1)
        return "Verified checkpoint"

    registry = ToolRegistry()
    registry.register(Tool("inspect", "read only", {"type": "object", "properties": {}},
                           lambda context: tool_runs.append(1) or "verified"))
    manager = ContextManager(model_limits=ModelLimits(32000), summarizer=SimpleNamespace(summarize=summarize))
    agent = Agent(SimpleNamespace(create_response=request), registry, tmp_path,
                  context_manager=manager, bypass_approvals=True)
    agent.restore_history([{"role": "assistant", "content": "Original evidence"}])
    try:
        assert agent.run("continue") == "done"
        assert len(main_calls) == 6 and len(summary_calls) == 4
        assert tool_runs == [1]
    finally:
        agent.close()
