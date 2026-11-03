import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from praxis_agent.agent import Agent
from praxis_agent.context.capacity import HeuristicTokenEstimator
from praxis_agent.context.manager import ContextManager
from praxis_agent.context.model_catalog import ModelLimits
from praxis_agent.events import ModelRequestCompleted, ModelRequestStarted
from praxis_agent.execution import ExecutionStopped
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.model_errors import IncompleteResponseError, ModelClientError, OutputTruncatedError, PromptTooLongError
from praxis_agent.tools.registry import Tool, ToolRegistry


@pytest.mark.parametrize("failure", ["empty", "truncated", "interrupted"])
def test_response_retries_reset_after_success_and_do_not_repeat_tools(tmp_path, failure):
    requests, tool_runs, events = [], [], []
    item = {"type": "function_call", "call_id": "c", "name": "inspect", "arguments": "{}"}

    def request(items, tools, **options):
        requests.append(deepcopy(items))
        assert "max_output_tokens" not in options
        if len(requests) not in {3, 6}:
            if failure == "empty":
                return ModelResponse(({"role": "assistant", "content": " \n"},), " \n", ())
            options["on_text_delta"]("UNFINISHED_RESPONSE")
            error = OutputTruncatedError if failure == "truncated" else IncompleteResponseError
            raise error("response incomplete")
        if len(requests) == 3:
            return ModelResponse((item,), "", (FunctionCall("c", "inspect", "{}"),))
        return ModelResponse(({"role": "assistant", "content": "done"},), "done", ())

    registry = ToolRegistry()
    registry.register(Tool("inspect", "inspect once", {"type": "object", "properties": {}},
                           lambda context: tool_runs.append(1) or "verified"))
    manager = ContextManager(model_limits=ModelLimits(32000))
    agent = Agent(SimpleNamespace(create_response=request), registry, tmp_path, context_manager=manager,
                  event_sink=SimpleNamespace(emit=events.append), bypass_approvals=True)
    try:
        assert agent.run("Inspect the task") == "done"
        assert tool_runs == [1]
        assert len(requests) == 6
        hints = [[item for item in items if item.get("role") == "developer"
                  and "previous response was incomplete" in item.get("content", "")] for items in requests]
        assert [len(items) for items in hints] == [0, 1, 1, 0, 1, 1]
        assert requests[1] == requests[2] and requests[4] == requests[5]
        saved = json.dumps(agent._history.snapshot())
        assert "UNFINISHED_RESPONSE" not in saved and "previous response was incomplete" not in saved
        assert "UNFINISHED_RESPONSE" not in json.dumps(requests)
        assert sum(item.get("type") == "function_call_output" for item in agent._history.snapshot()) == 1
        assert len([event for event in events if isinstance(event, ModelRequestCompleted)]) == 2
        for event in events:
            if isinstance(event, ModelRequestStarted):
                measured = HeuristicTokenEstimator().estimate_request(
                    instructions="", input_items=event.input_items, tools=registry.definitions())
                assert abs(event.context_status.estimated_input_tokens - measured) <= 1
    finally:
        agent.close()


@pytest.mark.parametrize("failure", ["empty", "truncated", "interrupted"])
def test_incomplete_response_stops_after_two_retries(tmp_path, failure):
    requests = []

    def request(items, tools, **options):
        requests.append(deepcopy(items))
        if failure == "empty":
            return ModelResponse((), "", ())
        error = OutputTruncatedError if failure == "truncated" else IncompleteResponseError
        raise error("response incomplete")

    agent = Agent(SimpleNamespace(create_response=request), ToolRegistry(), tmp_path)
    try:
        with pytest.raises(IncompleteResponseError):
            agent.run("continue")
        assert len(requests) == 3
        assert requests[1] == requests[2]
        assert len(agent._history.snapshot()) == 1
    finally:
        agent.close()


@pytest.mark.parametrize("failure", [
    ModelClientError("unauthorized", code="401"),
    ModelClientError("invalid parameter", code="400"),
    ModelClientError("filtered", code="content_filter"),
    ModelClientError("service unavailable after SDK retries"),
    ExecutionStopped("cancelled"),
    RuntimeError("internal error"),
])
def test_response_recovery_does_not_retry_unrelated_errors(tmp_path, failure):
    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        raise failure

    agent = Agent(SimpleNamespace(create_response=request), ToolRegistry(), tmp_path)
    try:
        with pytest.raises(type(failure)):
            agent.run("continue")
        assert len(calls) == 1
        assert len(agent._history.snapshot()) == 1
    finally:
        agent.close()


@pytest.mark.parametrize("last_failure", [None, PromptTooLongError, IncompleteResponseError])
def test_response_and_main_ptl_budgets_remain_independent(tmp_path, last_failure):
    requests, summaries = [], []
    failures = [IncompleteResponseError, PromptTooLongError, OutputTruncatedError, PromptTooLongError,
                last_failure]

    def request(items, tools, **options):
        requests.append(deepcopy(items))
        failure = failures[len(requests) - 1]
        if failure is not None:
            raise failure("request failed")
        return ModelResponse((), "done", ())

    def summarize(*args, **options):
        summaries.append(1)
        return "Verified checkpoint"

    manager = ContextManager(model_limits=ModelLimits(32000),
                             summarizer=SimpleNamespace(summarize=summarize))
    agent = Agent(SimpleNamespace(create_response=request), ToolRegistry(), tmp_path, context_manager=manager)
    agent.restore_history([{"role": "assistant", "content": "Original evidence"}])
    try:
        if last_failure is None:
            assert agent.run("continue") == "done"
        else:
            with pytest.raises(last_failure):
                agent.run("continue")
        assert len(requests) == 5
        assert len(summaries) == 2
        assert "previous response was incomplete" not in json.dumps(agent._history.snapshot())
    finally:
        agent.close()
