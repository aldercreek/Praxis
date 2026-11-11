from dataclasses import replace
import json
from math import ceil
from types import SimpleNamespace

import pytest

from support.models import model_config
from praxis_agent.context.capacity import ContextPolicy, HeuristicTokenEstimator
from praxis_agent.context.model_metadata import ModelLimits
from praxis_agent.context.summarizer import ModelContextSummarizer, SUMMARY_INSTRUCTIONS
from praxis_agent.factory import _create_context_manager
from praxis_agent.model import ModelResponse


@pytest.mark.parametrize("window,reserve", [
    (128000, 16000), (128001, 16000), (160000, 16000), (200000, 20000),
    (256000, 25600), (256001, 25600), (320000, 32000), (512000, 32000),
])
def test_main_output_reserve_scales_with_window(window, reserve):
    limits = ModelLimits(window)
    policy = ContextPolicy.for_model(limits)
    assert policy.output_reserve_tokens == reserve
    assert policy.effective_input_limit(limits) == int(window * 0.95) - reserve
    # History growth changes pressure, not the capacity reserved for the next response.
    assert policy.assess(limits, 10000).effective_input_limit_tokens == policy.assess(limits, 90000).effective_input_limit_tokens


@pytest.mark.parametrize("boundary", [128000, 160000, 256000, 320000, 512000])
def test_input_capacity_does_not_drop_at_budget_boundaries(boundary):
    capacities = []
    for window in range(boundary - 50, boundary + 51):
        limits = ModelLimits(window)
        capacities.append(ContextPolicy.for_model(limits).effective_input_limit(limits))
    assert all(0 <= after - before <= 1 for before, after in zip(capacities, capacities[1:]))


def test_main_reserve_and_summary_output_are_independent_configurations():
    config = model_config(model_limits=ModelLimits(160000),
        context_policy=ContextPolicy.for_model(ModelLimits(160000), output_reserve_tokens=20000, summary_output_tokens=12000))
    policy = config.context_policy
    assert policy.output_reserve_tokens == 20000
    assert policy.summary_output_tokens == 12000
    assert policy.effective_input_limit(config.model_limits) == 132000
    assert replace(policy, summary_output_tokens=24000).effective_input_limit(config.model_limits) == 132000
    assert replace(policy, output_reserve_tokens=16000).summary_output_tokens == 12000


def test_provider_output_cap_bounds_reserve_and_summary_generation():
    limits = ModelLimits(128000, max_output_tokens=8000)
    assert ContextPolicy.for_model(limits).effective_input_limit(limits) == 113600
    requests = []
    def respond(items, **options):
        requests.append(options)
        return ModelResponse((), "Verified evidence.", ())
    ModelContextSummarizer(SimpleNamespace(create_response=respond), limits).summarize(
        [{"role": "assistant", "content": "Evidence"}], "", max_output_tokens=16000,
    )
    assert requests == [{"tools": (), "max_output_tokens": 8000}]


def test_summary_chunk_margin_covers_three_percent_underestimate_without_losing_material():
    requests = []
    estimator = HeuristicTokenEstimator()
    limits = ModelLimits(128000, max_input_tokens=20000)
    reference = [{"role": "user", "content": "Retain the verified amount and its source."}]
    material = {"message_id": "m1", "role": "assistant", "content": "Evidence 8271: amount 42. " * 6000}

    def respond(items, **options):
        raw = estimator.estimate_request(instructions=SUMMARY_INSTRUCTIONS, input_items=items, tools=())
        assert ceil(raw * 1.03) <= limits.max_input_tokens
        assert options["max_output_tokens"] == 16000
        requests.append(json.loads(items[0]["content"]))
        return ModelResponse((), "Amount is 42 (m1).", ())

    ModelContextSummarizer(SimpleNamespace(create_response=respond), limits).summarize(
        [material], "", reference_context=reference, target_tokens=1500,
    )
    assert len(requests) > 1
    fragments = []
    for index, request in enumerate(requests):
        assert request["reference_context"] == reference
        assert request["previous_summary"] == ("Amount is 42 (m1)." if index else "")
        assert request["summary_token_budget"] == 1500
        fragments.extend(request["selected_messages"])
    assert json.loads("".join(part["text"] for part in fragments)) == material


def test_factory_does_not_limit_summary_input_using_the_main_output_reserve():
    requests = []
    def respond(items, **options):
        requests.append((items, options))
        return ModelResponse((), "Verified evidence (m1).", ())
    limits = ModelLimits(160000)
    sources = [{"message_id": "m1", "role": "assistant", "content": "evidence " * 60000}]
    batches = []
    for reserve in (16000, 32000):
        config = model_config(api_key="test", model="example",
            model_limits=ModelLimits(160000),
            context_policy=ContextPolicy.for_model(limits, output_reserve_tokens=reserve))
        manager = _create_context_manager(config, "", SimpleNamespace(create_response=respond), config)
        requests.clear()
        manager._summarizer.summarize(sources, "")
        assert len(requests) > 1
        assert all(options["max_output_tokens"] == 24000 for _, options in requests)
        batches.append(list(requests))
    assert batches[0] == batches[1]
