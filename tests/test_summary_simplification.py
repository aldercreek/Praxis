import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.responses import ResponseIncompleteEvent

from support.models import model_config
from praxis_agent.context.capacity import ContextPolicy
from praxis_agent.context.manager import ContextManager
from praxis_agent.context.model_metadata import ModelLimits
from praxis_agent.context.models import ConversationHistory
from praxis_agent.context.summarizer import ModelContextSummarizer, SUMMARY_INSTRUCTIONS
from praxis_agent.model import ModelClientError, ModelResponse, OpenAIModelClient
from praxis_agent.model_errors import OutputTruncatedError, SummaryError


class Client:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def create_response(self, items, **kwargs):
        self.calls.append((deepcopy(items), kwargs))
        return ModelResponse((), self.text, ())


def test_summary_uses_output_cap_and_a_shorter_prompt_target():
    client = Client("Verified tolerance is 0.001.")
    summarizer = ModelContextSummarizer(client, ModelLimits(128000))
    messages = [{"role": "user", "content": "Keep the verified tolerance."}]
    saved = deepcopy(messages)
    assert summarizer.summarize(messages, "Preserve tolerance", target_tokens=120) == client.text.strip()
    assert len(client.calls) == 1
    items, options = client.calls[0]
    assert options == {"tools": (), "max_output_tokens": 16000}
    assert json.loads(items[0]["content"])["summary_token_budget"] == 120
    assert json.loads(items[0]["content"])["compression_instructions"] == "Preserve tolerance"
    assert messages == saved


def test_input_chunking_keeps_all_material_and_carries_summary():
    client = Client("Verified checkpoint")
    summarizer = ModelContextSummarizer(client, ModelLimits(32000), input_limit_tokens=1400)
    messages = [{"message_id": "m1", "role": "user", "content": "evidence " * 3000}]
    reference = [{"role": "user", "content": "Check expiry handling; preserve the public API."},
                 {"role": "developer", "content": "auth.py changed; tests pending."}]
    assert summarizer.summarize(messages, "Keep evidence", reference_context=reference) == client.text
    assert len(client.calls) > 1
    fragments = []
    offset = 0
    for index, (items, options) in enumerate(client.calls):
        data = json.loads(items[0]["content"])
        assert data["previous_summary"] == (client.text if index else "")
        assert data["reference_context"] == reference
        assert summarizer.estimator.estimate_request(
            instructions=SUMMARY_INSTRUCTIONS, input_items=items, tools=(),
        ) <= summarizer.input_limit_tokens
        assert data["compression_instructions"] == "Keep evidence"
        assert options == {"tools": (), "max_output_tokens": 4000}
        for fragment in data["selected_messages"]:
            assert fragment["serialized_offset"] == offset
            fragments.append(fragment["text"])
            offset += len(fragment["text"])
    assert json.loads("".join(fragments)) == messages[0]


def test_oversized_reference_context_fails_before_sending_a_summary_request():
    client = Client("checkpoint")
    summarizer = ModelContextSummarizer(client, ModelLimits(32000), input_limit_tokens=1400)
    with pytest.raises(SummaryError, match="reference context.*input budget"):
        summarizer.summarize(
            [{"role": "assistant", "content": "Verified finding"}], "",
            reference_context=[{"role": "user", "content": "requirements " * 3000}],
        )
    assert not client.calls


def test_reference_context_omits_binary_and_reasoning_without_mutating_the_original():
    client = Client("checkpoint")
    summarizer = ModelContextSummarizer(client, ModelLimits(32000))
    reference = [
        {"type": "reasoning", "summary": [{"text": "PRIVATE_REASONING"}]},
        {"role": "user", "reasoning_content": "PRIVATE_REASONING", "content": [
            {"type": "input_text", "text": "Check the attached evidence"},
            {"type": "input_image", "image_url": "data:image/png;base64,IMAGE_BYTES"},
            {"type": "input_file", "file_data": "data:application/pdf;base64,FILE_BYTES"},
        ]},
    ]
    original = deepcopy(reference)
    summarizer.summarize([{"role": "assistant", "content": "Observation"}], "",
                         reference_context=reference)
    data = json.loads(client.calls[0][0][0]["content"])
    sanitized = json.dumps(data["reference_context"])
    assert len(data["reference_context"]) == 1
    assert "Check the attached evidence" in sanitized
    assert "input_image attachment" in sanitized and "input_file attachment" in sanitized
    for private in ("PRIVATE_REASONING", "IMAGE_BYTES", "FILE_BYTES"):
        assert private not in sanitized
    assert reference == original


@pytest.mark.parametrize("direct", [True, False])
def test_active_compression_rejects_overflow_without_saving(direct):
    client = Client("evidence " * 5000)
    manager = ContextManager(
        model_limits=ModelLimits(8000),
        summarizer=ModelContextSummarizer(client, ModelLimits(8000)),
    )
    history = ConversationHistory([{"role": "user", "content": "Original requirement"}])
    original = deepcopy(history.snapshot())
    manager.build_view(history)
    kwargs = {"summary": client.text} if direct else {"instructions": "Preserve requirements"}
    with pytest.raises(ValueError, match="compacted context"):
        manager.change_messages("compress", ["m1"], **kwargs)
    assert len(client.calls) == (0 if direct else 1)
    assert not manager.operations.records
    assert history.snapshot() == original


def test_active_compression_counts_tools_runtime_and_pending_operations():
    manager = ContextManager(model_limits=ModelLimits(8000), policy=ContextPolicy(output_reserve_tokens=4000))
    history = ConversationHistory([
        {"role": "user", "content": "First requirement"},
        {"role": "user", "content": "Second requirement"},
    ])
    manager.build_view(
        history,
        transient_items=({"role": "developer", "content": "state " * 500},),
        tool_definitions=({"name": "example", "description": "definition " * 300},),
    )
    manager.change_messages("compress", ["m1"], summary="summary " * 400)
    with pytest.raises(ValueError, match="compacted context"):
        manager.change_messages("compress", ["m2"], summary="summary " * 400)
    assert len(manager.operations.records) == 1


def test_automatic_overflow_preserves_history_on_summary_failure():
    client = Client("evidence " * 5000)
    limits = ModelLimits(8000)
    manager = ContextManager(
        model_limits=limits,
        policy=ContextPolicy(output_reserve_tokens=4000),
        summarizer=ModelContextSummarizer(client, limits, input_limit_tokens=20000),
    )
    history = ConversationHistory([{"role": "assistant", "content": "original " * 1800},
                                   {"role": "user", "content": "Continue"}])
    original = deepcopy(history.snapshot())
    state = [{"role": "developer", "content": "Recorded task state"}]
    with pytest.raises(SummaryError, match="input budget"):
        manager.build_view(history, transient_items=state)
    assert len(client.calls) == 1
    data = json.loads(client.calls[0][0][0]["content"])
    assert data["compression_instructions"] == ""
    assert data["reference_context"] == [original[-1], *state]
    assert "Recorded task state" not in json.dumps(data["selected_messages"])
    assert not manager.operations.records
    assert history.snapshot() == original


@pytest.mark.parametrize("reason,expected", [
    ("max_output_tokens", OutputTruncatedError), ("length", OutputTruncatedError),
    ("content_filter", ModelClientError),
])
def test_incomplete_generation_is_not_accepted_as_summary(monkeypatch, reason, expected):
    event = ResponseIncompleteEvent.model_validate({
        "type": "response.incomplete", "sequence_number": 1,
        "response": {
            "id": "r1", "created_at": 0, "model": "example", "object": "response",
            "output": [], "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
        },
    })
    event.response.incomplete_details.reason = reason
    class Stream:
        async def __aiter__(self):
            yield event

        async def close(self):
            pass

    sdk = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=Stream())), close=AsyncMock())
    monkeypatch.setattr("praxis_agent.model.AsyncOpenAI", lambda **kwargs: sdk)
    client = OpenAIModelClient(model_config(api_key="test", model="test"))
    try:
        with pytest.raises(expected, match="incomplete") as error:
            client.create_response([], [])
        assert type(error.value) is expected
        assert error.value.code == reason
    finally:
        client.close()
