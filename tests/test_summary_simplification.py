import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.responses import ResponseIncompleteEvent

from praxis_agent.config import ModelConfig
from praxis_agent.context.manager import ContextManager
from praxis_agent.context.model_catalog import ModelLimits
from praxis_agent.context.models import ConversationHistory
from praxis_agent.context.summarizer import ModelContextSummarizer
from praxis_agent.model import ModelClientError, ModelResponse, OpenAIModelClient


class Client:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def create_response(self, items, **kwargs):
        self.calls.append((deepcopy(items), kwargs))
        return ModelResponse((), self.text, ())


def test_long_summary_is_returned_once_without_output_cap_or_truncation():
    client = Client("evidence " * 10000)
    summarizer = ModelContextSummarizer(client, ModelLimits(128000))
    messages = [{"role": "user", "content": "Keep the verified tolerance."}]
    saved = deepcopy(messages)
    assert summarizer.summarize(messages, "Preserve tolerance") == client.text.strip()
    assert len(client.calls) == 1
    items, options = client.calls[0]
    assert options == {"tools": ()}
    assert json.loads(items[0]["content"])["compression_instructions"] == "Preserve tolerance"
    assert messages == saved


def test_input_chunking_keeps_all_material_and_carries_summary_and_state():
    client = Client("Verified checkpoint")
    summarizer = ModelContextSummarizer(client, ModelLimits(32000), input_limit_tokens=1400)
    messages = [{"message_id": "m1", "role": "user", "content": "evidence " * 3000}]
    state = [{"role": "developer", "content": "Current task state"}]
    assert summarizer.summarize(messages, "Keep evidence", runtime_state=state) == client.text
    assert len(client.calls) > 1
    fragments = []
    offset = 0
    for index, (items, options) in enumerate(client.calls):
        data = json.loads(items[0]["content"])
        assert data["previous_summary"] == (client.text if index else "")
        assert data["runtime_state"] == state
        assert data["compression_instructions"] == "Keep evidence"
        assert options == {"tools": ()}
        for fragment in data["selected_messages"]:
            assert fragment["serialized_offset"] == offset
            fragments.append(fragment["text"])
            offset += len(fragment["text"])
    assert json.loads("".join(fragments)) == messages[0]


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
    manager = ContextManager(model_limits=ModelLimits(8000))
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


def test_automatic_overflow_calls_summary_once_and_preserves_history_on_failure():
    client = Client("evidence " * 5000)
    limits = ModelLimits(8000)
    manager = ContextManager(
        model_limits=limits,
        summarizer=ModelContextSummarizer(client, limits, input_limit_tokens=20000),
    )
    history = ConversationHistory([{"role": "user", "content": "original " * 6000}])
    original = deepcopy(history.snapshot())
    state = [{"role": "developer", "content": "Recorded task state"}]
    with pytest.raises(ValueError, match="compacted context"):
        manager.build_view(history, transient_items=state)
    assert len(client.calls) == 1
    data = json.loads(client.calls[0][0][0]["content"])
    assert data["compression_instructions"] == ""
    assert data["runtime_state"] == state
    assert not manager.operations.records
    assert history.snapshot() == original


def test_incomplete_generation_is_not_accepted_as_summary(monkeypatch):
    event = ResponseIncompleteEvent.model_validate({
        "type": "response.incomplete", "sequence_number": 1,
        "response": {
            "id": "r1", "created_at": 0, "model": "example", "object": "response",
            "output": [], "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
        },
    })
    class Stream:
        async def __aiter__(self):
            yield event

        async def close(self):
            pass

    sdk = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=Stream())), close=AsyncMock())
    monkeypatch.setattr("praxis_agent.model.AsyncOpenAI", lambda **kwargs: sdk)
    client = OpenAIModelClient(ModelConfig(api_key="test", model="test"))
    try:
        with pytest.raises(ModelClientError, match="incomplete"):
            client.create_response([], [])
    finally:
        client.close()
