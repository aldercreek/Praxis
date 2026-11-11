import json

import httpx2
import pytest
from openai import AsyncOpenAI

from praxis_agent.chat_completions import chat_messages
from praxis_agent.context.model_metadata import ModelInputCapabilities
from praxis_agent.model import OpenAIModelClient
from praxis_agent.model_errors import IncompleteResponseError, ModelClientError, OutputTruncatedError
from support.models import model_config


def chunk(delta=None, finish=None, usage=None):
    return {"id": "chat-1", "object": "chat.completion.chunk", "created": 1, "model": "custom",
            "choices": [] if usage else [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
            "usage": usage}


def client_for(monkeypatch, streams, requests):
    def handle(request):
        requests.append((request, json.loads(request.content)))
        events = streams.pop(0)
        body = ''.join('data: '+json.dumps(e)+'\n\n' for e in events) + 'data: [DONE]\n\n'
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    monkeypatch.setattr('praxis_agent.model.AsyncOpenAI', lambda **options: AsyncOpenAI(
        **options, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle))))
    return OpenAIModelClient(model_config(protocol='chat_completions', api_key=''), instructions='system')


def test_chat_stream_tool_roundtrip_reasoning_usage_and_no_auth(monkeypatch):
    requests = []
    streams = [[
        chunk({'reasoning_content': 'Plan.'}),
        chunk({'tool_calls': [{'index': 0, 'id': 'call-1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"id":'}}]}),
        chunk({'tool_calls': [{'index': 0, 'function': {'arguments': '7}'}}]}),
        chunk(finish='tool_calls'),
        chunk(usage={'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14}),
    ], [chunk({'content': 'Found.'}), chunk(finish='stop')]]
    client = client_for(monkeypatch, streams, requests)
    tools = [{'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object', 'properties': {}}}]
    history = [{'role': 'user', 'content': 'Find 7'}]
    try:
        response = client.create_response(history, tools)
        assert response.function_calls[0].arguments == '{"id":7}'
        assert response.usage.total_tokens == 14
        history += list(response.output_items)
        history.append({'type': 'function_call_output', 'call_id': 'call-1', 'output': 'row 7'})
        deltas = []
        final = client.create_response(history, tools, on_text_delta=deltas.append, max_output_tokens=1000)
        assert final.output_text == 'Found.'
        assert deltas == ['Found.']
        request, data = requests[1]
        assert request.url.path == '/v1/chat/completions'
        assert 'authorization' not in request.headers
        assert data['max_completion_tokens'] == 1000
        assert data['messages'][2]['reasoning_content'] == 'Plan.'
        assert data['messages'][2]['tool_calls'][0]['id'] == 'call-1'
        assert data['messages'][3] == {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'row 7'}
        assert data['tools'][0]['function']['name'] == 'lookup'
    finally:
        client.close()


@pytest.mark.parametrize('finish,error', [('length', OutputTruncatedError), ('content_filter', ModelClientError), (None, IncompleteResponseError)])
def test_chat_rejects_partial_stream(monkeypatch, finish, error):
    client = client_for(monkeypatch, [[chunk({'content': 'partial'}), chunk(finish=finish)]], [])
    try:
        with pytest.raises(error):
            client.create_response([{'role': 'user', 'content': 'test'}], [])
    finally:
        client.close()


def test_chat_keeps_parallel_results_before_image_attachments():
    items = [
        {'type': 'function_call', 'call_id': 'a', 'name': 'read', 'arguments': '{}'},
        {'type': 'function_call', 'call_id': 'b', 'name': 'read', 'arguments': '{}'},
        {'type': 'function_call_output', 'call_id': 'a', 'output': [
            {'type': 'input_text', 'text': 'image A'}, {'type': 'input_image', 'image_url': 'data:image/png;base64,aA=='}]},
        {'type': 'function_call_output', 'call_id': 'b', 'output': 'result B'},
    ]
    messages = chat_messages('system', items, ModelInputCapabilities(('image/png',)))
    assert [m['role'] for m in messages] == ['system', 'assistant', 'tool', 'tool', 'user']
    assert len(messages[1]['tool_calls']) == 2
    assert messages[-1]['content'][0]['type'] == 'image_url'
