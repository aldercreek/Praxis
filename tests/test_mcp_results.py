import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import (
    AudioContent, BlobResourceContents, CallToolResult, EmbeddedResource,
    ImageContent, ResourceLink, TextContent, TextResourceContents,
)

from praxis_agent.agent import Agent, _build_function_call_output
from praxis_agent.config import ModelConfig
from praxis_agent.context.capacity import HeuristicTokenEstimator
from praxis_agent.context.model_catalog import MODEL_INPUT_CATALOG, MODEL_LIMIT_CATALOG
from praxis_agent.context.summarizer import summary_input, summary_sources
from praxis_agent.context.tool_output import preview_tool_outputs
from praxis_agent.desktop.event_adapter import Transcript, display_event
from praxis_agent.events import ToolFinished
from praxis_agent.mcp import MCPConnections
from praxis_agent.mcp_results import decode_tool_output, mcp_tool_result
from praxis_agent.model import FunctionCall, ModelResponse, _response_input
from praxis_agent.sessions import SQLiteSessionStore
from praxis_agent.terminal.views import AgentView, ToolBlock
from praxis_agent.tools import Tool, ToolContext, create_default_registry
from praxis_agent.planning import RunState


PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='


def mixed_reply(*, is_error=False):
    return CallToolResult(content=[
        TextContent(type="text", text="Order B is unconfirmed."),
        ImageContent(type="image", data=PNG, mime_type="image/png"),
        AudioContent(type="audio", data="YXVkaW8=", mime_type="audio/wav"),
        ResourceLink(type="resource_link", name="orders", uri="file:///orders.csv", description="Order data"),
        EmbeddedResource(type="resource", resource=TextResourceContents(uri="file:///note.txt", text="Resource text")),
        EmbeddedResource(type="resource", resource=BlobResourceContents(uri="file:///plot.png", blob=PNG, mime_type="image/png")),
        EmbeddedResource(type="resource", resource=BlobResourceContents(uri="file:///report.pdf", blob="JVBERi0xLjQK", mime_type="application/pdf")),
        EmbeddedResource(type="resource", resource=BlobResourceContents(uri="file:///data.bin", blob="Ymlu", mime_type="application/octet-stream")),
    ], structured_content={"orders": [1, 2]}, is_error=is_error, meta={"private_metadata": "keep-in-code-only"})


@pytest.mark.parametrize("is_error", [False, True])
def test_mixed_results_preserve_data_and_separate_display(is_error):
    reply = mixed_reply(is_error=is_error)
    result = mcp_tool_result(reply)
    assert result.success is not is_error
    assert result.structured_output == reply.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert [b["type"] for b in result.model_content].count("input_image") == 2
    assert [b["type"] for b in result.model_content].count("input_file") == 1
    for text in ["Order B", "Resource text", "file:///orders.csv", "Structured content", "audio input", "application/octet-stream"]:
        assert text in result.output
    assert PNG not in result.output
    assert "private_metadata" not in json.dumps(result.model_content)
    output = _build_function_call_output(FunctionCall("c", "mcp_test_query", "{}"), result)
    assert decode_tool_output(output["output"]) == {"success": not is_error, "output": result.output}


@pytest.mark.parametrize("structured", [None, {}, [], 0, False, "", "value", [1, 2]])
def test_programmatic_success_does_not_render_or_unwrap(monkeypatch, structured):
    reply = mixed_reply()
    reply.structured_content = structured
    def must_not_render(*args):
        raise AssertionError("programmatic result must not render content blocks")
    monkeypatch.setattr("praxis_agent.mcp_results._content_blocks", must_not_render)
    result = mcp_tool_result(reply, programmatic=True)
    assert result.output == "" and result.model_content is None
    assert result.structured_output == reply.model_dump(mode="json", by_alias=True, exclude_none=True)


@pytest.mark.parametrize("structured", [{}, [], 0, False, ""])
def test_falsey_structured_content_and_no_deduplication(structured):
    reply = CallToolResult(content=[TextContent(type="text", text=json.dumps(structured))],
                           structured_content=structured)
    result = mcp_tool_result(reply)
    assert len(result.model_content) == 2
    assert result.model_content[0]["text"] == json.dumps(structured)
    assert result.model_content[1]["text"] == "Structured content:\n" + json.dumps(structured)


def test_empty_response_and_structured_only_error():
    assert "no content" in mcp_tool_result(CallToolResult(content=[])).output
    result = mcp_tool_result(CallToolResult(content=[], structured_content={"reason": "permission denied"}, is_error=True),
                             programmatic=True)
    assert not result.success and result.model_content is None
    assert "permission denied" in result.output


@pytest.mark.parametrize("stop_after_first", [False, True])
def test_immediate_activation_preserves_media_even_when_batch_stops(tmp_path, stop_after_first):
    connections = MCPConnections(tmp_path / "missing.json")
    record = {"id": "test/query", "server": "test", "original": "query",
              "alias": "mcp_test_query", "description": "Query", "schema_json": '{"type":"object"}',
              "approval": False}
    connections.index = SimpleNamespace(get=lambda identity: record)
    connections.status = {"test": {"state": "ready"}}
    connections.clients = {"test": object()}
    registry = create_default_registry()
    release = connections.register(registry)
    calls = []

    def check():
        if stop_after_first and calls:
            raise RuntimeError("cancelled")

    def invoke(name, arguments):
        calls.append(name)
        return mcp_tool_result(mixed_reply())

    context = ToolContext(tmp_path, RunState("activation"), invoke_tool=invoke, check_execution=check)
    try:
        result = registry.execute("activate_tools", {"tools": [
            {"tool_id": "test/query", "arguments": {}}, {"tool_id": "test/query"},
        ]}, context=context)
        assert result.success, result.output
        payload = json.loads(result.output)
        assert ("stopped" in payload) is stop_after_first
        assert len(payload["results"]) == (1 if stop_after_first else 2)
        assert [b["type"] for b in result.model_content].count("input_image") == 2
        assert [b["type"] for b in result.model_content].count("input_file") == 1
        assert PNG not in result.output
        assert calls == ["mcp_test_query"]
    finally:
        release()


@pytest.mark.parametrize("error", [TimeoutError, ConnectionError])
def test_transport_failure_reports_unknown_completion_without_retry(tmp_path, error):
    connections = MCPConnections(tmp_path / "missing.json")
    connections.servers = {"test": {}}
    client = AsyncMock()
    client.call_tool.side_effect = error("not exposed")
    connections.clients = {"test": client}
    async def invoke():
        connections.locks["test"] = asyncio.Lock()
        return await connections._call("test", "query", {}, programmatic=True)
    result = asyncio.run(invoke())
    assert not result.success
    assert "completion may be unknown" in result.output
    client.call_tool.assert_awaited_once()


def test_model_input_catalog_covers_every_configured_model():
    assert set(MODEL_INPUT_CATALOG) == set(MODEL_LIMIT_CATALOG)
    assert len(MODEL_INPUT_CATALOG) == 40


@pytest.mark.parametrize("name", ["auto(free)", "unknown-model", "qwen3-coder-next", "deepseek-v4-pro"])
def test_text_models_keep_text_and_replace_media_without_changing_success(name):
    item = _build_function_call_output(FunctionCall("c", "mcp_query", "{}"), mcp_tool_result(mixed_reply()))
    original = deepcopy(item)
    adapted = _response_input(item, ModelConfig(api_key="test", model=name).input_capabilities)
    assert all(block["type"] == "input_text" for block in adapted["output"])
    assert "Order B" in json.dumps(adapted)
    assert PNG not in json.dumps(adapted)
    assert json.loads(adapted["output"][0]["text"])["success"] is True
    assert item == original


@pytest.mark.parametrize("mime,data", [("image/svg+xml", "PHN2Zz4="), ("image/png", "not base64")])
def test_unsupported_or_corrupt_image_does_not_break_other_results(mime, data):
    reply = CallToolResult(content=[TextContent(type="text", text="operation completed"),
                                   ImageContent(type="image", mime_type=mime, data=data)])
    result = mcp_tool_result(reply)
    item = _build_function_call_output(FunctionCall("c", "mcp_query", "{}"), result)
    adapted = _response_input(item, ModelConfig(api_key="test", model="gpt-6-astra").input_capabilities)
    assert result.success
    assert all(b["type"] == "input_text" for b in adapted["output"])
    assert "operation completed" in json.dumps(adapted)
    assert "not provided" in json.dumps(adapted)


def test_media_survives_preview_and_is_not_base64_text_in_summary():
    reply = mixed_reply()
    reply.content.insert(0, TextContent(type="text", text="start" + "x" * 10000 + "end"))
    item = _build_function_call_output(FunctionCall("c", "mcp_query", "{}"), mcp_tool_result(reply))
    message = {"message_id": "m1", "payload": item}
    original = deepcopy(message)
    preview = preview_tool_outputs([message], 300)[0]
    media = lambda m: [b for b in m["payload"]["output"] if b["type"] != "input_text"]
    assert media(preview) == media(message)
    assert "characters omitted" in json.dumps(preview)
    summary = summary_input(summary_sources([message]), "summarize")
    assert PNG not in json.dumps(summary)
    assert "binary omitted" in json.dumps(summary)
    assert message == original
    usage = HeuristicTokenEstimator().estimate_breakdown(instructions="", input_items=[item], tools=[])
    assert usage["media"] > 0


@pytest.mark.parametrize("is_error", [False, True])
def test_agent_persists_media_and_restores_both_interfaces(tmp_path, is_error):
    result = mcp_tool_result(mixed_reply(is_error=is_error))
    registry = create_default_registry()
    registry.register(Tool("mcp_test_query", "Query", {"type": "object", "properties": {}}, lambda context: result))
    call = FunctionCall("c", "mcp_test_query", "{}")
    requests = []
    def respond(items, tools, **kwargs):
        requests.append(items)
        if len(requests) == 1:
            return ModelResponse(({"type": "function_call", "call_id": "c", "name": call.name, "arguments": "{}"},), "", (call,))
        return ModelResponse(({"role": "assistant", "content": "done"},), "done", ())
    store = SQLiteSessionStore(tmp_path / "history.sqlite3")
    session = store.create_session(tmp_path, "test")
    live = AgentView("main", "Main")
    agent = Agent(SimpleNamespace(create_response=respond), registry, tmp_path,
                  event_sink=SimpleNamespace(emit=live.consume), session_store=store,
                  session_id=session.id, session_model="test")
    try:
        assert agent.run("query") == "done"
        history = store.load_conversation_items(session.id)
        saved = next(i for i in history if i.get("type") == "function_call_output")
        assert any(b["type"] == "input_image" for b in saved["output"])
        restored = AgentView("main", "Main")
        restored.restore_history(history)
        assert next(b for b in live.committed_blocks if isinstance(b, ToolBlock)) == next(
            b for b in restored.committed_blocks if isinstance(b, ToolBlock))
        transcript = Transcript(history)
        tool = next(m for m in transcript.messages if m["role"] == "tool")
        assert tool["status"] == ("failed" if is_error else "completed")
        assert PNG not in tool["content"]
        event = display_event(ToolFinished("t", "c", "mcp_test_query", result))
        assert PNG not in json.dumps(event)
    finally:
        agent.close()
