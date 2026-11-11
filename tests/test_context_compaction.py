import json
from copy import deepcopy
from types import SimpleNamespace

import httpx2
import pytest
from mcp.types import CallToolResult, TextContent
from openai import AsyncOpenAI

from praxis_agent.agent import Agent, _build_function_call_output
from support.models import model_config
from praxis_agent.context.capacity import ContextPolicy, HeuristicTokenEstimator
from praxis_agent.context.compaction import trim_tool_messages
from praxis_agent.context.manager import ContextManager
from praxis_agent.context.model_metadata import ModelLimits
from praxis_agent.context.model_context import build_context_status_context
from praxis_agent.context.models import ConversationHistory
from praxis_agent.context.operations import ContextOperations
from praxis_agent.context.retrieval import read_history
from praxis_agent.context.summarizer import ModelContextSummarizer
from praxis_agent.context.tools import read_history as read_history_tool
from praxis_agent.model import FunctionCall, ModelResponse, OpenAIModelClient
from praxis_agent.model_errors import ModelClientError, PromptTooLongError, model_request_error
from praxis_agent.mcp_results import mcp_tool_result
from praxis_agent.sessions import SQLiteSessionStore
from praxis_agent.tools.registry import Tool, ToolContext, ToolRegistry, ToolResult


def tool_pair(body, *, success=True, name="inspect"):
    return [
        {"type": "function_call", "call_id": "c", "name": name, "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c",
         "output": json.dumps({"success": success, "output": body})},
    ]


@pytest.mark.parametrize("body,success,status,error", [
    ("done", True, "success", None),
    ({"exit_code": 0, "stdout": "done"}, True, "success", None),
    ({"exit_code": 2, "stderr": "permission denied"}, True, "success", None),
    ({"isError": True, "message": "Historical error"}, True, "success", None),
    ("invalid parameters", False, "failed", "invalid parameters"),
    ({"status": "running", "session_id": "job-1", "pid": 42}, True, "success", None),
    ({"status": "failed", "message": "Historical job failed"}, True, "success", None),
    ({"status": "cancelled", "message": "user stopped"}, True, "success", None),
    ({"status": "interrupted", "error": "disconnected"}, False, "failed", "disconnected"),
    ({"status": "exited", "exit_code": 0}, False, "failed", "exited"),
    ({"status": "failed", "exit_code": 2}, None, "unknown", None),
])
def test_trimming_uses_outer_success_for_record_contents(body, success, status, error):
    history = tool_pair(json.dumps(body) if isinstance(body, dict) else body,
                        success=success, name="read_record")
    original = deepcopy(history)
    trimmed = trim_tool_messages(ContextOperations().project(history))
    output = json.loads(trimmed[-1]["payload"]["output"])
    assert output["status"] == status
    assert output["source_message_id"] == "m2"
    assert output.get("success") is success
    assert not {"exit_code", "session_id", "pid", "isError", "stdout", "stderr"} & output.keys()
    if error:
        assert error in output["error"]
    else:
        assert "error" not in output
    assert history == original
    assert trim_tool_messages(trimmed) == trimmed


@pytest.mark.parametrize("name", ["run_command", "wait_command"])
@pytest.mark.parametrize("status,exit_code,success,error", [
    ("exited", 0, True, None),
    ("exited", 1, True, "The meaning of a nonzero exit code depends on the program."),
    ("exited", 2, True, "TLS_CERT_EXPIRED"),
    ("running", None, True, None),
    ("spawn_failed", None, False, "Unable to start command"),
    ("cancelled", None, False, "Command was stopped"),
    ("interrupted", None, False, "disconnected"),
    ("failed", None, False, "Command cleanup failed"),
])
def test_trimming_keeps_command_outcome_without_rewriting_success(name, status, exit_code, success, error):
    body = {"status": status, "exit_code": exit_code, "session_id": "job-1", "pid": 42,
            "stdout": "omitted command output", "stderr": error or "", "error": error}
    messages = ContextOperations().project(tool_pair(body, success=success, name=name))
    output = json.loads(trim_tool_messages(messages)[1]["payload"]["output"])
    assert output["success"] is success
    assert {key: output[key] for key in ("status", "exit_code", "session_id", "pid")} == {
        key: body[key] for key in ("status", "exit_code", "session_id", "pid")}
    assert "stdout" not in output and "stderr" not in output
    if error:
        assert error in output["error"]
    else:
        assert "error" not in output


def test_trimming_uses_call_ids_and_does_not_guess_missing_tool_names():
    body = {"status": "exited", "exit_code": 2, "stderr": "TLS_CERT_EXPIRED", "session_id": "job-1"}
    calls = [FunctionCall("read", "read_record", "{}"), FunctionCall("command", "run_command", "{}")]
    items = [{"type": "function_call", "call_id": call.call_id, "name": call.name, "arguments": "{}"}
             for call in calls]
    result = ToolResult(True, json.dumps(body), structured_output=body)
    items.extend(_build_function_call_output(call, result) for call in reversed(calls))
    items.append({"type": "function_call_output", "call_id": "missing",
                  "output": json.dumps({"success": True, "output": body})})
    trimmed = trim_tool_messages(ContextOperations().project(items))
    outputs = {m["payload"]["call_id"]: json.loads(m["payload"]["output"])
               for m in trimmed if m["payload"].get("type") == "function_call_output"}
    assert outputs["command"]["exit_code"] == 2
    assert outputs["command"]["status"] == "exited"
    assert outputs["command"]["success"] is True
    for call_id in ("read", "missing"):
        assert outputs[call_id]["success"] is True
        assert outputs[call_id]["status"] == "success"
        assert not {"exit_code", "session_id", "error"} & outputs[call_id].keys()


@pytest.mark.parametrize("is_error", [False, True])
def test_trimming_uses_mcp_protocol_errors_not_error_fields_in_content(is_error):
    body = {"isError": True, "status": "failed", "exit_code": 2, "message": "Remote failure record"}
    reply = CallToolResult(content=[TextContent(type="text", text=json.dumps(body))], is_error=is_error)
    call = FunctionCall("mcp", "mcp_query", "{}")
    items = [{"type": "function_call", "call_id": call.call_id, "name": call.name, "arguments": "{}"},
             _build_function_call_output(call, mcp_tool_result(reply))]
    output = json.loads(trim_tool_messages(ContextOperations().project(items))[1]["payload"]["output"])
    assert output["success"] is not is_error
    assert output["status"] == ("failed" if is_error else "success")
    if is_error:
        assert "Remote failure record" in output["error"]
    else:
        assert "error" not in output


def test_ptl_summary_receives_successful_read_outcome_without_reclassifying_the_record():
    requests = []

    def summarize(items, **options):
        requests.append(json.loads(items[0]["content"]))
        return ModelResponse((), "Record read successfully; use the original reference for its contents.", ())

    limits = ModelLimits(32000)
    manager = ContextManager(model_limits=limits, summarizer=ModelContextSummarizer(
        SimpleNamespace(create_response=summarize), limits))
    body = {"exit_code": 2, "stderr": "TLS_CERT_EXPIRED", "status": "failed"}
    history = ConversationHistory(tool_pair(json.dumps(body), name="read_record") + [
        {"role": "user", "content": "Continue the audit"},
    ])
    original = deepcopy(history.snapshot())
    manager.build_view(history)
    manager.recover_from_ptl(PromptTooLongError("input too long"))
    source = next(m for m in requests[0]["selected_messages"]
                  if m["payload"].get("type") == "function_call_output")
    assert source["message_id"] == "m2"
    assert json.loads(source["payload"]["output"]) == {
        "success": True, "output": json.dumps(body),
    }
    assert history.snapshot() == original


def test_trimming_bounds_errors_preserves_unknown_and_allows_argument_retrieval():
    history = tool_pair("HEAD " + "noise " * 3000 + " TAIL", success=False)
    history[0]["arguments"] = json.dumps({"content": "exact original\n" * 2000, "path": "sample.py"})
    messages = ContextOperations().project(history)
    trimmed = trim_tool_messages(messages)
    output = json.loads(trimmed[1]["payload"]["output"])
    assert len(output["error"]) < 1650
    assert "HEAD" in output["error"] and "TAIL" in output["error"]
    arguments = json.loads(trimmed[0]["payload"]["arguments"])
    assert arguments["path"] == "sample.py"
    assert "read_history m1" in arguments["content"]
    assert read_history(ConversationHistory(history), [], "m1", character_limit=None)["text"] == history[0]["arguments"]
    messages[1]["payload"]["output"] = "plain unclassified output"
    assert json.loads(trim_tool_messages(messages)[1]["payload"]["output"])["status"] == "unknown"


def test_omitted_media_is_retrievable_without_base64_in_text(tmp_path):
    items = tool_pair("image result")
    media = {"type": "input_image", "image_url": "data:image/png;base64,PRIVATE_BINARY"}
    items[-1]["output"] = [{"type": "input_text", "text": '{"success": true}'}, media]
    history = ConversationHistory(items)
    manager = ContextManager()
    manager.build_view(history)
    trimmed = trim_tool_messages(manager._current_messages)
    assert "PRIVATE_BINARY" not in json.dumps(trimmed)
    result = read_history_tool(ToolContext(tmp_path, None, history=history, context_manager=manager),
                               "m2", include_media=True)
    assert "PRIVATE_BINARY" not in result.output
    assert media in result.model_content
    assert items[-1]["output"][-1] == media


@pytest.mark.parametrize("tokens,pressure,reminder", [
    (6499, "normal", False), (6500, "soft", True),
    (7999, "soft", True), (8000, "hard", True),
    (8999, "hard", True), (9000, "emergency", True),
    (9499, "emergency", True), (9500, "emergency", True),
])
def test_pressure_boundaries(tokens, pressure, reminder):
    policy = ContextPolicy(safety_ratio=1, output_reserve_tokens=0)
    status = policy.assess(ModelLimits(10000), tokens)
    assert status.pressure == pressure
    text = build_context_status_context(status, policy)["content"]
    assert ("2. Reminder" in text) is reminder
    if reminder:
        assert "95%" in text


def test_reminder_is_replaced_and_removed_after_context_shrinks():
    history = ConversationHistory([{"role": "assistant", "content": "x" * 20000},
                                   {"role": "user", "content": "current requirement"}])
    manager = ContextManager(model_limits=ModelLimits(12000), policy=ContextPolicy(output_reserve_tokens=1000, summary_output_tokens=1000))
    view = manager.build_view(history)
    # Add enough history to cross soft but stay below automatic compaction.
    history.append({"role": "assistant", "content": "x" * 1000})
    view = manager.build_view(history)
    assert "2. Reminder" in view.items[-1]["content"]
    manager.change_messages("compress", ["m1", "m3"], summary="verified facts")
    reduced = manager.build_view(history)
    assert "2. Reminder" not in reduced.items[-1]["content"]
    assert json.dumps(reduced.items).count("<context_status>") == 1
    assert "context_status" not in json.dumps(history.snapshot())


@pytest.mark.parametrize("estimated,history_tokens,fixed_tokens,action", [
    (9499, 1000, 500, None),
    (9500, 2001, 500, "trim"),  # Between 20% of effective input and 20% of the window.
    (9500, 2200, 3000, "trim"),
    (9500, 2201, 500, "compress"),
    (9500, 1000, 8499, "trim"),
    (9500, 1000, 8500, "compress"),
    (9500, 1000, 8501, "compress"),
])
def test_exact_automatic_and_engineering_boundaries(estimated, history_tokens, fixed_tokens, action):
    runtime = {"role": "developer", "content": "FIXED_RUNTIME"}
    tools = ({"type": "function", "name": "inspect", "description": "FIXED_TOOLS"},)

    class Estimator:
        def estimate_request(self, **kwargs):
            content = json.dumps(kwargs["input_items"])
            if not kwargs["instructions"]:
                assert not kwargs["tools"]
                assert "FIXED_" not in content
                return history_tokens
            assert kwargs["instructions"] == "FIXED_INSTRUCTIONS"
            assert kwargs["tools"] == tools and runtime in kwargs["input_items"]
            for fixed in ("FIXED_USER", "FIXED_SYSTEM", "FIXED_DEVELOPER"):
                assert fixed in content
            if "Conversation summary" in content:
                return fixed_tokens + 100
            return fixed_tokens + history_tokens if "context_omitted" in content else estimated

    manager = ContextManager(
        model_limits=ModelLimits(11000), policy=ContextPolicy(safety_ratio=1, output_reserve_tokens=1000, summary_output_tokens=1000),
        instructions="FIXED_INSTRUCTIONS",
        token_estimator=Estimator(), summarizer=SimpleNamespace(summarize=lambda *a, **k: "verified"),
    )
    history = ConversationHistory([
        {"role": "system", "content": "FIXED_SYSTEM"},
        {"role": "developer", "content": "FIXED_DEVELOPER"},
        *tool_pair("result"), {"role": "user", "content": "FIXED_USER"},
    ])
    manager.build_view(history, transient_items=(runtime,), tool_definitions=tools)
    assert [op.action for op in manager.operations.records] == ([action] if action else [])


def test_trim_and_full_summary_round_trip_through_sqlite(tmp_path):
    path = tmp_path / "state.sqlite3"
    store = SQLiteSessionStore(path)
    session = store.create_session(tmp_path, "test")
    manager = ContextManager(model_limits=ModelLimits(10000), policy=ContextPolicy(output_reserve_tokens=1000, summary_output_tokens=1000))
    manager.operations.persist = lambda op: store.append_context_operation(session.id, op)
    items = tool_pair("log " * 10000)
    # Large arguments also cross the threshold after normal tool previews.
    items[0]["arguments"] = json.dumps({"content": "sample " * 6000})
    history = ConversationHistory(items)
    original = deepcopy(history.snapshot())
    manager.build_view(history)
    assert manager.operations.records[0].action == "trim"
    history.extend([{"role": "assistant", "content": "evidence " * 4000}, {"role": "user", "content": "continue"}])
    manager._summarizer = SimpleNamespace(summarize=lambda *a, **k: "verified evidence")
    expected = manager.build_view(history)
    assert [op.action for op in manager.operations.records] == ["trim", "compress"]
    restored = ContextManager(model_limits=ModelLimits(10000), policy=manager._policy)
    restored.operations.replay(SQLiteSessionStore(path).load_context_operations(session.id))
    assert restored.build_view(history).items == expected.items
    assert history.snapshot()[:2] == original


@pytest.mark.parametrize("channel", ["http", "stream"])
@pytest.mark.parametrize("code,error_type", [("context_length_exceeded", PromptTooLongError), ("invalid_api_key", ModelClientError)])
def test_provider_input_errors_keep_identity(monkeypatch, channel, code, error_type):
    def respond(request):
        error = {"code": code, "message": "provider rejected request", "type": "invalid_request_error",
                 "input_tokens": 12345, "max_input_tokens": 12000}
        if channel == "http":
            return httpx2.Response(400, json={"error": error})
        event = {"type": "error", "sequence_number": 0, "code": code, "message": error["message"], "param": None,
                 "input_tokens": 12345, "max_input_tokens": 12000}
        return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                               content="data: " + json.dumps(event) + "\n\n")

    monkeypatch.setattr("praxis_agent.model.AsyncOpenAI", lambda **options: AsyncOpenAI(
        **options, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond))))
    client = OpenAIModelClient(model_config(api_key="test", model="test", base_url="https://example.test/v1"))
    try:
        with pytest.raises(error_type) as error:
            client.create_response([], [])
        assert type(error.value) is error_type
        assert error.value.code == code
        if error_type is PromptTooLongError:
            assert error.value.required_reduction == 346
    finally:
        client.close()


@pytest.mark.parametrize("message,is_ptl", [("Prompt is too long", True),
    ("maximum context length is 1000", True), ("input tokens exceed the limit", True),
    ("connection timeout", False), ("max_output_tokens must be smaller", False), ("HTTP 429", False)])
def test_ptl_text_does_not_classify_network_or_output_limits(message, is_ptl):
    assert isinstance(model_request_error(message), PromptTooLongError) is is_ptl


@pytest.mark.parametrize("large_previous_summary", [False, True])
def test_summary_ptl_reduces_chunks_without_skipping_material(large_previous_summary):
    calls = []
    accepted = []
    def respond(items, **options):
        data = json.loads(items[0]["content"])
        calls.append((data, options))
        if len(calls) == 2:
            raise PromptTooLongError("input too long")
        accepted.append(data)
        text = "x" * 6500 if large_previous_summary and len(calls) == 1 else "checkpoint"
        return ModelResponse((), text, ())

    summarizer = ModelContextSummarizer(SimpleNamespace(create_response=respond), ModelLimits(32000), input_limit_tokens=4000)
    material = {"message_id": "m1", "role": "assistant", "content": "evidence " * 3000}
    assert summarizer.summarize([material], "preserve all facts") == "checkpoint"
    assert {options["max_output_tokens"] for _, options in calls} == {4000}
    # Every PTL restarts the whole task, even when the previous summary would fit.
    assert accepted[1]["previous_summary"] == ""
    accepted = accepted[1:]
    text = ""
    for data in accepted:
        for fragment in data["selected_messages"]:
            assert fragment["serialized_offset"] == len(text)
            text += fragment["text"]
    assert json.loads(text) == material


def test_main_ptl_shortens_summary_without_executing_tools_twice(tmp_path):
    tool_runs = []
    summary_limits = []
    summary_targets = []
    inputs = []
    registry = ToolRegistry()
    registry.register(Tool("inspect", "read only", {"type": "object", "properties": {}},
                           lambda context: tool_runs.append(1) or "done"))
    def summarize(messages, instructions, **options):
        summary_limits.append(options["max_output_tokens"])
        summary_targets.append(options["target_tokens"])
        return "verified fact " * (100 if len(summary_limits) == 1 else 10)
    def respond(items, tools, **options):
        inputs.append(deepcopy(items))
        if len(inputs) == 1:
            call = FunctionCall("c", "inspect", "{}")
            return ModelResponse((tool_pair("")[0],), "", (call,))
        if len(inputs) == 2:
            raise PromptTooLongError("input too long")
        return ModelResponse((), "done", ())
    manager = ContextManager(model_limits=ModelLimits(12000), policy=ContextPolicy(output_reserve_tokens=1000, summary_output_tokens=1000),
                             summarizer=SimpleNamespace(summarize=summarize))
    agent = Agent(SimpleNamespace(create_response=respond), registry, tmp_path, context_manager=manager, bypass_approvals=True)
    agent.restore_history([{"role": "assistant", "content": "old history " * 4000}])
    try:
        assert agent.run("read only; do not change files") == "done"
        assert tool_runs == [1]
        assert len(inputs) == 3
        assert summary_limits == [1000, 1000]
        assert summary_targets == [2400, 2400]
        assert "read only; do not change files" in json.dumps(inputs[-1])
    finally:
        agent.close()


def test_summary_ptl_does_not_consume_main_recovery_rounds(tmp_path):
    summary_calls = []
    main_calls = []
    def summarize(items, **kwargs):
        summary_calls.append(items)
        if len(summary_calls) == 1:
            raise PromptTooLongError("input too long")
        return ModelResponse((), "verified checkpoint", ())
    def main(*args, **kwargs):
        main_calls.append(1)
        raise PromptTooLongError("input too long")
    manager = ContextManager(model_limits=ModelLimits(32000), policy=ContextPolicy(output_reserve_tokens=1000, summary_output_tokens=1000),
        summarizer=ModelContextSummarizer(SimpleNamespace(create_response=summarize), ModelLimits(32000)))
    # A summary PTL uses only its own retries; the main request still gets two recoveries.
    agent = Agent(SimpleNamespace(create_response=main), ToolRegistry(), tmp_path, context_manager=manager)
    agent.restore_history([{"role": "assistant", "content": "old facts " * 600}])
    try:
        with pytest.raises(PromptTooLongError):
            agent.run("continue")
        assert len(main_calls) == 3
        assert len(summary_calls) > 2
    finally:
        agent.close()
