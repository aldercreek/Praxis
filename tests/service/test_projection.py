import json

import pytest

from praxis_agent.events import (
    AssistantTextDelta,
    ChildAgentEvent,
    ModelRequestCompleted,
    ModelRequestStarted,
)
from praxis_agent.model import ModelResponse
from praxis_agent.service.events import ServiceEventSink as DesktopEventSink
from praxis_agent.service.transcript import Transcript


def answer(text):
    return ModelResponse(({"role": "assistant", "content": text},), text, ())


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("kind,status", [("cancelled", "cancelled"), ("timeout", "failed"), ("runtime", "failed")])
def test_python_status_matches_live_and_restored_history(structured, kind, status):
    payload = {"error_kind": kind, "error": "Execution ended", "state_lost": True}
    result = {"success": False, "output": json.dumps(payload)}
    if structured:
        result["structured_output"] = payload
    transcript = Transcript()
    transcript.consume("ToolCallPrepared", "turn", "main", {
        "call_id": "python", "tool_name": "run_code", "arguments": {"code": "1 / 0"}, "requires_approval": False,
    })
    transcript.consume("ToolFinished", "turn", "main", {"call_id": "python", "result": result})
    assert transcript.messages[0]["status"] == status


def test_collaboration_reports_are_grouped_deduplicated_and_keep_user_examples():
    report = dict(sender_id="reviewer", recipient_id="parent", content="Review complete.")
    envelope = "[Agent collaboration messages; source-labelled task data]\n" + json.dumps([report])
    transcript = Transcript()
    transcript.consume("TurnStarted", "turn", "main", {"task": envelope})
    transcript.consume("ChildAgentStatus", "turn", "reviewer", {"task_summary": "Review", "status": "running"})
    transcript.consume("ChildAgentMessage", "turn", "reviewer", {"content": report["content"]})
    transcript.consume("MessagesReceived", "turn", "main", {"source": "agent", "items": [{"role": "user", "content": envelope}]})
    transcript.consume("ChildAgentStatus", "turn", "reviewer", {"status": "idle"})
    assert len(transcript.messages) == 3
    assert transcript.messages[0]["content"] == envelope
    assert transcript.messages[0]["source"] == "user"
    assert transcript.messages[1]["content"] == "Review"
    assert transcript.messages[1]["status"] == "idle"
    assert transcript.messages[2]["agent_id"] == "reviewer"
    assert transcript.messages[2]["content"] == "Review complete."
    assert transcript.messages[2]["source"] == "agent"


@pytest.mark.parametrize("event", ["TurnStarted", "MessagesReceived"])
def test_worker_inputs_keep_each_sender_including_batches_and_repeated_content(event):
    reports = [
        dict(sender_id="parent", recipient_id="worker", content="Check this"),
        dict(sender_id="reviewer", recipient_id="worker", content="Check this"),
    ]
    envelope = "[Agent collaboration messages; source-labelled task data]\n" + json.dumps(reports)
    data = {"task": envelope} if event == "TurnStarted" else {"source": "agent", "items": [{"role": "user", "content": envelope}]}
    data["child_turn_id"] = "worker-run"
    transcript = Transcript()
    transcript.consume(event, "turn", "worker", data)
    assert len({message["id"] for message in transcript.messages}) == 2
    for message, report in zip(transcript.messages, reports, strict=True):
        assert {key: message[key] for key in report} == report
        assert message["role"] == "user"
        assert message["source"] == "agent"
        assert message["agent_id"] == "worker"
        assert message["turn_id"] == "turn"
        assert message["child_turn_id"] == "worker-run"


def test_event_adapter_omits_prompts_and_does_not_duplicate_child_events():
    received = []

    class Runtime:
        def publish(self, *event):
            received.append(event)

    sink = DesktopEventSink(Runtime())
    child = ChildAgentEvent(
        "parent-turn", "worker-1", AssistantTextDelta("child-turn", 1, "hello")
    )
    sink.emit_live(child)
    sink.emit_deferred(child)
    sink.emit(
        ModelRequestStarted(
            "turn", 1, ({"role": "developer", "content": "private prompt"},)
        )
    )
    sink.emit(ModelRequestCompleted("turn", 1, answer("public answer")))
    assert len(received) == 3
    assert received[0] == (
        "AssistantTextDelta",
        "parent-turn",
        "worker-1",
        {"request_number": 1, "text": "hello", "child_turn_id": "child-turn",
         "occurred_at": child.event.occurred_at, "timestamp": child.event.timestamp,
         "purpose": "agent", "compaction_id": None, "round_number": 0, "chunk_number": 0},
    )
    assert "private prompt" not in json.dumps(received)
    assert "response" not in received[-1][-1]


def test_child_response_numbers_can_restart_within_a_parent_turn():
    transcript = Transcript()
    for child_turn, text in [("child-1", "first result"), ("child-2", "second result")]:
        transcript.consume(
            "AssistantTextDelta",
            "parent-turn",
            "worker-1",
            {
                "request_number": 1,
                "text": text,
                "child_turn_id": child_turn,
            },
        )
    assert [message["content"] for message in transcript.messages] == [
        "first result",
        "second result",
    ]
    assert [message["id"] for message in transcript.messages] == [
        "assistant:parent-turn:worker-1:child-1:1",
        "assistant:parent-turn:worker-1:child-2:1",
    ]


def test_response_retry_preserves_partial_as_failed_without_affecting_other_agents():
    transcript = Transcript()
    transcript.consume("AssistantTextDelta", "turn", "main", {"request_number": 1, "text": "partial"})
    transcript.consume("AssistantTextDelta", "turn", "worker", {"request_number": 1, "text": "working"})
    transcript.consume("ModelRequestFailed", "turn", "main", {"request_number": 1, "error_type": "IncompleteResponseError", "message": "stream interrupted"})
    transcript.consume("ModelRequestStarted", "turn", "main", {"request_number": 2})
    assert [message["status"] for message in transcript.messages] == ["failed", "streaming"]
    transcript.consume("ModelRequestCompleted", "turn", "main", {"request_number": 2, "output_text": "done"})
    transcript.consume("TurnCompleted", "turn", "main", {})
    assert [(message["content"], message["status"]) for message in transcript.messages] == [
        ("partial", "failed"), ("working", "completed"), ("done", "completed"),
    ]


@pytest.mark.parametrize("failure", ["multi_agent_config", "mcp_constructor"])
def test_factory_initialization_failures_do_not_leak_model_clients(
    tmp_path, monkeypatch, failure
):
    import praxis_agent.factory as agent_module

    clients = []

    class Client:
        def __init__(self, *args, **kwargs):
            self.closed = False
            clients.append(self)

        def close(self):
            self.closed = True

    def invalid_mcp():
        raise ValueError("invalid MCP configuration")

    monkeypatch.setattr(agent_module, "OpenAIModelClient", Client)
    monkeypatch.setattr(agent_module, "discover_skills", lambda workspace: [])
    monkeypatch.setattr(agent_module, "MCPConnections", invalid_mcp)
    if failure == "multi_agent_config":
        from pathlib import Path
        import os
        Path(os.environ["PRAXIS_CONFIG"]).write_text("models: {}", encoding="utf-8")
    with pytest.raises(ValueError):
        agent_module.create_default_agent(
            tmp_path, session_database_path=tmp_path / "factory.sqlite3"
        )
    assert len(clients) == (0 if failure == "multi_agent_config" else 2)
    assert all(client.closed for client in clients)


def test_child_cancellation_finishes_only_its_execution_and_root_cancels_remaining_tools():
    transcript = Transcript()
    for agent, child_turn, call in [
        ("worker-a", "child-a", "a"),
        ("worker-a", "child-b", "b"),
        ("worker-b", "child-c", "c"),
    ]:
        transcript.consume(
            "ToolCallPrepared",
            "parent",
            agent,
            {
                "call_id": call,
                "tool_name": "run_command",
                "arguments": {},
                "requires_approval": True,
                "child_turn_id": child_turn,
            },
        )
    transcript.consume(
        "ApprovalResolved",
        "parent",
        "worker-a",
        {
            "call_id": "a",
            "tool_name": "run_command",
            "approved": False,
            "child_turn_id": "child-a",
        },
    )
    transcript.consume(
        "TurnCancelled", "parent", "worker-a", {"child_turn_id": "child-a"}
    )
    assert [message["status"] for message in transcript.messages] == [
        "cancelled",
        "approval",
        "approval",
    ]
    transcript.consume("TurnCancelled", "parent", "main", {})
    assert [message["status"] for message in transcript.messages] == ["cancelled"] * 3


def test_denied_finished_tool_remains_denied_after_root_cancellation():
    transcript = Transcript()
    transcript.consume(
        "ToolCallPrepared",
        "turn",
        "main",
        {
            "call_id": "denied",
            "tool_name": "run_command",
            "arguments": {},
            "requires_approval": True,
        },
    )
    transcript.consume(
        "ApprovalResolved",
        "turn",
        "main",
        {
            "call_id": "denied",
            "tool_name": "run_command",
            "approved": False,
        },
    )
    transcript.consume(
        "ToolFinished",
        "turn",
        "main",
        {
            "call_id": "denied",
            "tool_name": "run_command",
            "result": {
                "success": False,
                "output": "tool execution was declined by the user; do not retry",
            },
        },
    )
    transcript.consume("TurnCancelled", "turn", "main", {})
    transcript.consume("TurnSettled", "turn", "main", {"status": "cancelled"})
    assert transcript.messages[0]["status"] == "denied"
    assert transcript.messages[0]["tool_finished"] is True


@pytest.mark.parametrize(
    "output,expected",
    [
        ("tool execution was declined by the user; do not retry", "denied"),
        ({"status": "cancelled", "stdout": "partial"}, "cancelled"),
        ("ExecutionStopped: command cancelled", "cancelled"),
    ],
)
def test_restored_tool_keeps_denied_or_cancelled_result_status(output, expected):
    transcript = Transcript()
    transcript.consume("ToolCallPrepared", "turn", "main", {"call_id": "c", "tool_name": "run_command", "arguments": {}, "requires_approval": False})
    transcript.consume("ToolFinished", "turn", "main", {"call_id": "c", "result": {"success": False, "output": output}})
    assert transcript.messages[0]["status"] == expected
    assert transcript.messages[0]["tool_finished"] is True


def test_extension_catalog_uses_loaded_registrations_and_keeps_disabled_entries(service):
    from dataclasses import replace
    from praxis_agent.config import ToolExtensionConfig
    from praxis_agent.tool_extensions import ToolExtensions
    from praxis_agent.hooks import HookManager
    from praxis_agent.terminal.slash_commands import extensions_result
    h = service()
    path = h.path / "extensions.py"
    path.write_text("from praxis_agent.sdk import tool, hook\n"
                    "@tool(description='Lookup records')\ndef lookup(key: str): return key\n"
                    "@hook('before_tool_call', tools=['custom__*'])\ndef policy(event): pass\n", encoding="utf-8")
    configs = (ToolExtensionConfig("custom", path), ToolExtensionConfig("disabled", h.path / "absent.py", False))
    h.manager.config = replace(h.manager.config, tool_extensions=configs, hook_extensions=configs)
    h.manager.tool_extensions = ToolExtensions(configs)
    h.manager.hooks = HookManager(configs)
    path.unlink()  # Listing must not re-import the file or infer its registrations.
    data = h.client().request("GET", "/api/extensions")
    assert data["tools"][0]["name"] == "custom__lookup"
    assert data["tools"][0]["source"] == "python:custom"
    assert data["hooks"] == [dict(name="custom.policy", event="before_tool_call", tools=["custom__*"], order=1)]
    assert sum(not item["enabled"] for item in data["extensions"]) == 2
    assert "Lookup records" in extensions_result(data, "tools").plain
    assert "before_tool_call" in extensions_result(data, "hooks").plain
