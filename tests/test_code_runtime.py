"""Exercise the shipped Monty worker through Praxis's real tool executor."""
import asyncio
import json
import textwrap
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent

from praxis_agent.mcp import MCPConnections
from praxis_agent.model import FunctionCall
from praxis_agent.planning import RunState
from praxis_agent.tool_execution import ToolCallExecutor
from praxis_agent.tools import Tool, ToolContext, ToolResult, create_default_registry
from praxis_agent.tools.code import CodeConfig, CodeRuntime


@pytest.fixture
def code_tools(tmp_path):
    runtime = CodeRuntime(output_dir=tmp_path / "code-logs")
    registry = create_default_registry()
    registry.register(runtime.tool())
    executor = ToolCallExecutor(
        registry, ToolContext(tmp_path, RunState("Monty runtime")),
        publish_event=lambda event: None,
    )

    def run(code, *, success=True):
        result = executor.execute("test-turn", FunctionCall(
            "code-call", "run_code", json.dumps({
                "code": textwrap.dedent(code),
            }),
        ))
        assert result.success is success, result.output
        return json.loads(result.output)

    try:
        yield run, registry, executor
    finally:
        runtime.close()


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_csv_toml_and_json_through_file_tools(code_tools, tmp_path, newline):
    run, _, _ = code_tools
    csv_text = ('category,count,note\n"a,b",2,"first\nsecond"\n'
                'other,1,"a ""quote"""\n"a,b",3,last\n')
    (tmp_path / "input.csv").write_bytes(csv_text.replace("\n", newline).encode("utf-8"))
    (tmp_path / "settings.toml").write_text(
        "[report]\nmin_count = 2\n", encoding="utf-8",
    )
    result = run('''
        import csv
        import tomllib
        source = call_tool("read_file", {"relative_path": "input.csv"})
        settings = tomllib.loads(
            call_tool("read_file", {"relative_path": "settings.toml"})["text"]
        )
        rows = list(csv.DictReader(source["text"].splitlines(keepends=True)))
        totals = {}
        for row in rows:
            totals[row["category"]] = totals.get(row["category"], 0) + int(row["count"])
        report = [
            {"category": key, "count": value} for key, value in totals.items()
            if value >= settings["report"]["min_count"]
        ]
        report = sorted(report, key=lambda item: item["count"], reverse=True)
        {"report": report, "notes": [row["note"] for row in rows]}
    ''')
    assert result["result"] == {
        "report": [{"category": "a,b", "count": 5}],
        "notes": [f"first{newline}second", 'a "quote"', "last"],
    }
    assert result["worker_created"] is True

    saved = run('''
        import json
        call_tool("write_file", {
            "relative_path": "out/report.json",
            "content": json.dumps(report, ensure_ascii=False, indent=2),
        })
    ''')
    assert saved["worker_created"] is False
    assert [call["tool"] for call in saved["calls"]] == ["write_file"]
    assert json.loads((tmp_path / "out/report.json").read_text(encoding="utf-8")) == [
        {"category": "a,b", "count": 5},
    ]


def test_functions_and_mutable_data_persist_in_the_same_worker(code_tools):
    run, _, _ = code_tools
    run("values = [1, 2]\ndef total():\n    return sum(values)\ntotal()")

    resumed = run("values.append(3)\nprint(total())\ntotal()")
    assert resumed["result"] == 6
    assert resumed["stdout"] == "6\n"
    assert resumed["worker_created"] is False
    assert run("total()")["result"] == 6


def test_type_check_rejects_script_before_any_tool_runs(code_tools, tmp_path):
    run, _, _ = code_tools
    rejected = run('''
        call_tool("write_file", {"relative_path": "must-not-exist.txt", "content": "bad"})
        count: int = "invalid"
    ''', success=False)

    assert rejected["error_kind"] == "type_check"
    assert rejected["phase"] == "preflight"
    assert rejected["state_lost"] is False
    assert rejected["calls"] == []
    assert not (tmp_path / "must-not-exist.txt").exists()


def test_runtime_failure_keeps_partial_mutations_and_external_effects(code_tools, tmp_path):
    run, _, _ = code_tools
    run("values = [1]")
    failed = run('''
        values.append(2)
        call_tool("write_file", {"relative_path": "completed.txt", "content": "done"})
        raise ValueError("after write")
    ''', success=False)

    assert failed["error_kind"] == "runtime"
    assert failed["state_lost"] is False
    assert failed["calls"] == [{"tool": "write_file", "success": True}]
    assert (tmp_path / "completed.txt").read_text(encoding="utf-8") == "done"
    assert run("values")["result"] == [1, 2]


@pytest.mark.parametrize("expression", ['{(1, 2): "value"}', '{b"key": "value"}', '{1, 2}', "b'value'"])
@pytest.mark.parametrize("existing_worker", [False, True])
def test_result_encoding_failure_keeps_executed_changes(code_tools, tmp_path, expression, existing_worker):
    run, _, _ = code_tools
    if existing_worker:
        run("saved: int = 1")
    failed = run('saved = 2\n'
                 'print("before encoding")\n'
                 'call_tool("write_file", {"relative_path": "completed.txt", "content": "done"})\n'
                 + expression, success=False)

    assert failed["error_kind"] == "serialize"
    assert failed["phase"] == "serialize"
    assert failed["state_lost"] is False
    assert failed["worker_created"] is (not existing_worker)
    assert failed["stdout"] == "before encoding\n"
    assert failed["calls"] == [{"tool": "write_file", "success": True}]
    assert (tmp_path / "completed.txt").read_text(encoding="utf-8") == "done"
    assert run("saved")["result"] == 2


@pytest.mark.parametrize("structured", [None, {}, {"orders": [{"id": 42}]}])
@pytest.mark.parametrize("is_error", [False, True])
def test_mcp_structured_result_through_code_bridge(code_tools, tmp_path, structured, is_error):
    run, registry, _ = code_tools
    reply = CallToolResult(
        content=[TextContent(type="text", text="Original response text")],
        structured_content=structured, is_error=is_error,
    )
    connections = MCPConnections(tmp_path / "missing-mcp.json")
    connections.servers = {"test": {}}
    client = AsyncMock()
    client.call_tool.return_value = reply
    connections.clients = {"test": client}
    captured = []

    def query(context):
        async def invoke():
            connections.locks["test"] = asyncio.Lock()
            return await connections._call("test", "query", {})
        result = asyncio.run(invoke())
        captured.append(result)
        return result

    registry.register(Tool(
        "mcp_test_structured", "Read test data",
        {"type": "object", "properties": {}, "additionalProperties": False},
        query, programmatic=True,
    ))
    response = run('call_tool("mcp_test_structured", {})', success=not is_error)
    result = captured[0]
    assert result.success is (not is_error)
    assert result.structured_output == structured
    assert json.loads(result.output) == reply.model_dump(
        mode="json", by_alias=True, exclude_none=True,
    )
    client.call_tool.assert_awaited_once_with("query", {})
    if is_error:
        assert response["error_kind"] == "tool"
        assert "Original response text" in response["error"]
    else:
        assert response["result"] == (structured if structured is not None else result.output)


def test_mcp_envelope_retry_and_partial_progress_survive_between_calls(code_tools):
    run, registry, _ = code_tools
    attempts = {}

    def query(context, key):
        attempts[key] = attempts.get(key, 0) + 1
        if key == "unavailable":
            return ToolResult(False, "Permission denied")
        if attempts[key] == 1:
            return ToolResult(False, "MCP call failed (TimeoutError)")
        return ToolResult(True, json.dumps({
            "content": [{"type": "text", "text": "ordinary text, not JSON"}],
            "structuredContent": {"key": key, "value": 42},
        }))

    registry.register(Tool(
        "mcp_test_query", "Read test data", {
            "type": "object", "properties": {"key": {"type": "string"}},
            "required": ["key"], "additionalProperties": False,
        }, query, programmatic=True,
    ))
    run('''
        import json
        import time
        def read_mcp_with_retry(tool_name: str, arguments: dict):
            for attempt in range(3):
                try:
                    raw = call_tool(tool_name, arguments)
                except RuntimeError as error:
                    if "TimeoutError" not in str(error) or attempt == 2:
                        raise
                    time.sleep(0.01)
                else:
                    return json.loads(raw)
        results = {}
        errors = {}
    ''')
    partial = run('''
        for key in ["ready", "unavailable"]:
            try:
                envelope = read_mcp_with_retry("mcp_test_query", {"key": key})
                results[key] = envelope["structuredContent"]
            except RuntimeError as error:
                errors[key] = str(error)
        {"done": list(results), "failed": list(errors)}
    ''')

    assert partial["state_lost"] is False
    assert partial["result"] == {"done": ["ready"], "failed": ["unavailable"]}
    assert attempts == {"ready": 2, "unavailable": 1}
    assert run("results['ready']['value']")["result"] == 42


@pytest.mark.parametrize("name,arguments", [
    ("run_command", {"argv": ["must-not-run"]}),
    ("run_code", {"code": "1"}),
    ("update_plan", {}),
])
def test_bridge_rejects_tools_outside_whitelist(code_tools, name, arguments):
    run, _, _ = code_tools
    failed = run(f"call_tool({name!r}, {arguments!r})", success=False)

    assert failed["error_kind"] == "tool"
    assert "Tool unavailable for programmatic calls" in failed["error"]
    assert failed["calls"][0]["success"] is False


def test_bridge_keeps_tool_approval(code_tools):
    run, registry, executor = code_tools
    invoked = []
    approvals = []
    registry.register(Tool(
        "protected_read", "A read requiring approval", {"type": "object"},
        lambda context: invoked.append(True),
        requires_approval=True, programmatic=True,
    ))
    executor.approve_tool_call = lambda name, arguments: approvals.append(name) or False

    failed = run('call_tool("protected_read", {})', success=False)

    assert failed["error_kind"] == "tool"
    assert approvals == ["protected_read"]
    assert invoked == []


def test_more_than_5000_tool_calls_can_complete(code_tools):
    run, registry, _ = code_tools
    calls = 0

    def count(context):
        nonlocal calls
        calls += 1
        return ToolResult(True, "ok")

    registry.register(Tool(
        "count", "Count host calls", {"type": "object"}, count, programmatic=True,
    ))
    result = run('for i in range(5001):\n    call_tool("count", {})\n"done"')
    assert result["result"] == "done"
    assert result["state_lost"] is False
    assert calls == 5001


def test_code_output_streams_and_final_value_are_separate(code_tools, tmp_path):
    run, _, _ = code_tools
    result = run('''
        import sys
        print("normal")
        print("problem", file=sys.stderr)
        "result" * 10000
    ''')
    assert result["stdout"] == "normal\n"
    assert result["stderr"] == "problem\n"
    assert result["result"] == "result" * 10000
    assert result["output_files"] == {}
    assert not (tmp_path / "code-logs").exists()


def test_code_output_overflow_is_readable_through_file_tool(code_tools):
    run, _, _ = code_tools
    result = run('''
        import sys
        print("头" * 6000, end="")
        print("中" * 10000, end="")
        print("尾" * 24000, end="")
        print("warning" * 5000, end="", file=sys.stderr)
        42
    ''')
    assert result["result"] == 42
    assert result["stdout"] == "头" * 6000 + "\n... [10000 characters omitted] ...\n" + "尾" * 24000
    assert result["stderr"] == ("warning" * 5000)[:6000] + "\n... [5000 characters omitted] ...\n" + ("warning" * 5000)[-24000:]
    for name, text in {"stdout": "头" * 6000 + "中" * 10000 + "尾" * 24000,
                       "stderr": "warning" * 5000}.items():
        reference = result["output_files"][name]
        assert reference["complete"] is True
        assert reference["total_chars"] == len(text)
        original = run(f'call_tool("read_file", {{"relative_path": {reference["path"]!r}}})')
        assert original["result"]["text"] == text
    # Every invocation gets its own files; old history references remain valid.
    later = run('print("new" * 11000, end="")')
    assert later["output_files"]["stdout"]["path"] != result["output_files"]["stdout"]["path"]


def test_failed_code_keeps_complete_overflow_log(code_tools):
    from pathlib import Path

    run, _, _ = code_tools
    result = run('print("x" * 40000, end="")\nraise ValueError("failed after output")', success=False)
    assert result["state_lost"] is False
    assert result["error_kind"] == "runtime"
    reference = result["output_files"]["stdout"]
    assert reference["complete"] is True
    assert Path(reference["path"]).read_text(encoding="utf-8") == "x" * 40000


def test_output_file_failure_keeps_executed_changes(code_tools, tmp_path):
    run, _, _ = code_tools
    run("values = [1]")
    (tmp_path / "code-logs").write_text("not a directory", encoding="utf-8")
    result = run('values.append(2)\nprint("x" * 40000)', success=False)
    assert result["error_kind"] == "output"
    assert result["state_lost"] is False
    assert "stdout" in result["output_errors"]
    assert result["output_files"] == {}
    assert run("values")["result"] == [1, 2]
