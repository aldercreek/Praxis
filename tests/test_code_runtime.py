"""Exercise the shipped Monty worker through Praxis's real tool executor."""
import json
import textwrap

import pytest

from praxis_agent.model import FunctionCall
from praxis_agent.planning import RunState
from praxis_agent.tool_execution import ToolCallExecutor
from praxis_agent.tools import Tool, ToolContext, ToolResult, create_default_registry
from praxis_agent.tools.code import CodeConfig, CodeRuntime


@pytest.fixture
def code_tools(tmp_path):
    runtime = CodeRuntime(CodeConfig(max_processes=1))
    registry = create_default_registry()
    registry.register(runtime.tool())
    executor = ToolCallExecutor(
        registry, ToolContext(tmp_path, RunState("Monty compatibility")),
        publish_event=lambda event: None,
    )

    def run(code, session_id="test", *, success=True):
        result = executor.execute("test-turn", FunctionCall(
            "code-call", "run_code", json.dumps({
                "session_id": session_id, "code": textwrap.dedent(code),
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
    assert result["state_saved"] is True

    saved = run('''
        import json
        call_tool("write_file", {
            "relative_path": "out/report.json",
            "content": json.dumps(report, ensure_ascii=False, indent=2),
        })
    ''')
    assert saved["session_created"] is False
    assert [call["tool"] for call in saved["calls"]] == ["write_file"]
    assert json.loads((tmp_path / "out/report.json").read_text(encoding="utf-8")) == [
        {"category": "a,b", "count": 5},
    ]


def test_restored_functions_use_saved_data_and_sessions_are_separate(code_tools):
    run, _, _ = code_tools
    run("values = [1, 2]\ndef total():\n    return sum(values)\ntotal()")

    resumed = run("values.append(3)\nprint(total())\ntotal()")
    assert resumed["result"] == 6
    assert resumed["stdout"] == "6\n"
    assert resumed["session_created"] is False
    assert run("'values' in locals()", session_id="separate")["result"] is False
    assert run("total()")["result"] == 6


def test_type_check_rejects_script_before_any_tool_runs(code_tools, tmp_path):
    run, _, _ = code_tools
    rejected = run('''
        call_tool("write_file", {"relative_path": "must-not-exist.txt", "content": "bad"})
        count: int = "invalid"
    ''', success=False)

    assert rejected["error_kind"] == "type_check"
    assert rejected["phase"] == "preflight"
    assert rejected["state_saved"] is False
    assert rejected["calls"] == []
    assert not (tmp_path / "must-not-exist.txt").exists()


def test_runtime_failure_preserves_snapshot_but_not_external_effects(code_tools, tmp_path):
    run, _, _ = code_tools
    run("values = [1]")
    failed = run('''
        values.append(2)
        call_tool("write_file", {"relative_path": "completed.txt", "content": "done"})
        raise ValueError("after write")
    ''', success=False)

    assert failed["error_kind"] == "runtime"
    assert failed["state_saved"] is False
    assert failed["calls"] == [{"tool": "write_file", "success": True}]
    assert (tmp_path / "completed.txt").read_text(encoding="utf-8") == "done"
    assert run("values")["result"] == [1]


def test_mcp_envelope_retry_and_partial_progress_survive_restore(code_tools):
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

    assert partial["state_saved"] is True
    assert partial["result"] == {"done": ["ready"], "failed": ["unavailable"]}
    assert attempts == {"ready": 2, "unavailable": 1}
    assert run("results['ready']['value']")["result"] == 42


@pytest.mark.parametrize("name,arguments", [
    ("run_command", {"argv": ["must-not-run"]}),
    ("run_code", {"session_id": "nested", "code": "1"}),
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
