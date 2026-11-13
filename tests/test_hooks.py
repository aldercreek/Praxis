import json
import sys
import textwrap
from dataclasses import replace

import pytest
import yaml

from praxis_agent.config import ToolExtensionConfig, load_config
from praxis_agent.events import ApprovalRequested, ToolCallPrepared, ToolFinished
from praxis_agent.hooks import HookManager
from praxis_agent.model import FunctionCall
from praxis_agent.planning import RunState
from praxis_agent.tool_execution import ToolCallExecutor
from praxis_agent.tool_extensions import ToolExtensions
from praxis_agent.tools import Tool, ToolContext, ToolRegistry, ToolResult


def load(tmp_path, source, name="policy"):
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    config = ToolExtensionConfig(name, path)
    manager = HookManager((config,))
    module = next(m for m in sys.modules.values() if getattr(m, "__file__", None) == str(path))
    return manager, module, config


def executor(tmp_path, hooks, handler=None, **options):
    registry = ToolRegistry()
    registry.register(Tool("sample", "Sample tool", {
        "type": "object", "properties": {"value": {"type": "integer"}},
        "required": ["value"], "additionalProperties": False,
    }, handler or (lambda context, value: str(value)), strict=False, programmatic=True, parallel_safe=True))
    events = []
    engine = ToolCallExecutor(registry, ToolContext(tmp_path, RunState("hooks")), events.append,
                              hooks=hooks, session_id="session", agent_id="worker", **options)
    return engine, events


def call(engine, arguments=None, name="sample"):
    return engine.execute("turn", FunctionCall("call", name, json.dumps(arguments or {})))


def test_argument_chain_snapshots_and_final_approval(tmp_path):
    hooks, module, _ = load(tmp_path, '''
        from praxis_agent.sdk import hook, HookDecision
        seen = []
        @hook("before_tool_call", tools=["sam*"])
        def first(event):
            seen.append(event)
            event.arguments["ignored"] = True
            return HookDecision.allow(arguments={"value": 2}, requires_approval=True)
        @hook("before_tool_call")
        def second(event):
            assert event.original_arguments == {}
            assert event.arguments == {"value": 2}
            assert event.approval_requested
            return HookDecision.allow(arguments={"value": 3})
        @hook("before_tool_call", tools=["other"])
        def unmatched(event): raise AssertionError()
    ''')
    approvals, invoked = [], []
    def approve(name, args):
        approvals.append(dict(args))
        args["value"] = 99  # Approval consumers cannot alter the approved call.
        return True
    engine, events = executor(tmp_path, hooks,
        handler=lambda context, value: invoked.append(value) or str(value),
        approve_tool_call=approve, bypass_approvals=True)
    result = call(engine)
    assert result.success and result.output == "3"
    assert approvals == [{"value": 3}] and invoked == [3]
    prepared = next(e for e in events if isinstance(e, ToolCallPrepared))
    assert prepared.arguments == {"value": 3} and prepared.requires_approval
    event = module.seen[0]
    assert (event.session_id, event.agent_id, event.turn_id, event.call_id) == ("session", "worker", "turn", "call")
    assert event.workspace == tmp_path and event.parameters["required"] == ["value"]


@pytest.mark.parametrize("decision", ["HookDecision.deny('blocked')", "HookDecision.allow(arguments={'value': 'bad'})", "1 / 0"])
def test_rejected_calls_never_execute_or_approve(tmp_path, decision):
    hooks, module, _ = load(tmp_path, f'''
        from praxis_agent.sdk import hook, HookDecision
        results = []
        @hook("before_tool_call")
        def before(event): return {decision}
        @hook("on_tool_error")
        def after(event): results.append(event)
    ''')
    invoked = []
    engine, events = executor(tmp_path, hooks, handler=lambda context, value: invoked.append(value),
                              approve_tool_call=lambda *_: pytest.fail("unexpected approval"))
    assert not call(engine, {"value": 1}).success
    assert invoked == [] and module.results == []
    assert not any(isinstance(e, ApprovalRequested) for e in events)


@pytest.mark.parametrize("fail", [False, True])
def test_result_chain_replaces_all_channels_and_keeps_execution_status(tmp_path, fail):
    hooks, module, _ = load(tmp_path, '''
        from praxis_agent.sdk import hook, HookOutput
        seen = []
        def clean(event):
            seen.append(event)
            return HookOutput("clean", {"safe": True})
        def suffix(event):
            assert "secret" in event.original_result.output
            assert event.result.output == "clean"
            return HookOutput("final", event.result.structured_output, [{"type": "text", "text": "final"}])
        success = hook("after_tool_call")(clean)
        def failure(event): return clean(event)
        failure = hook("on_tool_error")(failure)
        @hook("after_tool_call")
        def success_end(event): return suffix(event)
        @hook("on_tool_error")
        def failure_end(event): return suffix(event)
    ''')
    saved = []
    def handler(context, value):
        if fail: raise ValueError("secret failure")
        return ToolResult(True, "secret", {"secret": 1},
                          model_content=[{"type": "text", "text": "secret"}], on_saved=lambda: saved.append(True))
    engine, events = executor(tmp_path, hooks, handler=handler)
    result = call(engine, {"value": 1})
    assert result.success is not fail
    assert result.output == "final" and result.structured_output == {"safe": True}
    assert result.model_content == [{"type": "text", "text": "final"}]
    assert module.seen[0].duration_ms >= 0
    assert next(e for e in events if isinstance(e, ToolFinished)).result == result
    if not fail:
        result.on_saved()
        assert saved == [True]


def test_result_hook_failure_hides_raw_content_and_live_command_output(tmp_path):
    hooks, _, _ = load(tmp_path, '''
        from praxis_agent.sdk import hook
        @hook("after_tool_call")
        def broken(event): raise ValueError(event.result.output)
    ''')
    def handler(context, value):
        assert context.on_command_output is None
        return ToolResult(True, "secret", {"secret": True}, model_content=[{"type": "text", "text": "secret"}])
    engine, events = executor(tmp_path, hooks, handler=handler)
    result = call(engine, {"value": 1})
    assert result.success and "processing the result" in result.output
    assert "secret" not in result.output and result.structured_output is None and result.model_content is None
    assert "secret" not in repr(events)


def test_nested_parallel_and_python_tools_share_hooks(tmp_path):
    hooks, module, config = load(tmp_path, '''
        from praxis_agent.sdk import hook, tool, HookDecision, HookOutput
        seen = []
        @tool(description="extension", requires_approval=False, parallel_safe=True)
        def query(value: int): return {"value": value}
        @hook("before_tool_call", tools=["policy__*"])
        def before(event):
            seen.append(event)
            return HookDecision.allow(arguments={"value": event.arguments["value"] + 1})
        @hook("after_tool_call", tools=["policy__*"])
        def after(event): return HookOutput("filtered", event.result.structured_output)
    ''')
    engine, _ = executor(tmp_path, hooks)
    ToolExtensions((config,)).register(engine.tool_registry)
    def bridge(context):
        a = context.invoke_programmatic_tool("policy__query", {"value": 1})
        b = context.invoke_parallel_tools([
            {"name": "policy__query", "arguments": {"value": 2}},
            {"name": "policy__query", "arguments": {"value": 3}},
        ], 2)
        return ToolResult(True, "ok", {"values": [r.structured_output["value"] for r in [a, *b]]})
    engine.tool_registry.register(Tool("bridge", "nested", {"type": "object"}, bridge, strict=False))
    result = call(engine, name="bridge")
    assert result.structured_output == {"values": [2, 3, 4]}
    assert len(module.seen) == 3
    assert all(e.parent_call_id == "call" and e.programmatic_call for e in module.seen)
    assert len({e.call_id for e in module.seen}) == 3
    # Explicit outer parallel dispatch follows the same pipeline.
    results = list(engine.execute_many("parallel", [FunctionCall(str(i), "policy__query", '{"value": 4}') for i in range(2)]))
    assert all(r.output == "filtered" and r.structured_output == {"value": 5} for _, r in results)


def test_hook_config_loading_and_reload_boundary(tmp_path, model_configuration_file, service):
    path = model_configuration_file.parent / "hooks.py"
    path.write_text('from praxis_agent.sdk import hook\n@hook("before_tool_call")\ndef observe(event): pass\n')
    data = yaml.safe_load(model_configuration_file.read_text())
    data["hook_extensions"] = [{"id": "policy", "path": "hooks.py"}]
    model_configuration_file.write_text(yaml.safe_dump(data))
    config = load_config(model_configuration_file)
    assert config.hook_extensions == (ToolExtensionConfig("policy", path),)
    harness = service()
    assert len(harness.manager.hooks.handlers) == 1
    data["hook_extensions"][0]["enabled"] = False
    model_configuration_file.write_text(yaml.safe_dump(data))
    from praxis_agent.service.errors import ServiceError
    with pytest.raises(ServiceError, match="restart"):
        harness.manager.configure({"config_path": str(model_configuration_file)})
    assert len(harness.manager.hooks.handlers) == 1
    assert HookManager((replace(config.hook_extensions[0], enabled=False),)).handlers == ()


def test_builtin_file_rewrite_still_requires_approval_and_tracks_final_path(tmp_path):
    from praxis_agent.tools.registry import create_default_registry
    hooks, module, _ = load(tmp_path, '''
        from praxis_agent.sdk import hook, HookDecision, HookOutput
        completed = []
        @hook("before_tool_call", tools=["write_file"])
        def redirect(event):
            return HookDecision.allow(arguments={**event.arguments, "relative_path": "actual.txt"})
        @hook("after_tool_call", tools=["write_file"])
        def clean(event):
            completed.append(event)
            return HookOutput("written")
    ''')
    engine, _ = executor(tmp_path, hooks, approve_tool_call=lambda *_: False)
    engine.tool_registry = create_default_registry()
    tool = next(t for t in engine.tool_registry.tools if t.name == "write_file")
    engine.tool_registry.unregister("write_file")
    engine.tool_registry.register(replace(tool, requires_approval=True))
    arguments = {"relative_path": "original.txt", "content": "text"}
    assert not call(engine, arguments, "write_file").success
    assert not (tmp_path / "actual.txt").exists() and module.completed == []
    engine.approve_tool_call = lambda *_: True
    result = call(engine, arguments, "write_file")
    assert result.success and result.output == "written"
    assert result.file_change is None
    assert (tmp_path / "actual.txt").read_text() == "text"
    assert not (tmp_path / "original.txt").exists()
    assert "actual.txt" in engine.context.run_state.changed_files


def test_terminal_renders_transformed_command_payload():
    from io import StringIO
    from rich.console import Console
    from praxis_agent.terminal.views import AgentView
    from praxis_agent.terminal.cells import ToolHeading, tool_preview
    view = AgentView("main", "Praxis")
    view.consume(ToolCallPrepared("turn", "call", "run_command", {"command": "echo test"}, False))
    view.consume(ToolFinished("turn", "call", "run_command", ToolResult(True, "filtered", {"safe": True})))
    block = view.committed_blocks[-1]
    assert block.state == "completed"
    capture = StringIO()
    console = Console(file=capture)
    console.print(ToolHeading((block,)))
    console.print(tool_preview((block,)))
    assert "safe" in capture.getvalue()
