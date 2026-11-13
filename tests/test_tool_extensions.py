import json
import sys
import textwrap
from dataclasses import replace

import pytest
import yaml

from praxis_agent.config import ToolExtensionConfig, load_config
from praxis_agent.events import ApprovalRequested, ToolStarted, ToolFinished
from praxis_agent.model import FunctionCall
from praxis_agent.planning import RunState
from praxis_agent.sdk import tool
from praxis_agent.tool_execution import ToolCallExecutor
from praxis_agent.tool_extensions import ToolExtensions
from praxis_agent.tools import Tool, ToolContext, ToolRegistry


def extension(tmp_path, source, name="sample"):
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return ToolExtensionConfig(name, path)


def executor(tmp_path, extensions, approve=lambda *_: True, **options):
    registry = ToolRegistry()
    extensions.register(registry)
    events = []
    execute = ToolCallExecutor(registry, ToolContext(tmp_path, RunState("extensions")), events.append,
                               approve_tool_call=approve, **options)
    return execute, events


def call(execute, arguments=None, name="sample__query"):
    return execute.execute("turn", FunctionCall("call", name, json.dumps(arguments or {})))


def test_decorator_preserves_function_and_both_import_forms(tmp_path):
    def original(value: str):
        return value
    assert tool(description="identity")(original) is original
    source = extension(tmp_path, '''
        from praxis_agent.sdk import tool
        import praxis_agent.sdk as Praxis
        from praxis_agent.sdk import tool as imported_helper
        def helper(): raise AssertionError("never called")
        @tool(description="first")
        def first(value: str): return value
        @Praxis.tool(description="second", name="renamed", requires_approval=False,
                     parallel_safe=True, programmatic=False)
        def second(): return "ok"
        alias = first
    ''')
    catalog = ToolExtensions((source,))
    assert [item.name for item in catalog.tools] == ["sample__first", "sample__renamed"]
    first, second = catalog.tools
    assert first.requires_approval and not first.parallel_safe and first.programmatic
    assert not second.requires_approval and second.parallel_safe and not second.programmatic


def test_config_paths_disabled_and_duplicate_ids(tmp_path, model_configuration_file, monkeypatch):
    config = yaml.safe_load(model_configuration_file.read_text())
    folder = model_configuration_file.parent
    marker = folder / "executed"
    (folder / "disabled.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()")
    config["tool_extensions"] = [{"id": "disabled", "path": "disabled.py", "enabled": False}]
    model_configuration_file.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    loaded = load_config(model_configuration_file)
    assert loaded.tool_extensions[0].path == folder / "disabled.py"
    assert ToolExtensions(loaded.tool_extensions).tools == ()
    assert not marker.exists()
    config["tool_extensions"].append(config["tool_extensions"][0])
    model_configuration_file.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Duplicate tool extension id"):
        load_config(model_configuration_file)


@pytest.mark.parametrize("identifier", ["9bad", "a.b", "a" * 33])
def test_invalid_ids(model_configuration_file, identifier):
    config = yaml.safe_load(model_configuration_file.read_text())
    config["tool_extensions"] = [{"id": identifier, "path": "unused.py"}]
    model_configuration_file.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="tool_extensions"):
        load_config(model_configuration_file)


def test_primitives_optional_defaults_schema_and_validation(tmp_path):
    catalog = ToolExtensions((extension(tmp_path, '''
        from __future__ import annotations
        from typing import Optional, Literal
        from praxis_agent.sdk import tool
        @tool(description="types")
        def query(text: str, number: int, ratio: float, flag: bool,
                  nullable: Optional[str], mode: Literal["fast", "slow"], *, default: int = 7,
                  optional: str | None = None) -> "UndefinedReturnType":
            return locals()
    '''),))
    schema = catalog.tools[0].parameters
    assert set(schema["required"]) == {"text", "number", "ratio", "flag", "nullable", "mode"}
    assert schema["additionalProperties"] is False
    assert schema["properties"]["nullable"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    execute, events = executor(tmp_path, catalog)
    values = dict(text="x", number=3, ratio=1.5, flag=False, nullable=None, mode="fast")
    result = call(execute, values)
    assert result.success and result.structured_output == dict(values, default=7, optional=None)
    for invalid in [dict(values, number="3"), dict(values, flag="false"), dict(values, number=True),
                    dict(values, extra=1), dict(values, mode="bad"), {k: v for k, v in values.items() if k != "nullable"}]:
        events.clear()
        assert not call(execute, invalid).success
        assert not any(isinstance(event, (ToolStarted, ApprovalRequested)) for event in events)


COMPLEX = '''
from __future__ import annotations
from dataclasses import dataclass, field
from praxis_agent.sdk import tool
effects = []
def defaults():
    effects.append("default")
    return ["default"]
@dataclass(frozen=True, slots=True)
class Child:
    name: str
    count: int = 1
    def __post_init__(self): effects.append("child")
@dataclass
class Query:
    child: Child
    children: list[Child]
    mapping: dict[str, Child]
    maybe: Child | None
    tags: list[str] = field(default_factory=defaults)
    hidden: int = field(default=42, init=False)
@tool(description="nested", parallel_safe=True)
def query(query: Query):
    assert isinstance(query, Query)
    assert isinstance(query.child, Child)
    assert all(isinstance(x, Child) for x in query.children)
    assert all(isinstance(x, Child) for x in query.mapping.values())
    assert query.maybe is None or isinstance(query.maybe, Child)
    effects.append("function")
    return query
'''


def complex_args():
    return {"query": {"child": {"name": "first"}, "children": [{"name": "second"}],
                      "mapping": {"key": {"name": "third"}}, "maybe": {"name": "fourth"}}}


def test_nested_dataclasses_and_approval_before_construction(tmp_path):
    config = extension(tmp_path, COMPLEX)
    catalog = ToolExtensions((config,))
    module = next(module for module in sys.modules.values()
                  if getattr(module, "__file__", None) == str(config.path))
    assert module.effects == []  # default_factory is not called when loading.
    execute, events = executor(tmp_path, catalog, approve=lambda *_: False)
    assert not call(execute, complex_args()).success
    assert module.effects == []
    execute.bypass_approvals = True
    result = call(execute, complex_args())
    assert result.success, result.output
    assert result.structured_output["child"] == {"name": "first", "count": 1}
    assert result.structured_output["hidden"] == 42
    assert result.structured_output["tags"] == ["default"]
    assert module.effects == ["child"] * 4 + ["default", "function"]
    assert any(isinstance(event, ToolFinished) for event in events)
    nullable = complex_args()
    nullable["query"]["maybe"] = None
    assert call(execute, nullable).structured_output["maybe"] is None
    for mutate in [lambda q: q.update(hidden=7), lambda q: q["children"][0].update(count="3"),
                   lambda q: q["mapping"]["key"].update(extra=1), lambda q: q["child"].pop("name")]:
        args = complex_args()
        mutate(args["query"])
        previous = module.effects[:]
        assert not call(execute, args).success
        assert module.effects == previous


@pytest.mark.parametrize("annotation", ["Missing", "Arbitrary", "Callable[[], str]"])
def test_unsupported_inputs_have_tool_location(tmp_path, annotation):
    config = extension(tmp_path, f'''
        from __future__ import annotations
        from typing import Callable
        class Arbitrary: pass
        from praxis_agent.sdk import tool
        @tool(description="bad")
        def query(value: {annotation}): pass
    ''')
    with pytest.raises(ValueError, match="extension sample, function query"):
        ToolExtensions((config,))


@pytest.mark.parametrize("definition", ["async def query(x: str): return x", "def query(x: str): yield x",
    "def query(x: str, /): return x", "def query(**kwargs: str): pass", "def query(x): pass"])
def test_invalid_function_signatures(tmp_path, definition):
    config = extension(tmp_path, 'from praxis_agent.sdk import tool\n@tool(description="bad")\n' + definition)
    with pytest.raises(ValueError, match="extension sample, function query"):
        ToolExtensions((config,))


def test_import_errors_and_failed_batch_cleanup(tmp_path):
    good = extension(tmp_path, 'from praxis_agent.sdk import tool\n@tool(description="good")\ndef query(): return 1', "good")
    bad = extension(tmp_path, 'import nonexistent_praxis_extension_dependency', "bad")
    previous = set(sys.modules)
    with pytest.raises(ValueError, match="Praxis service Python environment"):
        ToolExtensions((good, bad))
    assert not [name for name in set(sys.modules) - previous if name.startswith("_praxis_extension_")]
    with pytest.raises(ValueError, match="existing local .py"):
        ToolExtensions((replace(bad, path=tmp_path / "missing.py"),))


def test_duplicate_names_and_atomic_registry_registration(tmp_path):
    bad = extension(tmp_path, '''
        from praxis_agent.sdk import tool
        @tool(description="one", name="same")
        def one(): pass
        @tool(description="two", name="same")
        def two(): pass
    ''')
    with pytest.raises(ValueError, match="duplicate tool name"):
        ToolExtensions((bad,))
    good = extension(tmp_path, '''
        from praxis_agent.sdk import tool
        @tool(description="one")
        def one(): pass
        @tool(description="two")
        def two(): pass
    ''', "good")
    catalog = ToolExtensions((good,))
    registry = ToolRegistry()
    registry.register(catalog.tools[1])
    with pytest.raises(ValueError, match="conflict"):
        catalog.register(registry)
    assert registry.names == ("good__two",)


@pytest.mark.parametrize("options", ['name="invalid.name"', 'name="x" * 60', 'description=""',
                                    'requires_approval="yes"'])
def test_invalid_metadata(tmp_path, options):
    args = options if options.startswith("description=") else 'description="test", ' + options
    config = extension(tmp_path, f'from praxis_agent.sdk import tool\n@tool({args})\ndef query(): pass')
    with pytest.raises(ValueError, match="extension sample, function query"):
        ToolExtensions((config,))


@pytest.mark.parametrize("expression, expected", [('"hello"', "hello"), ('None', "null"),
    ('[1, True, None]', '[1, true, null]'), ('{"x": [1]}', '{"x": [1]}')])
def test_outputs(tmp_path, expression, expected):
    catalog = ToolExtensions((extension(tmp_path, f'from praxis_agent.sdk import tool\n@tool(description="result")\ndef query(): return {expression}'),))
    execute, _ = executor(tmp_path, catalog)
    result = call(execute)
    assert result.success and result.output == expected
    assert result.structured_output is None or isinstance(result.structured_output, dict)


@pytest.mark.parametrize("body, message", [('return object()', 'Unable to serialize unknown type'),
    ('return float("nan")', 'not JSON compliant'),
    ('x = []; x.append(x); return x', 'Circular reference'), ('raise RuntimeError("business failure")', 'business failure')])
def test_output_and_business_errors(tmp_path, body, message):
    catalog = ToolExtensions((extension(tmp_path, f'from praxis_agent.sdk import tool\n@tool(description="error")\ndef query(): {body}'),))
    execute, _ = executor(tmp_path, catalog)
    result = call(execute)
    assert not result.success and message in result.output


def test_programmatic_execution_uses_validation_and_approval(tmp_path):
    catalog = ToolExtensions((extension(tmp_path, COMPLEX),))
    execute, events = executor(tmp_path, catalog, approve=lambda *_: False)
    def bridge(context, arguments):
        return context.invoke_programmatic_tool("sample__query", arguments)
    execute.tool_registry.register(Tool("bridge", "test programmatic entry", {
        "type": "object", "properties": {"arguments": {"type": "object"}}, "required": ["arguments"],
    }, bridge, strict=False))
    assert not call(execute, {"arguments": complex_args()}, name="bridge").success
    assert any(isinstance(event, ApprovalRequested) for event in events)
    execute.approve_tool_call = lambda *_: True
    assert call(execute, {"arguments": complex_args()}, name="bridge").success
    assert not call(execute, {"arguments": {}}, name="bridge").success


def test_real_python_kernel_can_call_extension(tmp_path):
    from praxis_agent.tools.code import CodeRuntime
    catalog = ToolExtensions((extension(tmp_path, COMPLEX),))
    approvals = []
    execute, events = executor(tmp_path, catalog, approve=lambda name, args: approvals.append(name) or True)
    runtime = CodeRuntime(output_dir=tmp_path / "outputs")
    execute.tool_registry.register(runtime.tool())
    try:
        result = call(execute, {"code": f"call_tool('sample__query', {complex_args()!r})['child']['name']"}, name="run_code")
        assert result.success, result.output
        assert json.loads(result.output)["result"] == "first"
        assert "sample__query" in approvals
        assert any(isinstance(event, ToolStarted) and event.tool_name == "sample__query" for event in events)
    finally:
        runtime.close()


def test_parallel_safe_tools_reuse_existing_scheduler(tmp_path):
    catalog = ToolExtensions((extension(tmp_path, '''
        from threading import Barrier
        from praxis_agent.sdk import tool
        barrier = Barrier(2)
        @tool(description="parallel", requires_approval=False, parallel_safe=True)
        def query(value: int):
            barrier.wait(timeout=5)
            return {"value": value}
    '''),))
    execute, _ = executor(tmp_path, catalog)
    calls = [FunctionCall(str(index), "sample__query", json.dumps({"value": index})) for index in range(2)]
    results = list(execute.execute_many("parallel", calls))
    assert [result.structured_output for _, result in results] == [{"value": 0}, {"value": 1}]


def test_imported_marked_functions_and_forged_metadata(tmp_path, monkeypatch):
    from types import ModuleType
    imported = ModuleType("installed_extension_dependency")
    exec('from praxis_agent.sdk import tool\n@tool(description="external")\ndef external(): pass', imported.__dict__)
    monkeypatch.setitem(sys.modules, imported.__name__, imported)
    config = extension(tmp_path, 'from installed_extension_dependency import external')
    assert ToolExtensions((config,)).tools == ()
    forged = extension(tmp_path, 'def query(): pass\nquery.__praxis_tool__ = {"description": "forged"}', "forged")
    with pytest.raises(ValueError, match="invalid SDK tool metadata"):
        ToolExtensions((forged,))


def test_programmatic_opt_out(tmp_path):
    catalog = ToolExtensions((extension(tmp_path, '''
        from praxis_agent.sdk import tool
        @tool(description="direct only", requires_approval=False, programmatic=False)
        def query(): return "ok"
    '''),))
    execute, _ = executor(tmp_path, catalog)
    assert call(execute).success
    execute.context = replace(execute.context, programmatic_call=True)
    result = call(execute)
    assert not result.success and "unavailable for programmatic" in result.output


def test_business_parameters_do_not_collide_with_adapter_context(tmp_path):
    catalog = ToolExtensions((extension(tmp_path, '''
        from praxis_agent.sdk import tool
        @tool(description="business parameters")
        def query(context: str, arguments: dict[str, int]):
            return {"context": context, "arguments": arguments}
    '''),))
    execute, _ = executor(tmp_path, catalog)
    values = {"context": "business data", "arguments": {"count": 3}}
    result = call(execute, values)
    assert result.success and result.structured_output == values


def test_cancelled_during_constructor_does_not_start_function(tmp_path):
    catalog = ToolExtensions((extension(tmp_path, '''
        from dataclasses import dataclass
        from praxis_agent.sdk import tool
        effects = []
        @dataclass
        class Query:
            value: str
            def __post_init__(self): effects.append("constructed")
        @tool(description="cancel")
        def query(value: Query):
            effects.append("called")
            return "unexpected"
    '''),))
    function = catalog.tools[0].handler
    # Stop at the post-conversion cancellation checkpoint.
    checkpoints = []
    def check():
        checkpoints.append(True)
        if len(checkpoints) == 2:
            raise RuntimeError("cancelled after conversion")
    context = ToolContext(tmp_path, RunState("cancel"), check_execution=check)
    with pytest.raises(RuntimeError, match="cancelled after conversion"):
        function(context, value={"value": "x"})
    module = next(module for module in sys.modules.values()
                  if getattr(module, "__file__", None) == str(tmp_path / "sample.py"))
    assert module.effects == ["constructed"]


def test_service_rejects_extension_changes_before_mutation(service, model_configuration_file, tmp_path):
    harness = service()
    original, factory = harness.manager.config, harness.manager.factory
    config = yaml.safe_load(model_configuration_file.read_text())
    config["tool_extensions"] = [{"id": "new", "path": "missing.py", "enabled": False}]
    new = tmp_path / "new.yaml"
    new.write_text(yaml.safe_dump(config))
    from praxis_agent.service.errors import ServiceError
    with pytest.raises(ServiceError, match="restart"):
        harness.manager.configure({"config_path": str(new), "bypass_approvals": True})
    assert harness.manager.config is original
    assert harness.manager.factory is factory
    assert not harness.manager.bypass_approvals


def test_service_shares_loaded_tools_across_sessions_and_workers(tmp_path, model_configuration_file, monkeypatch):
    from praxis_agent.model import ModelResponse
    from praxis_agent.multi_agent.models import AgentStatus
    from praxis_agent.service.manager import SessionManager
    from praxis_agent.service.store import ServiceStore
    from praxis_agent.sessions import SQLiteSessionStore
    marker = tmp_path / "imports.txt"
    source = extension(tmp_path, f'''
        from pathlib import Path
        from praxis_agent.sdk import tool, hook
        @hook("before_tool_call")
        def observe(event): pass
        with Path({str(marker)!r}).open("a") as file: file.write("imported\\n")
        @tool(description="local")
        def query(value: str = "ok"): return value
    ''')
    config = yaml.safe_load(model_configuration_file.read_text())
    config["tool_extensions"] = [{"id": source.id, "path": str(source.path)}]
    config["hook_extensions"] = [{"id": source.id, "path": str(source.path)}]
    model_configuration_file.write_text(yaml.safe_dump(config))
    seen = []
    class Client:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
        def create_response(self, input_items, tools, **kwargs):
            seen.append(tools)
            return ModelResponse(({"role": "assistant", "content": "done"},), "done", ())
    monkeypatch.setattr("praxis_agent.factory.OpenAIModelClient", Client)
    monkeypatch.setattr("praxis_agent.factory.discover_skills", lambda _: ())
    from praxis_agent.mcp import MCPConnections
    mcp = MCPConnections(tmp_path / "missing-mcp.json")
    mcp.start()
    manager = SessionManager(ServiceStore(SQLiteSessionStore(tmp_path / "state.sqlite")), mcp=mcp)
    agents = []
    try:
        assert marker.read_text() == "imported\n"  # eager service startup
        for _ in range(2):
            agents.append(manager.factory(tmp_path, session_database_path=tmp_path / "agents.sqlite"))
        assert agents[0].hooks is agents[1].hooks is manager.hooks
        assert len(manager.hooks.handlers) == 1
        assert "sample__query" in agents[0].tool_registry.names
        assert agents[0].tool_registry is not agents[1].tool_registry
        agents[0].tool_registry.unregister("sample__query")
        assert "sample__query" in agents[1].tool_registry.names
        runtime = agents[1].collaboration
        spawned = runtime.spawn("done", working_directory=str(tmp_path))
        with runtime.condition:
            assert runtime.condition.wait_for(lambda: runtime.handles[spawned["agent_id"]].status == AgentStatus.IDLE, timeout=10)
        assert runtime.handles[spawned["agent_id"]].agent.hooks is manager.hooks
        assert runtime.handles[spawned["agent_id"]].agent.hook_session_id == agents[1].session_id
        assert any(any(item["name"] == "sample__query" for item in tools) for tools in seen)
        manager.configure({"config_path": str(model_configuration_file)})
        assert marker.read_text() == "imported\n"
    finally:
        for agent in agents:
            agent.close()
        manager.close()


def test_pydantic_types_constraints_and_recursive_objects(tmp_path):
    catalog = ToolExtensions((extension(tmp_path, '''
        from __future__ import annotations
        from dataclasses import dataclass
        from datetime import datetime
        from enum import Enum
        from pathlib import Path
        from typing import Annotated, Literal
        from uuid import UUID
        from pydantic import BaseModel, ConfigDict, Field
        from praxis_agent.sdk import tool
        class Environment(str, Enum):
            PROD = "prod"
        class Query(BaseModel):
            model_config = ConfigDict(extra="forbid")
            release_id: UUID
            since: datetime
            environment: Environment
            count: Annotated[int, Field(ge=1, description="Number of releases")]
        @dataclass
        class Node:
            name: str
            child: Node | None = None
        @tool(description="rich types")
        def query(query: Query, node: Node, tags: set[str], pair: tuple[int, str],
                  choice: int | str, path: Path):
            assert isinstance(query, Query) and isinstance(query.release_id, UUID)
            assert isinstance(query.since, datetime) and query.environment is Environment.PROD
            assert isinstance(node.child, Node) and isinstance(tags, set)
            assert isinstance(pair, tuple) and isinstance(path, Path)
            return {"query": query, "node": node, "tags": tags, "pair": pair,
                    "choice": choice, "path": path}
    '''),))
    execute, events = executor(tmp_path, catalog)
    values = {"query": {"release_id": "550e8400-e29b-41d4-a716-446655440000",
                        "since": "2026-11-13T10:00:00+08:00", "environment": "prod", "count": 2},
              "node": {"name": "root", "child": {"name": "leaf"}},
              "tags": ["one"], "pair": [1, "two"], "choice": 3, "path": "report.txt"}
    result = call(execute, values)
    assert result.success, result.output
    assert result.structured_output == {**values, "node": {"name": "root", "child": {"name": "leaf", "child": None}}}
    for change in ({"count": 0}, {"count": "2"}, {"extra": 1}):
        events.clear()
        assert not call(execute, {**values, "query": {**values["query"], **change}}).success
        assert not any(isinstance(event, (ToolStarted, ApprovalRequested)) for event in events)
    for change in ({"release_id": "invalid"}, {"since": "not-a-date"}):
        assert not call(execute, {**values, "query": {**values["query"], **change}}).success


def test_custom_type_validators_only_run_after_approval(tmp_path):
    config = extension(tmp_path, '''
        from typing import Annotated
        from pydantic import AfterValidator, PlainSerializer
        from praxis_agent.sdk import tool
        effects = []
        def positive(value):
            effects.append("validated")
            if value < 0: raise ValueError("must be positive")
            return value
        Value = Annotated[int, AfterValidator(positive), PlainSerializer(lambda v: str(v))]
        from pydantic import BaseModel
        class Query(BaseModel):
            value: Value
        @tool(description="custom validation")
        def query(value: Query):
            effects.append("called")
            return value
    ''')
    catalog = ToolExtensions((config,))
    module = next(module for module in sys.modules.values()
                  if getattr(module, "__file__", None) == str(config.path))
    execute, _ = executor(tmp_path, catalog, approve=lambda *_: False)
    assert module.effects == []
    assert not call(execute, {"value": {"value": 1}}).success
    assert module.effects == []
    execute.bypass_approvals = True
    result = call(execute, {"value": {"value": 1}})
    assert result.success and result.structured_output == {"value": "1"}
    assert module.effects == ["validated", "called"]
    module.effects.clear()
    result = call(execute, {"value": {"value": -1}})
    assert not result.success and "must be positive" in result.output
    assert module.effects == ["validated"]
