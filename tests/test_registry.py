from pathlib import Path

from praxis_agent.planning import RunState
from praxis_agent.tools import ToolContext, create_default_registry


def _context(workspace: Path, task: str = "test tools") -> ToolContext:
    return ToolContext(workspace, RunState(task))


def test_default_registry_contains_all_builtin_tools() -> None:
    registry = create_default_registry()

    assert registry.names == (
        "read_file",
        "edit_file",
        "create_file",
        "list_files",
        "find_files",
        "search_text",
        "run_command",
        "update_plan",
        "delete",
        "compress",
        "search_history",
        "read_history",
    )
    assert registry.tools[0].description == (
        "Read a bounded line range from a UTF-8 workspace file."
    )
    assert registry.requires_approval("run_command") is True


def test_registry_returns_model_tool_definitions() -> None:
    registry = create_default_registry()

    definitions = registry.definitions()

    assert [definition["name"] for definition in definitions] == list(
        registry.names
    )
    assert definitions[0] == {
        "type": "function",
        "name": "read_file",
        "description": (
            "Read a bounded line range from a UTF-8 workspace file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": "Path to read, relative to the workspace root.",
                },
                "start_line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "First one-based line to return.",
                },
                "max_lines": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 400,
                    "description": "Maximum number of lines to return.",
                },
            },
            "required": ["relative_path", "start_line", "max_lines"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def test_registry_returns_new_parameter_schema_copies() -> None:
    registry = create_default_registry()

    first_definitions = registry.definitions()
    first_definitions[0]["parameters"]["required"].clear()

    second_definitions = registry.definitions()

    assert second_definitions[0]["parameters"]["required"] == [
        "relative_path",
        "start_line",
        "max_lines",
    ]


def test_registry_executes_a_tool_with_its_workspace(tmp_path: Path) -> None:
    source_file = tmp_path / "notes.txt"
    source_file.write_text("hello\n", encoding="utf-8")
    registry = create_default_registry()

    result = registry.execute(
        "read_file",
        {"relative_path": "notes.txt"},
        context=_context(tmp_path),
    )

    assert result.success is True
    assert result.output == "[lines 1-1 of 1]\nhello\n"


def test_registry_returns_an_error_for_unknown_tools(tmp_path: Path) -> None:
    registry = create_default_registry()

    result = registry.execute(
        "missing_tool",
        {},
        context=_context(tmp_path),
    )

    assert result.success is False
    assert result.output == "unknown tool: missing_tool"


def test_registry_returns_an_error_for_invalid_arguments(tmp_path: Path) -> None:
    registry = create_default_registry()

    result = registry.execute(
        "read_file",
        {},
        context=_context(tmp_path),
    )

    assert result.success is False
    assert result.output.startswith("TypeError:")


def test_update_plan_uses_runtime_context_not_model_arguments(
    tmp_path: Path,
) -> None:
    registry = create_default_registry()
    first_context = _context(tmp_path, "first task")
    second_context = _context(tmp_path, "second task")

    definition = next(
        item for item in registry.definitions() if item["name"] == "update_plan"
    )
    assert definition["name"] == "update_plan"
    assert set(definition["parameters"]["properties"]) == {"plan"}

    first_result = registry.execute(
        "update_plan",
        {"plan": [{"step": "Inspect", "status": "completed"}]},
        context=first_context,
    )
    second_result = registry.execute(
        "update_plan",
        {"plan": [{"step": "Implement", "status": "in_progress"}]},
        context=second_context,
    )

    assert first_result.success is True
    assert second_result.success is True
    assert first_context.run_state.plan[0].description == "Inspect"
    assert second_context.run_state.plan[0].description == "Implement"
