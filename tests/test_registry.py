import json
import sys
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate

from praxis_agent.planning import RunState
from praxis_agent.tools import ToolContext, create_default_registry


def _context(workspace: Path, task: str = "test tools") -> ToolContext:
    return ToolContext(workspace, RunState(task))


def test_default_registry_contains_all_builtin_tools() -> None:
    registry = create_default_registry()

    assert registry.names == (
        "read_file",
        "edit_file",
        "write_file",
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
    assert registry.requires_approval("run_command") is True


def test_registry_returns_model_tool_definitions() -> None:
    registry = create_default_registry()

    definitions = registry.definitions()

    assert [definition["name"] for definition in definitions] == list(
        registry.names
    )
    definition = definitions[0]
    assert definition["type"] == "function"
    assert definition["name"] == "read_file"
    assert definition["description"]
    assert definition["strict"] is False
    schema = definition["parameters"]
    for arguments in (
        {"relative_path": "notes.txt"},
        {"relative_path": "notes.txt", "start_line": 2, "end_line": 3},
        {"relative_path": "notes.txt", "start_offset": 0, "end_line": None, "end_offset": None},
    ):
        validate(arguments, schema)
    for arguments in ({}, {"relative_path": "notes.txt", "start_line": 0},
                      {"relative_path": "notes.txt", "max_lines": 2}):
        with pytest.raises(ValidationError):
            validate(arguments, schema)


def test_registry_returns_new_parameter_schema_copies() -> None:
    registry = create_default_registry()

    first_definitions = registry.definitions()
    first_definitions[0]["parameters"]["required"].clear()

    second_definitions = registry.definitions()

    assert second_definitions[0]["parameters"]["required"] == [
        "relative_path",
    ]


def test_registry_executes_a_tool_with_its_workspace(tmp_path: Path) -> None:
    source_file = tmp_path / "notes.txt"
    source_file.write_bytes(b"hello\n")
    registry = create_default_registry()

    result = registry.execute(
        "read_file",
        {"relative_path": "notes.txt"},
        context=_context(tmp_path),
    )

    assert result.success is True
    expected = {"text": "hello\n", "truncated": False, "next_position": None, "note": ""}
    assert json.loads(result.output) == expected
    assert result.structured_output == expected


@pytest.mark.parametrize("directory", ["default", "nested", "absolute"])
def test_command_workdir_reaches_the_process(tmp_path: Path, directory: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = {"default": workspace, "nested": workspace / "nested",
              "absolute": tmp_path / "外部 project"}[directory]
    target.mkdir(exist_ok=True)
    (target / "input.txt").write_text(directory, encoding="utf-8")
    arguments = {"argv": [sys.executable, "-c", "from pathlib import Path; print(Path('input.txt').read_text())"]}
    if directory != "default":
        arguments["workdir"] = "nested" if directory == "nested" else str(target)
    context = _context(workspace)

    result = create_default_registry().execute("run_command", arguments, context=context)

    assert result.success, result.output
    assert result.structured_output["exit_code"] == 0
    assert result.structured_output["stdout"] == directory + "\n"
    assert context.workspace == workspace


def test_command_schema_and_executor_accept_workdir_without_the_old_parameter(tmp_path: Path) -> None:
    registry = create_default_registry()
    definition = next(tool for tool in registry.definitions() if tool["name"] == "run_command")
    arguments = {"argv": [sys.executable, "--version"], "workdir": str(tmp_path), "timeout_seconds": 10}
    validate(arguments, definition["parameters"])
    arguments["relative_path"] = arguments.pop("workdir")

    with pytest.raises(ValidationError):
        validate(arguments, definition["parameters"])
    result = registry.execute("run_command", arguments, context=_context(tmp_path))
    assert not result.success
    assert "relative_path" in result.output


@pytest.mark.parametrize("arguments", [{}, {"argv": ["unused"], "command": "unused"},
                                       {"argv": None}, {"command": None}, {"argv": [42]}])
def test_command_schema_rejects_invalid_choices_before_execution(tmp_path, arguments):
    registry = create_default_registry()
    definition = next(tool for tool in registry.definitions() if tool["name"] == "run_command")
    with pytest.raises(ValidationError):
        validate(arguments, definition["parameters"])
    result = registry.execute("run_command", arguments, context=_context(tmp_path))
    assert not result.success
    assert "run_command 未执行" in result.output


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
