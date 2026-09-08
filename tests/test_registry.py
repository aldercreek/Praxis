from pathlib import Path

from praxis_agent.tools import create_default_registry


def test_default_registry_contains_all_filesystem_tools(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

    assert registry.names == (
        "read_file",
        "edit_file",
        "create_file",
        "list_files",
        "search_text",
        "run_command",
    )
    assert registry.tools[0].description == (
        "Read a bounded line range from a UTF-8 workspace file."
    )
    assert registry.tools[-1].requires_approval is True


def test_registry_returns_model_tool_definitions(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

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


def test_registry_returns_new_parameter_schema_copies(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

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
    registry = create_default_registry(tmp_path)

    result = registry.execute("read_file", {"relative_path": "notes.txt"})

    assert result.success is True
    assert result.output == "[lines 1-1 of 1]\nhello\n"


def test_registry_returns_an_error_for_unknown_tools(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

    result = registry.execute("missing_tool", {})

    assert result.success is False
    assert result.output == "unknown tool: missing_tool"


def test_registry_returns_an_error_for_invalid_arguments(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

    result = registry.execute("read_file", {})

    assert result.success is False
    assert result.output.startswith("TypeError:")
