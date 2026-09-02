from pathlib import Path

from praxis_agent.tools import create_default_registry


def test_default_registry_contains_all_filesystem_tools(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

    assert registry.names == (
        "read_file",
        "replace_text",
        "create_file",
        "list_files",
        "search_text",
    )
    assert registry.tools[0].description == "Read a UTF-8 workspace file."


def test_registry_executes_a_tool_with_its_workspace(tmp_path: Path) -> None:
    source_file = tmp_path / "notes.txt"
    source_file.write_text("hello\n", encoding="utf-8")
    registry = create_default_registry(tmp_path)

    result = registry.execute("read_file", {"relative_path": "notes.txt"})

    assert result.success is True
    assert result.output == "hello\n"


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
