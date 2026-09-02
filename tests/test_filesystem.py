from pathlib import Path

import pytest

from praxis_agent.tools import (
    create_file,
    list_files,
    read_file,
    replace_text,
    search_text,
)


def test_read_file_returns_utf8_text(tmp_path: Path) -> None:
    source_file = tmp_path / "notes.txt"
    source_file.write_text("hello, Praxis\n", encoding="utf-8")

    content = read_file(tmp_path, "notes.txt")

    assert content == "hello, Praxis\n"


def test_read_file_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="file does not exist"):
        read_file(tmp_path, "missing.txt")


def test_read_file_rejects_a_path_outside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="path must stay inside the workspace"):
        read_file(workspace, "../outside.txt")


def test_replace_text_updates_one_exact_match(tmp_path: Path) -> None:
    source_file = tmp_path / "config.txt"
    source_file.write_text("mode=old\n", encoding="utf-8")

    diff = replace_text(tmp_path, "config.txt", "mode=old", "mode=new")

    assert source_file.read_text(encoding="utf-8") == "mode=new\n"
    assert "--- a/config.txt" in diff
    assert "+++ b/config.txt" in diff
    assert "-mode=old" in diff
    assert "+mode=new" in diff


def test_replace_text_rejects_a_missing_match(tmp_path: Path) -> None:
    source_file = tmp_path / "config.txt"
    source_file.write_text("mode=current\n", encoding="utf-8")

    with pytest.raises(ValueError, match="old_text was not found"):
        replace_text(tmp_path, "config.txt", "mode=old", "mode=new")

    assert source_file.read_text(encoding="utf-8") == "mode=current\n"


def test_replace_text_rejects_multiple_matches(tmp_path: Path) -> None:
    source_file = tmp_path / "config.txt"
    source_file.write_text("mode=old\nmode=old\n", encoding="utf-8")

    with pytest.raises(ValueError, match="found 2 matches"):
        replace_text(tmp_path, "config.txt", "mode=old", "mode=new")

    assert source_file.read_text(encoding="utf-8") == "mode=old\nmode=old\n"


def test_replace_text_rejects_a_path_outside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("mode=old\n", encoding="utf-8")

    with pytest.raises(ValueError, match="path must stay inside the workspace"):
        replace_text(workspace, "../outside.txt", "mode=old", "mode=new")

    assert outside_file.read_text(encoding="utf-8") == "mode=old\n"


def test_create_file_creates_parent_directories(tmp_path: Path) -> None:
    diff = create_file(tmp_path, "src/example.py", "value = 1\n")

    created_file = tmp_path / "src" / "example.py"
    assert created_file.read_text(encoding="utf-8") == "value = 1\n"
    assert "--- /dev/null" in diff
    assert "+++ b/src/example.py" in diff
    assert "+value = 1" in diff


def test_create_file_does_not_overwrite_an_existing_file(tmp_path: Path) -> None:
    existing_file = tmp_path / "example.py"
    existing_file.write_text("original\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="file already exists"):
        create_file(tmp_path, "example.py", "replacement\n")

    assert existing_file.read_text(encoding="utf-8") == "original\n"


def test_create_file_rejects_a_path_outside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="path must stay inside the workspace"):
        create_file(workspace, "../outside.txt", "secret")

    assert not (tmp_path / "outside.txt").exists()


def test_list_files_returns_sorted_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.py").write_text("b\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("ignored\n", encoding="utf-8")

    result = list_files(tmp_path)

    assert result.splitlines() == ["a.py", "nested/b.py"]


def test_search_text_returns_path_line_and_content(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "first.py").write_text(
        "value = 1\nneedle = value\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "second.py").write_text(
        "# needle\n",
        encoding="utf-8",
    )

    result = search_text(tmp_path, "needle", "src")

    assert result.splitlines() == [
        "src/first.py:2:needle = value",
        "src/second.py:1:# needle",
    ]
