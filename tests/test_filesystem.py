from pathlib import Path

import pytest

from praxis_agent.tools import (
    create_file,
    edit_file,
    find_files,
    list_files,
    read_file,
    search_text,
)


def test_read_file_returns_utf8_text(tmp_path: Path) -> None:
    source_file = tmp_path / "notes.txt"
    source_file.write_text("hello, Praxis\n", encoding="utf-8")

    content = read_file(tmp_path, "notes.txt")

    assert content == "[lines 1-1 of 1]\nhello, Praxis\n"


def test_read_file_returns_the_requested_line_range(tmp_path: Path) -> None:
    source_file = tmp_path / "notes.txt"
    source_file.write_text(
        "one\ntwo\nthree\nfour\n",
        encoding="utf-8",
    )

    content = read_file(tmp_path, "notes.txt", start_line=2, max_lines=2)

    assert content == (
        "[lines 2-3 of 4; next_start_line=4]\ntwo\nthree\n"
    )


def test_read_file_limits_the_number_of_requested_lines(tmp_path: Path) -> None:
    source_file = tmp_path / "notes.txt"
    source_file.write_text("content\n", encoding="utf-8")

    with pytest.raises(ValueError, match="max_lines must not exceed 400"):
        read_file(tmp_path, "notes.txt", max_lines=401)


def test_read_file_limits_a_single_long_line(tmp_path: Path) -> None:
    source_file = tmp_path / "notes.txt"
    source_file.write_text("x" * 20_001, encoding="utf-8")

    content = read_file(tmp_path, "notes.txt")
    header, visible_content = content.split("\n", 1)

    assert header == (
        "[lines 1-1 of 1; line 1 truncated at 20000 characters]"
    )
    assert visible_content == "x" * 20_000


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


def test_edit_file_applies_multiple_replacements(tmp_path: Path) -> None:
    source_file = tmp_path / "config.txt"
    source_file.write_text(
        "host=localhost\nport=8080\n",
        encoding="utf-8",
    )

    diff = edit_file(
        tmp_path,
        "config.txt",
        [
            {"old_text": "host=localhost", "new_text": "host=example.com"},
            {"old_text": "port=8080", "new_text": "port=443"},
        ],
    )

    assert source_file.read_text(encoding="utf-8") == (
        "host=example.com\nport=443\n"
    )
    assert "-host=localhost" in diff
    assert "+host=example.com" in diff
    assert "-port=8080" in diff
    assert "+port=443" in diff


def test_edit_file_leaves_file_unchanged_when_a_later_edit_fails(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "config.txt"
    original_content = "host=localhost\nport=8080\n"
    source_file.write_text(original_content, encoding="utf-8")

    with pytest.raises(ValueError, match="edit 2 old_text was not found"):
        edit_file(
            tmp_path,
            "config.txt",
            [
                {
                    "old_text": "host=localhost",
                    "new_text": "host=example.com",
                },
                {"old_text": "port=9999", "new_text": "port=443"},
            ],
        )

    assert source_file.read_text(encoding="utf-8") == original_content


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
    (tmp_path / "package.egg-info").mkdir()
    (tmp_path / "package.egg-info" / "PKG-INFO").write_text(
        "ignored\n",
        encoding="utf-8",
    )

    result = list_files(tmp_path)

    assert result.splitlines() == ["a.py", "nested/", "nested/b.py"]


def test_list_files_respects_max_depth(tmp_path: Path) -> None:
    (tmp_path / "src" / "package").mkdir(parents=True)
    (tmp_path / "src" / "main.py").write_text("main\n", encoding="utf-8")
    (tmp_path / "src" / "package" / "module.py").write_text(
        "module\n",
        encoding="utf-8",
    )

    result = list_files(tmp_path, max_depth=1)

    assert result.splitlines() == ["src/"]


def test_find_files_returns_sorted_matching_paths(tmp_path: Path) -> None:
    (tmp_path / "src" / "package").mkdir(parents=True)
    (tmp_path / "src" / "package" / "second.py").write_text(
        "second\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "first.py").write_text("first\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text(
        "ignored\n",
        encoding="utf-8",
    )

    result = find_files(tmp_path, "*.py")

    assert result.splitlines() == [
        "src/first.py",
        "src/package/second.py",
    ]


def test_find_files_reports_truncated_results(tmp_path: Path) -> None:
    for file_name in ("c.py", "a.py", "b.py"):
        (tmp_path / file_name).write_text("content\n", encoding="utf-8")

    result = find_files(tmp_path, "*.py", max_results=2)

    assert result.splitlines() == [
        "a.py",
        "b.py",
        "... truncated after 2 files",
    ]


def test_find_files_falls_back_to_python_without_ripgrep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text(
        "content\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "praxis_agent.tools.filesystem.shutil.which",
        lambda _: None,
    )

    result = find_files(tmp_path, "*.py")

    assert result == "src/module.py"


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
    (tmp_path / "src" / "ignored.txt").write_text(
        "needle in text\n",
        encoding="utf-8",
    )

    result = search_text(
        tmp_path,
        "needle",
        "src",
        file_pattern="*.py",
    )

    assert result.splitlines() == [
        "src/first.py:2:needle = value",
        "src/second.py:1:# needle",
    ]
