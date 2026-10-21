import json
from pathlib import Path

import pytest

from praxis_agent.tools import (
    edit_file,
    find_files,
    list_files,
    read_file,
    search_text,
    write_file,
)


def test_read_file_returns_utf8_text(tmp_path: Path) -> None:
    source_file = tmp_path / "notes.txt"
    source_file.write_bytes("你好, Praxis\r\n".encode("utf-8"))

    content = read_file(tmp_path, "notes.txt")

    assert content == {"text": "你好, Praxis\r\n", "truncated": False,
                       "next_position": None, "note": ""}


def test_read_file_returns_the_requested_line_range(tmp_path: Path) -> None:
    source_file = tmp_path / "notes.txt"
    source_file.write_text(
        "one\ntwo\nthree\nfour\n",
        encoding="utf-8", newline="",
    )

    content = read_file(tmp_path, "notes.txt", start_line=2, end_line=3)

    assert content == {"text": "two\nthree\n", "truncated": False,
                       "next_position": None, "note": ""}


def test_read_file_rejects_invalid_ranges(tmp_path: Path) -> None:
    source_file = tmp_path / "notes.txt"
    source_file.write_text("content\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line must be a positive integer"):
        read_file(tmp_path, "notes.txt", end_line=0)
    with pytest.raises(ValueError, match="end_line is required"):
        read_file(tmp_path, "notes.txt", end_offset=1)
    with pytest.raises(ValueError, match="must not precede"):
        read_file(tmp_path, "notes.txt", start_offset=3, end_line=1, end_offset=2)


@pytest.mark.parametrize("text", ["", "one", "one\ntwo\n", "one\r\ntwo\r\n"])
@pytest.mark.parametrize("end_offset", [None, 0, 1000])
def test_read_file_limits_oversized_end_lines_to_eof(tmp_path, text, end_offset):
    (tmp_path / "notes.txt").write_bytes(text.encode("utf-8"))

    result = read_file(tmp_path, "notes.txt", end_line=401, end_offset=end_offset)

    assert result["text"] == text
    assert result["truncated"] is False
    assert result["next_position"] is None
    assert "EOF" in result["note"]


def test_read_file_starting_past_eof_returns_an_explained_empty_result(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"one\ntwo")

    result = read_file(tmp_path, "notes.txt", start_line=30, end_line=40)

    assert result["text"] == ""
    assert result["truncated"] is False
    assert result["next_position"] is None
    assert "Start line is beyond EOF" in result["note"]


@pytest.mark.parametrize("arguments", [
    {"start_line": 0}, {"start_line": True}, {"end_line": -1}, {"end_line": 1.5},
    {"end_line": False}, {"start_offset": -1}, {"end_line": 99, "end_offset": -1},
    {"end_line": 99, "end_offset": True}, {"start_offset": 50},
    {"end_line": 1, "end_offset": 50},
    {"start_line": 30, "end_line": 20},
    {"start_line": 30, "start_offset": 2, "end_line": 30, "end_offset": 1},
])
def test_read_file_still_rejects_invalid_positions_when_clamping(tmp_path, arguments):
    (tmp_path / "notes.txt").write_bytes(b"one\ntwo")
    with pytest.raises(ValueError):
        read_file(tmp_path, "notes.txt", **arguments)


def test_read_file_paginates_an_oversized_range_without_losing_characters(tmp_path):
    text = "中文\\\"\t\r\n" * 120
    (tmp_path / "notes.txt").write_bytes(text.encode("utf-8"))
    position = {"line": 1, "offset": 0}
    chunks = []

    for _ in range(100):
        result = read_file(tmp_path, "notes.txt", start_line=position["line"],
                           start_offset=position["offset"], end_line=1000, character_limit=420)
        assert len(json.dumps(result, ensure_ascii=False)) <= 420
        assert "EOF" in result["note"]
        chunks.append(result["text"])
        if not result["truncated"]:
            assert result["next_position"] is None
            break
        assert result["text"]
        assert result["next_position"] != position
        position = result["next_position"]
    else:
        pytest.fail("Read continuation did not reach EOF")

    assert "".join(chunks) == text


def test_read_file_limits_a_single_long_line(tmp_path: Path) -> None:
    source_file = tmp_path / "notes.txt"
    source_file.write_text("x" * 20_001, encoding="utf-8")

    content = read_file(tmp_path, "notes.txt", character_limit=16000)
    assert len(json.dumps(content, ensure_ascii=False)) <= 16000
    assert content["truncated"] is True
    assert content["note"]
    position = content["next_position"]
    assert position == {"line": 1, "offset": len(content["text"])}
    remainder = read_file(tmp_path, "notes.txt", start_line=position["line"],
                          start_offset=position["offset"], character_limit=16000)
    assert remainder["truncated"] is False
    assert remainder["next_position"] is None
    assert content["text"] + remainder["text"] == "x" * 20_001


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

    result = edit_file(
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
    assert result.path == "config.txt"
    assert result.added == result.removed == 2
    assert "-port=8080" in result.diff and "+port=443" in result.diff


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


def test_write_file_creates_parent_directories(tmp_path: Path) -> None:
    result = write_file(tmp_path, "src/example.py", "value = 1\n")

    created_file = tmp_path / "src" / "example.py"
    assert created_file.read_text(encoding="utf-8") == "value = 1\n"
    assert result.path == "src/example.py"
    assert result.added > 0 and result.removed == 0


def test_write_file_overwrites_an_existing_file(tmp_path: Path) -> None:
    existing_file = tmp_path / "example.py"
    existing_file.write_text("original\n", encoding="utf-8")

    assert write_file(tmp_path, "example.py", "replacement\n").path == "example.py"

    assert existing_file.read_text(encoding="utf-8") == "replacement\n"


def test_write_file_rejects_a_path_outside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="path must stay inside the workspace"):
        write_file(workspace, "../outside.txt", "secret")

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

    assert result == {"entries": [
        {"path": ".venv", "type": "directory"},
        {"path": "nested", "type": "directory"},
        {"path": "package.egg-info", "type": "directory"},
        {"path": "a.py", "type": "file"},
    ]}


def test_list_files_lists_only_the_requested_directory(tmp_path: Path) -> None:
    (tmp_path / "src" / "package").mkdir(parents=True)
    (tmp_path / "src" / "main.py").write_text("main\n", encoding="utf-8")
    (tmp_path / "src" / "package" / "module.py").write_text(
        "module\n",
        encoding="utf-8",
    )

    result = list_files(tmp_path, "src")

    assert result == {"entries": [
        {"path": "src/package", "type": "directory"},
        {"path": "src/main.py", "type": "file"},
    ]}


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

    result = find_files(tmp_path, "*.py", ignore_patterns=[".venv/"])

    assert result["files"] == [
        "src/first.py",
        "src/package/second.py",
    ]


def test_find_files_returns_all_results_without_implicit_exclusions(tmp_path: Path) -> None:
    for file_name in ("c.py", "a.py", "b.py"):
        (tmp_path / file_name).write_text("content\n", encoding="utf-8")

    (tmp_path / ".hidden.py").write_text("content\n", encoding="utf-8")
    result = find_files(tmp_path, "*.py")
    assert result == {"files": [".hidden.py", "a.py", "b.py", "c.py"]}


def test_find_files_requires_ripgrep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "praxis_agent.tools.filesystem.shutil.which",
        lambda _: None,
    )

    with pytest.raises(RuntimeError, match="ripgrep .* is required"):
        find_files(tmp_path, "*.py")


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

    assert result == {"results": [
        "src/first.py:2:needle = value",
        "src/second.py:1:# needle",
    ], "next_offset": None}
