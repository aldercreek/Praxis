"""Recovery guidance must describe the actual effects of a failed edit."""

import pytest

from praxis_agent.planning import RunState
from praxis_agent.tools import ToolContext, create_default_registry
from praxis_agent.tools.text_file import TextFile


def execute(root, edits, path="config.txt"):
    return create_default_registry().execute(
        "edit_file", {"relative_path": path, "edits": edits},
        context=ToolContext(root, RunState("edit recovery")),
    )


@pytest.mark.parametrize("second, expected, hint", [
    ({"old_text": "missing", "new_text": "new"}, "old_text was not found", "read_file"),
    ({"old_text": "repeat", "new_text": "new"}, "found 2 matches", "enough surrounding context"),
    ({"old_text": "repeat", "new_text": "repeat"}, "new_text must differ", "Remove this unchanged"),
    ({"old_text": "", "new_text": "insert"}, "old_text must not be empty", "unique existing text"),
])
def test_later_edit_failure_explains_recovery_and_never_saves(tmp_path, monkeypatch, second, expected, hint):
    path = tmp_path / "config.txt"
    original = b"first\r\nrepeat\r\nrepeat\r\n"
    path.write_bytes(original)
    monkeypatch.setattr(TextFile, "write", lambda *args: pytest.fail("validation failure must not save"))
    result = execute(tmp_path, [{"old_text": "first", "new_text": "changed"}, second])
    assert not result.success
    assert "config.txt" in result.output
    assert "Edit 2/2 failed" in result.output
    assert "No changes from this call were written (including earlier edits)" in result.output
    assert expected in result.output and hint in result.output
    assert path.read_bytes() == original


def test_not_found_after_prior_replacement_explains_sequential_matching(tmp_path):
    (tmp_path / "config.txt").write_text("old", encoding="utf-8")
    result = execute(tmp_path, [
        {"old_text": "old", "new_text": "middle"},
        {"old_text": "old", "new_text": "final"},
    ])
    assert not result.success and "the content after earlier replacements" in result.output
    # Apply the advised correction; both replacements commit together.
    recovered = execute(tmp_path, [
        {"old_text": "old", "new_text": "middle"},
        {"old_text": "middle", "new_text": "final"},
    ])
    assert recovered.success
    assert (tmp_path / "config.txt").read_text(encoding="utf-8") == "final"


def test_edits_that_cancel_each_other_explain_no_change(tmp_path):
    (tmp_path / "config.txt").write_bytes(b"old")
    result = execute(tmp_path, [
        {"old_text": "old", "new_text": "new"},
        {"old_text": "new", "new_text": "old"},
    ])
    assert not result.success and "cancel each other out" in result.output
    assert "No changes from this call were written" in result.output
    assert (tmp_path / "config.txt").read_bytes() == b"old"


def test_invalid_value_identifies_item_and_field_without_saving(tmp_path):
    (tmp_path / "config.txt").write_bytes(b"old")
    result = execute(tmp_path, [
        {"old_text": "old", "new_text": "new"},
        {"old_text": "new", "new_text": None},
    ])
    assert not result.success
    assert "Edit 2/2 failed" in result.output
    assert "old_text and new_text must be strings (got str and NoneType)" in result.output
    assert "No changes from this call were written" in result.output
    assert (tmp_path / "config.txt").read_bytes() == b"old"


@pytest.mark.parametrize("path, hint", [
    ("missing.txt", "write_file"), ("folder", "not a directory"), ("../outside.txt", "inside the workspace"),
])
def test_bad_paths_have_actionable_guidance(tmp_path, path, hint):
    (tmp_path / "folder").mkdir()
    result = execute(tmp_path, [{"old_text": "old", "new_text": "new"}], path)
    assert not result.success
    assert path in result.output and hint in result.output
    assert "No changes from this call were written" in result.output


def test_read_permission_failure_reports_no_writes(tmp_path, monkeypatch):
    path = tmp_path / "config.txt"
    path.write_bytes(b"old")

    def fail(cls, path):
        raise PermissionError("read denied")

    monkeypatch.setattr(TextFile, "read", classmethod(fail))
    result = execute(tmp_path, [{"old_text": "old", "new_text": "new"}])
    assert not result.success and "read permissions" in result.output
    assert "No changes from this call were written" in result.output
    assert path.read_bytes() == b"old"


def test_partial_write_error_does_not_claim_rollback(tmp_path, monkeypatch):
    path = tmp_path / "config.txt"
    path.write_bytes(b"old")

    def partial_write(self, path, content):
        path.write_bytes(b"n")
        raise OSError("disk full")

    monkeypatch.setattr(TextFile, "write", partial_write)
    result = execute(tmp_path, [{"old_text": "old", "new_text": "new"}])
    assert not result.success and "disk full" in result.output
    assert "the file may have been partially modified" in result.output
    assert "read_file" in result.output
    assert "No changes from this call were written" not in result.output
    assert path.read_bytes() == b"n"
