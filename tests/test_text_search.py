import json
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from praxis_agent.context.capacity import ContextPolicy
from praxis_agent.execution import ExecutionStopped, Run
from praxis_agent.planning import RunState
from praxis_agent.tools import ToolContext, create_default_registry, search_text
from praxis_agent.tools import text_search


def write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode("utf-8"))


@pytest.mark.parametrize("mode", ["content", "files", "count"])
def test_modes_page_in_path_order_and_count_lines_not_occurrences(tmp_path, mode):
    for name in ("c.txt", "a.txt", "b.txt"):
        write(tmp_path, name, "hit hit\nnone\nhit\n")
    actual, offset = [], 0
    while True:
        page = search_text(tmp_path, "hit", mode=mode, limit=2, offset=offset)
        assert set(page) == {"results", "next_offset"}
        actual.extend(page["results"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    expected = {
        "content": [f"{name}:{line}:{text}" for name in ("a.txt", "b.txt", "c.txt")
                    for line, text in ((1, "hit hit"), (3, "hit"))],
        "files": ["a.txt", "b.txt", "c.txt"],
        "count": [{"path": name, "count": 2} for name in ("a.txt", "b.txt", "c.txt")],
    }
    assert actual == expected[mode]
    assert search_text(tmp_path, "hit", mode=mode, offset=99) == {"results": [], "next_offset": None}
    assert search_text(tmp_path, "absent", mode=mode) == {"results": [], "next_offset": None}


def test_context_is_merged_and_does_not_advance_match_offset(tmp_path):
    write(tmp_path, "a.txt", "before\nhit one\nhit two\nafter\ngap\nhit three\nend\n")
    first = search_text(tmp_path, "hit", context_lines=1, limit=1)
    assert first == {"results": ["a.txt-1-before", "a.txt:2:hit one", "a.txt-3-hit two"], "next_offset": 1}
    second = search_text(tmp_path, "hit", context_lines=1, limit=1, offset=1)
    assert second == {"results": ["a.txt-2-hit one", "a.txt:3:hit two", "a.txt-4-after"], "next_offset": 2}
    merged = search_text(tmp_path, "hit", context_lines=1, limit=2)
    assert merged == {"results": ["a.txt-1-before", "a.txt:2:hit one", "a.txt:3:hit two", "a.txt-4-after"], "next_offset": 2}
    assert search_text(tmp_path, "hit", context_lines=1, limit=1, offset=2) == {
        "results": ["a.txt-5-gap", "a.txt:6:hit three", "a.txt-7-end"], "next_offset": None}


def test_context_at_eof_gaps_and_file_boundaries(tmp_path):
    write(tmp_path, "a.txt", "hit\n" + "gap\n" * 7 + "hit")
    write(tmp_path, "b.txt", "hit")
    assert search_text(tmp_path, "hit", context_lines=1) == {
        "results": ["a.txt:1:hit", "a.txt-2-gap", "--", "a.txt-8-gap", "a.txt:9:hit", "--", "b.txt:1:hit"],
        "next_offset": None,
    }


@pytest.mark.parametrize("mode", ["content", "files", "count"])
def test_literal_regex_case_unicode_and_filters(tmp_path, mode):
    write(tmp_path, "src/例子.py", "USER.NAME\nuserXname\n保存用户(用户)\n")
    write(tmp_path, "src/excluded.py", "user.name\n")
    write(tmp_path, "src/other.txt", "user.name\n")
    args = dict(relative_path="src", file_pattern="*.py", ignore_patterns=["excluded.py"], mode=mode)
    assert search_text(tmp_path, "user.name", **args)["results"] == []
    assert len(search_text(tmp_path, "user.name", case_sensitive=False, **args)["results"]) == 1
    assert len(search_text(tmp_path, r"保存用户\(", regex=True, **args)["results"]) == 1
    page = search_text(tmp_path, r"user.name", regex=True, case_sensitive=False, **args)
    if mode == "count":
        assert page["results"] == [{"path": "src/例子.py", "count": 2}]
    elif mode == "content":
        assert len(page["results"]) == 2
    else:
        assert page["results"] == ["src/例子.py"]


@pytest.mark.parametrize("pattern", ["(", "(?<=x)y", r"(x)\1"])
def test_invalid_or_unsupported_regex_has_recovery_hint(tmp_path, pattern):
    write(tmp_path, "a.txt", "xxxy")
    with pytest.raises(RuntimeError, match="Fix the regular expression or use regex=false"):
        search_text(tmp_path, pattern, regex=True)


@pytest.mark.parametrize("kwargs", [
    {"mode": "unknown"}, {"regex": "true"}, {"case_sensitive": 1}, {"context_lines": -1},
    {"context_lines": True}, {"offset": -1}, {"offset": 0.5}, {"limit": 0}, {"limit": True},
    {"query": ""}, {"query": "a\nb"},
])
def test_invalid_arguments(tmp_path, kwargs):
    with pytest.raises((ValueError, TypeError)):
        search_text(tmp_path, **{"query": "hit", **kwargs})


def test_search_includes_hidden_and_gitignored_files_and_handles_single_file(tmp_path):
    write(tmp_path, ".gitignore", "ignored.txt\n")
    write(tmp_path, "ignored.txt", "hit")
    write(tmp_path, ".hidden.txt", "hit")
    assert search_text(tmp_path, "hit", mode="files")["results"] == [".hidden.txt", "ignored.txt"]
    assert search_text(tmp_path, "hit", relative_path="ignored.txt")["results"] == ["ignored.txt:1:hit"]
    assert search_text(tmp_path, "hit", relative_path="ignored.txt", mode="files")["results"] == ["ignored.txt"]
    assert search_text(tmp_path, "hit", relative_path="ignored.txt", mode="count")["results"] == [{"path": "ignored.txt", "count": 1}]


@pytest.mark.parametrize("mode", ["content", "files", "count"])
def test_character_budget_pages_without_dropping_records(tmp_path, mode):
    for index in range(9):
        write(tmp_path, f"file-{index}.txt", "hit\n")
    budget = 125
    actual, offset = [], 0
    while True:
        page = search_text(tmp_path, "hit", mode=mode, limit=50, offset=offset, character_limit=budget)
        assert len(json.dumps(page, ensure_ascii=False)) <= budget
        assert page["results"]
        actual.extend(page["results"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] == offset + len(page["results"])
        offset = page["next_offset"]
    assert len(actual) == 9
    assert actual == search_text(tmp_path, "hit", mode=mode)["results"]


def test_budget_with_context_advances_only_selected_matches(tmp_path):
    write(tmp_path, "a.txt", "\n".join(f"hit {i}" if i % 3 == 0 else "context" for i in range(30)))
    offset, selected = 0, []
    while True:
        page = search_text(tmp_path, "hit", context_lines=1, offset=offset, character_limit=220)
        assert len(json.dumps(page, ensure_ascii=False)) <= 220
        matches = [row for row in page["results"] if row.startswith("a.txt:")]
        selected.extend(matches)
        if page["next_offset"] is None:
            break
        assert page["next_offset"] == offset + len(matches)
        offset = page["next_offset"]
    assert selected == [f"a.txt:{i + 1}:hit {i}" for i in range(0, 30, 3)]


@pytest.mark.parametrize("budget", [130, 240])
def test_long_line_preview_preserves_match_and_progress(tmp_path, budget):
    write(tmp_path, "a.txt", "中文" * 50_000 + "needle" + "尾" * 50_000 + "\nneedle\n")
    page = search_text(tmp_path, "needle", character_limit=budget)
    assert len(json.dumps(page, ensure_ascii=False)) <= budget
    assert "needle" in page["results"][0]
    assert "preview; use read_file" in page["results"][0]
    assert page["next_offset"] == 1
    assert search_text(tmp_path, "needle", offset=1)["results"] == ["a.txt:2:needle"]


def test_escaping_is_included_in_output_budget_and_small_budget_fails(tmp_path):
    write(tmp_path, "a.txt", 'hit "\\\t' * 300)
    page = search_text(tmp_path, "hit", character_limit=180)
    assert len(json.dumps(page, ensure_ascii=False)) <= 180
    assert "preview" in page["results"][0]
    with pytest.raises(ValueError, match="budget"):
        search_text(tmp_path, "hit", character_limit=20)


@pytest.mark.parametrize("programmatic", [False, True])
def test_registry_exposes_new_schema_and_applies_policy_budget(tmp_path, programmatic):
    write(tmp_path, "a.txt", "hit\n" * 100)
    registry = create_default_registry(ContextPolicy(max_tool_output_chars=200))
    definition = next(tool for tool in registry.definitions() if tool["name"] == "search_text")
    assert set(definition["parameters"]["properties"]) == {
        "query", "relative_path", "file_pattern", "ignore_patterns", "mode", "regex",
        "case_sensitive", "context_lines", "limit", "offset",
    }
    result = registry.execute("search_text", {"query": "hit", "regex": True, "limit": 100},
                              context=ToolContext(tmp_path, RunState("search"), programmatic_call=programmatic))
    assert result.success, result.output
    assert len(result.output) <= 200
    assert json.loads(result.output) == result.structured_output
    assert result.structured_output["next_offset"] > 0


def fake_search_process(monkeypatch, script):
    original = text_search.subprocess.Popen
    started, processes = Event(), []

    def spawn(*args, **kwargs):
        process = original([sys.executable, "-u", "-c", script], **kwargs)
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(text_search.subprocess, "Popen", spawn)
    return started, processes


def test_cancel_wakes_blocked_search_reader_and_reaps_process(tmp_path, monkeypatch):
    started, processes = fake_search_process(monkeypatch, "import time; time.sleep(30)")
    scope = Run()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(scope.execute, lambda: search_text(tmp_path, "hit"))
        try:
            assert started.wait(3)
            scope.cancel()
            with pytest.raises(ExecutionStopped):
                future.result(timeout=3)
        finally:
            scope.cancel()
    assert processes[0].poll() is not None
    assert processes[0].stdout.closed
    assert not scope._callbacks


def test_page_boundary_stops_remaining_scan(tmp_path, monkeypatch):
    _, processes = fake_search_process(monkeypatch,
        "import sys,time; sys.stdout.buffer.write(b'a.txt\\0b.txt\\0'); sys.stdout.flush(); time.sleep(30)")
    page = search_text(tmp_path, "hit", mode="files", limit=1)
    assert page == {"results": ["a.txt"], "next_offset": 1}
    assert processes[0].poll() is not None
    assert processes[0].stdout.closed


def test_search_failure_does_not_masquerade_as_empty_success(tmp_path, monkeypatch):
    fake_search_process(monkeypatch, "import sys; sys.exit(2)")
    with pytest.raises(RuntimeError, match="Search failed"):
        search_text(tmp_path, "hit")


def test_existing_search_timeout_still_cleans_up_process(tmp_path, monkeypatch):
    from praxis_agent.tools import filesystem

    monkeypatch.setattr(filesystem, "_SEARCH_TIMEOUT_SECONDS", 0.1)
    _, processes = fake_search_process(monkeypatch, "import time; time.sleep(30)")
    with pytest.raises(TimeoutError, match="Search timed out"):
        search_text(tmp_path, "hit")
    assert processes[0].poll() is not None
    assert processes[0].stdout.closed
