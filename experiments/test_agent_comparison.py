import json
import subprocess
import sys

import agent_comparison as comparison
import tool_design_benchmark as base


def test_newline_grade_rejects_semantically_correct_but_changed_encoding(tmp_path):
    f = comparison.make_fixture(tmp_path, "newline_matrix", 0)
    before = base.snapshot(f.root)
    for name, expected in f.expected.items():
        (f.root / name).write_bytes(bytes.fromhex(expected))
    assert comparison.grade(f, "newline_matrix", "done", before)["passed"]
    path = f.root / "bom.txt"
    path.write_bytes(path.read_bytes().removeprefix(b"\xef\xbb\xbf"))
    assert not comparison.grade(f, "newline_matrix", "done", before)["passed"]


def test_output_grade_requires_real_execution(tmp_path):
    f = comparison.make_fixture(tmp_path, "output_tail", 0)
    before = base.snapshot(f.root)
    answer = json.dumps(f.expected)
    assert not comparison.grade(f, "output_tail", answer, before)["passed"]
    subprocess.run([sys.executable, "noisy_report.py"], cwd=f.root, check=True, stdout=subprocess.DEVNULL)
    assert comparison.grade(f, "output_tail", answer, before)["passed"]
    with (f.root / "run.log").open("wb") as log:
        subprocess.run([sys.executable, "noisy_report.py"], cwd=f.root, check=True, stdout=log)
    assert comparison.grade(f, "output_tail", answer, before)["auxiliary_output_files"] == ["run.log"]
    base.put(f.root, "unrelated.txt", "unexpected")
    assert not comparison.grade(f, "output_tail", answer, before)["passed"]


def test_workbuddy_stream_deduplicates_usage_and_records_error():
    message = {"id": "r1", "model": "model", "usage": {"input_tokens": 10},
               "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "x"}}]}
    events = [{"type": "assistant", "message": message, "observed_seconds": 1},
              {"type": "assistant", "message": message, "observed_seconds": 1},
              {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1",
                 "content": [{"type": "text", "text": "missing"}], "is_error": True}]}},
              {"type": "result", "result": "answer", "usage": {"input_tokens": 10, "output_tokens": 2}}]
    result = comparison.parse_external("workbuddy", events)
    assert result["model_requests"] == 1
    assert result["tool_calls"] == result["tool_errors"] == 1
    assert result["answer"] == "answer"
    assert result["tools"][0]["output"] == "missing"


def test_codex_counts_completed_calls_once_and_keeps_metadata_warning_separate():
    item = {"id": "tool", "type": "command_execution", "command": "rg x", "status": "failed", "exit_code": 1}
    events = [{"type": "item.started", "item": item},
              {"type": "item.completed", "item": item, "observed_seconds": 3},
              {"type": "item.completed", "item": {"id": "warn", "type": "error", "message": "metadata fallback"}},
              {"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 3}}]
    result = comparison.parse_external("codex", events)
    assert result["tool_calls"] == result["tool_errors"] == 1
    assert result["engine_errors"] == ["metadata fallback"]


def test_reasoning_is_not_persisted():
    assert comparison.sanitize_event({"item": {"type": "reasoning", "text": "private"}}) is None
    event = {"message": {"content": [{"type": "thinking", "text": "private"}, {"type": "text", "text": "public"}]}}
    assert comparison.sanitize_event(event)["message"]["content"] == [{"type": "text", "text": "public"}]
