"""Offline checks that benchmark grading detects incorrect task outcomes."""

import json

import pytest

from tool_design_benchmark import CASES, fixture, grade, put, snapshot


def solve(f, case):
    if case == "precise_edit":
        for name, text in f.expected.items():
            put(f.root, name, text)
        return "done"
    if case == "multifile_fix":
        put(f.root, "pricing.py", "def total_price(unit_price, quantity, shipping):\n    return unit_price * quantity + shipping\n")
        put(f.root, "invoice.py", "def invoice(items, shipping):\n    return sum(p*q for p,q in items) + shipping if items else 0\n")
        return "done"
    return json.dumps(f.expected, ensure_ascii=False)


@pytest.mark.parametrize("case", CASES)
def test_accepts_correct_result_and_rejects_unrelated_file_edit(tmp_path, case):
    f = fixture(tmp_path, case, 0)
    before = snapshot(f.root)
    answer = solve(f, case)
    assert grade(f, case, answer, before)["passed"]
    put(f.root, "unrelated.txt", "unexpected change")
    result = grade(f, case, answer, before)
    assert not result["passed"]
    assert result["unexpected_changes"] == ["unrelated.txt"]


@pytest.mark.parametrize("case", CASES)
def test_claiming_success_is_not_evidence_of_correctness(tmp_path, case):
    f = fixture(tmp_path, case, 1)
    result = grade(f, case, "已完成，所有结果都正确。", snapshot(f.root))
    assert not result["passed"]


def test_exact_edit_rejects_newline_conversion_even_when_text_matches(tmp_path):
    f = fixture(tmp_path, "precise_edit", 0)
    before = snapshot(f.root)
    solve(f, "precise_edit")
    path = f.root / "src/连接 设置.py"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    result = grade(f, "precise_edit", "done", before)
    assert not result["passed"]
    assert not result["checks"]["src/连接 设置.py"]


def test_hidden_boundary_checks_reject_fix_that_only_passes_public_examples(tmp_path):
    f = fixture(tmp_path, "multifile_fix", 0)
    before = snapshot(f.root)
    solve(f, "multifile_fix")
    put(f.root, "invoice.py", "def invoice(items, shipping):\n    return sum(p*q for p,q in items) + shipping\n")
    result = grade(f, "multifile_fix", "tests passed", before)
    assert not result["passed"]
    assert not result["checks"]["independent_behavior_checks"]


def test_path_separator_and_json_fences_do_not_create_false_failures(tmp_path):
    f = fixture(tmp_path, "search_noise", 0)
    answer = {**f.expected, "definition_path": f.expected["definition_path"].replace("/", "\\"),
              "reference_paths": [p.replace("/", "\\") for p in f.expected["reference_paths"]]}
    assert grade(f, "search_noise", "```json\n" + json.dumps(answer) + "\n```", snapshot(f.root))["passed"]
