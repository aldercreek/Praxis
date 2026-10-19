"""Summarize saved executions without making additional model requests."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import statistics


# These two compound commands were reviewed against their complete saved output.
# No general shell-parser assumptions are made from their final exit code.
REVIEWED_CATEGORIES = {
    ("r0-multifile_fix-codex_native", 2): "expected_search_no_match",  # rg --files found no AGENTS.md.
    ("r1-output_tail-praxis", 3): "filter_pipeline_returned_no_output",  # Cause not established by the empty stderr.
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def exit_code(tool):
    if tool.get("exit_code") is not None:
        return tool["exit_code"]
    match = re.search(r"(?im)^exit[_ ]code:\s*(-?\d+)\s*$", str(tool.get("output", "")))
    return int(match[1]) if match else None


def classify(tool):
    """Conservative labels for observable errors, not a general command parser."""
    output = str(tool.get("output", ""))
    code = exit_code(tool)
    if tool.get("success") and code in (None, 0):
        return None
    if "failed to connect to git-ai background service" in output:
        return "environment_git_ai_daemon"
    if "add_item" in output and "fatal error" in output:
        return "environment_git_bash_crash"
    if "does not have any commits yet" in output or ("FileNotFoundError" in output and "refs/heads/master" in output):
        return "fixture_has_no_git_commit"
    if "line must be between" in output:
        return "tool_read_end_beyond_eof"
    if "SyntaxError" in output or "Non-hexadecimal digit" in output:
        return "generated_command_syntax"
    if "AssertionError" in output:
        return "generated_verification_assertion"
    if "Start directory is not importable" in output:
        return "generated_test_invocation"
    if "Cannot find path" in output:
        return "generated_path"
    if "不是内部或外部命令" in output:
        return "generated_shell_quoting"
    # Reviewed trials use this deliberately absent symbol to test no-match handling.
    # Exit 1 alone cannot distinguish a no-match from a real command error.
    arguments = json.dumps(tool.get("arguments", tool.get("command", "")), ensure_ascii=False)
    if code == 1 and "PAYMENT_GATEWAY_V9" in arguments and "rg" in arguments:
        if not output.strip() or "stderr:\n<empty>" in output or "Stderr: (empty)" in output:
            return "expected_search_no_match"
    return "unclassified_nonzero_or_tool_error"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("D:/Users/wanjl/Downloads/praxis-agent-comparison-20261018"))
    args = parser.parse_args()
    trials, incidents = [], []
    for path in sorted((args.output / "trials").glob("*/result.json")):
        result = read_json(path)
        events = [json.loads(line) for line in (path.parent / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        engine = result["engine"]
        elapsed = result["elapsed_seconds"]
        if engine.startswith("codex"):
            start = next(e["observed_seconds"] for e in events if e.get("type") == "turn.started")
            elapsed -= start
            timing_basis = "external elapsed minus turn.started observed time"
        elif engine == "workbuddy":
            elapsed = next(e["duration_ms"] / 1000 for e in reversed(events) if e.get("type") == "result")
            timing_basis = "engine result.duration_ms"
        else:
            timing_basis = "agent.run through agent.close; initialization excluded"
        counts = Counter()
        for index, tool in enumerate(result["tools"]):
            category = REVIEWED_CATEGORIES.get((path.parent.name, index)) or classify(tool)
            if category:
                counts[category] += 1
                output = str(tool.get("output", ""))
                diagnostic = [line for line in output.splitlines() if any(word in line.lower() for word in
                    ("error", "fatal", "failed", "cannot", "line must", "不是内部或外部命令"))]
                incidents.append({"trial": path.parent.name, "tool_index": index, "tool": tool["name"],
                    "reported_tool_success": tool.get("success"), "exit_code": exit_code(tool), "category": category,
                    "diagnostic": " | ".join(diagnostic)[-600:], "source": str(path)})
        models = sorted({request["model"] for request in result["requests"] if request.get("model")})
        trials.append({"trial": path.parent.name, "engine": engine, "case": result["case"], "repeat": result["repeat"],
            "passed": result["passed"], "tool_calls": result["tool_calls"],
            "raw_elapsed_seconds": round(result["elapsed_seconds"], 3),
            "normalized_task_seconds": round(elapsed, 3), "timing_basis": timing_basis,
            "reported_tool_error_flags": result["tool_errors"], "reviewed_incidents": dict(counts),
            "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
            "requested_model": result["requested_model"], "returned_models_where_exposed": models,
            "history_reads": sum(t["name"] == "read_history" for t in result["tools"]),
            "max_saved_tool_output_chars": max((len(str(t.get("output", ""))) for t in result["tools"]), default=0)})
    groups, cases = [], []
    for engine in ("praxis", "codex", "workbuddy", "codex_native"):
        group = [r for r in trials if r["engine"] == engine]
        if not group:
            continue
        counts = Counter()
        for trial in group:
            counts.update(trial["reviewed_incidents"])
        groups.append({"engine": engine, "trials": len(group), "passed": sum(r["passed"] for r in group),
            "total_tool_calls": sum(r["tool_calls"] for r in group),
            "mean_tool_calls": round(statistics.mean(r["tool_calls"] for r in group), 2),
            "median_normalized_task_seconds": round(statistics.median(r["normalized_task_seconds"] for r in group), 2),
            "mean_normalized_task_seconds": round(statistics.mean(r["normalized_task_seconds"] for r in group), 2),
            "reviewed_incidents": dict(counts)})
        for case in sorted({r["case"] for r in group}):
            rows = [r for r in group if r["case"] == case]
            cases.append({"engine": engine, "case": case, "trials": len(rows), "passed": sum(r["passed"] for r in rows),
                "mean_tools": round(statistics.mean(r["tool_calls"] for r in rows), 2),
                "mean_task_seconds": round(statistics.mean(r["normalized_task_seconds"] for r in rows), 2)})
    notes = [
        "48 same-alias trials; Codex native reference is a separate group. Small Windows fixture tasks, not a product leaderboard.",
        "Same gateway alias is not independently pinned routing. Returned model exposed by Praxis/WorkBuddy; Codex events omit provider model id.",
        "WorkBuddy is the installed desktop-bundled execution engine with custom model; no desktop UI or default-model claim.",
        "Product prompts, model protocol, permissions and request/turn caps differ. Same task content; absolute per-trial fixture paths differ.",
        "A 1,000,000-token context setting is an experimental assumption, not a capacity/stress test. Summed request tokens are not peak context or billed cost.",
        "Tool call counts measure exposed top-level events; a command can do multiple operations. They are not equal units of work.",
        "Normalized timing reduces startup bias but uses different timing boundaries. Do not infer statistically significant speed rankings from two repeats.",
        "Fresh git repositories have no initial commit or tracked baseline; git history failures are fixture artifacts.",
        "git-ai daemon failures inside Codex sandbox and Git Bash crashes are local environment effects, not engine architecture defects.",
        "Reported tool error flags differ by engine; review exit status and content. Nonzero search-no-match is expected. Multi-command shells can hide earlier exit codes.",
        "Grading checks fixture-root contents and outputs, not all filesystem side effects. Some capture commands used temporary paths outside the fixture.",
        "Exact stdout log captures are accepted consistently; see grading-clarification.json. Preflight auth/model/sandbox failures are excluded from task trials.",
        "No MCP, subagents, long-running-session, cancellation or stale-write concurrency benchmark was performed in these eight cases. Related recommendations come from source review.",
    ]
    analysis = {"notes": notes, "groups": groups, "cases": cases, "trials": trials, "incidents": incidents}
    (args.output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "case-comparison.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    print(json.dumps(groups, ensure_ascii=False, indent=2))
    print("Unclassified incidents:", [(i["trial"], i["tool_index"]) for i in incidents if i["category"].startswith("unclassified")])


if __name__ == "__main__":
    main()
