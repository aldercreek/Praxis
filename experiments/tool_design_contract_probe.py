"""Measure integration differences without invoking a model."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from praxis_agent.execution import Run
from praxis_agent.model import FunctionCall
from praxis_agent.planning import RunState
from praxis_agent.terminal.views import AgentView, ToolBlock
from praxis_agent.tool_execution import ToolCallExecutor
from praxis_agent.tools.code import CodeRuntime
from praxis_agent.tools.registry import ToolContext, create_default_registry


def measure(root, name, arguments):
    state = RunState("tool contract probe")
    registry = create_default_registry()
    code = CodeRuntime()
    registry.register(code.tool("parent"))
    view = AgentView("main", "main")
    approvals = []
    executor = ToolCallExecutor(registry, ToolContext(root, state), view.consume,
                               approve_tool_call=lambda *a: approvals.append(a[0]) or True)
    call = FunctionCall("probe", name, json.dumps(arguments))
    try:
        result = Run().execute(lambda: executor.execute("probe-turn", call))
    finally:
        code.close()
    block = next(b for b in view.committed_blocks if isinstance(b, ToolBlock))
    return {"tool": name, "success": result.success, "output": result.output,
            "structured_output": result.structured_output,
            "has_structured_output": result.structured_output is not None,
            "has_file_diff": result.file_change is not None,
            "recorded_changed_files": sorted(state.changed_files),
            "ui_failed": block.failed, "ui_exploration": block.exploration,
            "approval_requests": len(approvals), "parallel_safe": registry.is_parallel_safe(name)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="praxis-tool-contracts-"))
    (root / "sample.txt").write_bytes(b"hello\n")
    rows = {}
    rows["dedicated_empty_search"] = measure(root, "search_text", {"query": "NO_MATCH", "relative_path": "sample.txt"})
    rows["read_past_end"] = measure(root, "read_file", {"relative_path": "sample.txt", "end_line": 50})
    rows["command_empty_search"] = measure(root, "run_command", {"argv": ["rg", "-n", "NO_MATCH", "sample.txt"],
                                                                  "workdir": ".", "timeout_seconds": 60})
    rows["dedicated_edit"] = measure(root, "edit_file", {"relative_path": "sample.txt",
                                                         "edits": [{"old_text": "hello", "new_text": "world"}]})
    rows["dedicated_edit"]["original_line_ending"] = "LF"
    rows["dedicated_edit"]["preserved_line_ending"] = (root / "sample.txt").read_bytes() == b"world\n"
    rows["command_edit"] = measure(root, "run_command", {"argv": [sys.executable, "-X", "utf8", "-c",
        "from pathlib import Path; Path('sample.txt').write_text('changed\\n',encoding='utf-8')"],
        "workdir": ".", "timeout_seconds": 60})
    command = {"argv": [sys.executable, "-c", "print(1)"], "workdir": ".", "timeout_seconds": 60}
    rows["run_code_command_bridge"] = measure(root, "run_code", {
        "session_id": "probe", "code": "call_tool('run_command', " + repr(command) + ")"})
    # Verify observations arose from completed real operations, rather than setup failures.
    assert rows["dedicated_empty_search"]["success"]
    assert rows["command_empty_search"]["structured_output"]["exit_code"] == 1
    assert not rows["command_empty_search"]["ui_failed"]
    assert rows["read_past_end"]["success"]
    assert rows["dedicated_edit"]["preserved_line_ending"]
    assert rows["dedicated_edit"]["success"] and rows["command_edit"]["success"]
    assert (root / "sample.txt").read_text(encoding="utf-8") == "changed\n"
    assert "Tool unavailable for programmatic calls: run_command" in rows["run_code_command_bridge"]["output"]
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
