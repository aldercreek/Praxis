import json
import sys
from types import SimpleNamespace

import pytest
from rich.console import Console

from praxis_agent.agent import Agent
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.planning import RunState
from praxis_agent.sessions import SQLiteSessionStore
from praxis_agent.terminal.cells import ToolHeading, tool_title
from praxis_agent.terminal.views import AgentView, ToolBlock
from praxis_agent.tools import ToolContext, create_default_registry


@pytest.mark.parametrize("ending,code,color", [
    ("exited", 0, "green"), ("exited", 7, "red"),
    ("spawn_failed", None, "red"), ("running", None, "#a0a0a0"),
])
def test_command_result_has_one_persisted_copy_and_restores_the_live_view(tmp_path, ending, code, color):
    arguments = {"argv": [sys.executable, "-u", "-c", f"print('unique-output-marker'); exit({code})"]}
    if ending == "spawn_failed":
        arguments = {"argv": [str(tmp_path / "missing.exe")]}
    elif ending == "running":
        arguments = {"argv": [sys.executable, "-u", "-c", "import time; print('unique-output-marker'); time.sleep(60)"],
                     "yield_time_ms": 1000}
    call = FunctionCall("command", "run_command", json.dumps(arguments))
    requests = []

    def respond(items, tools, **kwargs):
        requests.append(items)
        if len(requests) == 1:
            return ModelResponse(({"type": "function_call", "call_id": call.call_id,
                                   "name": call.name, "arguments": call.arguments},), "", (call,))
        return ModelResponse(({"role": "assistant", "content": "done"},), "done", ())

    store = SQLiteSessionStore(tmp_path / "history.sqlite3")
    session = store.create_session(tmp_path, "test")
    live = AgentView("main", "main")
    agent = Agent(SimpleNamespace(create_response=respond), create_default_registry(), tmp_path,
                  event_sink=SimpleNamespace(emit=live.consume), bypass_approvals=True,
                  session_store=store, session_id=session.id, session_model="test")
    try:
        assert agent.run("execute") == "done"
        history = store.load_conversation_items(session.id)
        saved = next(item for item in history if item.get("type") == "function_call_output")
        result = json.loads(saved["output"])
        assert result["success"] == (ending in {"exited", "running"})
        assert set(result["output"]) == {"session_id", "status", "exit_code", "stdout", "stderr", "error"}
        assert result["output"]["status"] == ending
        if ending == "exited":
            assert result["output"]["exit_code"] == code
        assert saved["output"].count("unique-output-marker") == (0 if ending == "spawn_failed" else 1)
        restored = AgentView("main", "main")
        restored.restore_history(history)
        before = next(block for block in live.committed_blocks if isinstance(block, ToolBlock))
        after = next(block for block in restored.committed_blocks if isinstance(block, ToolBlock))
        assert before == after
        title = tool_title((after,))
        assert title.plain.startswith("Run ")
        assert not title.style
        colored = [segment for segment in Console(width=100).render(ToolHeading((after,)))
                   if segment.text.strip() and segment.style and segment.style.color]
        assert len(colored) == 1 and colored[0].text.strip() == "•"
        assert colored[0].style.color.name == color
        assert after.failed == (ending == "spawn_failed")
    finally:
        agent.close()


def test_search_no_match_keeps_exit_one_without_marking_tool_failed(tmp_path):
    (tmp_path / "input.txt").write_text("hello\n", encoding="utf-8")
    result = create_default_registry().execute("run_command", {
        "argv": ["rg", "NO_MATCH", "input.txt"],
    }, context=ToolContext(tmp_path, RunState("search")))
    assert result.success
    assert result.structured_output["status"] == "exited"
    assert result.structured_output["exit_code"] == 1
    assert result.structured_output["stdout"] == result.structured_output["stderr"] == ""
    assert "The meaning of a nonzero exit code depends on the program" in result.structured_output["error"]
