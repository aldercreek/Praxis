from pathlib import Path

import pytest

from praxis_agent.instructions import load_global_instructions


@pytest.fixture
def global_file(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    directory = tmp_path / ".praxis"
    directory.mkdir()
    return directory / "AGENTS.md"


def test_missing_and_empty_global_instructions(global_file):
    assert load_global_instructions() == ""
    global_file.write_text("\n \t", encoding="utf-8")
    assert load_global_instructions() == ""


def test_global_instructions_preserve_body_and_reload(global_file):
    body = "# 全局约定\n\n使用中文。\n" + "Long rule.\n" * 4000
    global_file.write_text(body, encoding="utf-8-sig")
    result = load_global_instructions()
    assert body in result
    assert "\ufeff" not in result
    assert "AGENTS.md" in result
    global_file.write_text("Updated rule", encoding="utf-8")
    assert "Updated rule" in load_global_instructions()
    assert body not in load_global_instructions()


def test_invalid_global_file_reports_source(global_file):
    global_file.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="Cannot read global instructions.*AGENTS.md"):
        load_global_instructions()


def test_unreadable_global_file_is_not_silently_ignored(global_file):
    global_file.mkdir()
    with pytest.raises(ValueError, match="Cannot read global instructions.*AGENTS.md"):
        load_global_instructions()


def test_factory_keeps_global_rules_for_parent_worker_and_resume(global_file, tmp_path, monkeypatch):
    from praxis_agent.factory import create_default_agent
    from praxis_agent.mcp import MCPConnections
    from praxis_agent.model import ModelResponse
    from praxis_agent.multi_agent.models import AgentStatus

    created = []

    class Client:
        def __init__(self, config, instructions, **kwargs):
            self.instructions = instructions
            created.append(self)

        def close(self):
            pass

        def create_response(self, *args, **kwargs):
            return ModelResponse(({"role": "assistant", "content": "done"},), "done", ())

    monkeypatch.setattr("praxis_agent.factory.OpenAIModelClient", Client)
    monkeypatch.setattr("praxis_agent.factory.discover_skills", lambda workspace: ())
    global_file.write_text("GLOBAL_SENTINEL", encoding="utf-8")
    workspace = tmp_path / "unrelated-start"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("DO_NOT_AUTOLOAD", encoding="utf-8")
    worker = tmp_path / "actual-target"
    worker.mkdir()
    mcp = MCPConnections(tmp_path / "missing.json")
    mcp.start()
    options = dict(session_database_path=tmp_path / "sessions.sqlite", mcp=mcp)
    agent = create_default_agent(workspace, **options)
    session_id = agent.session_id
    try:
        assert "GLOBAL_SENTINEL" in created[0].instructions
        assert "DO_NOT_AUTOLOAD" not in created[0].instructions
        assert agent.context_manager._instructions == created[0].instructions
        agent.context_manager.reset()
        assert "GLOBAL_SENTINEL" in agent.context_manager._instructions
        runtime = agent.collaboration
        spawned = runtime.spawn("Say done", working_directory=str(worker))
        with runtime.condition:
            assert runtime.condition.wait_for(
                lambda: runtime.handles[spawned["agent_id"]].status == AgentStatus.IDLE,
                timeout=10,
            )
        assert "GLOBAL_SENTINEL" in created[2].instructions
        assert "DO_NOT_AUTOLOAD" not in created[2].instructions
        assert "GLOBAL_SENTINEL" not in created[1].instructions  # summarizer
    finally:
        agent.close()
    global_file.write_text("UPDATED_GLOBAL", encoding="utf-8")
    resumed = create_default_agent(workspace, resume_session_id=session_id, **options)
    try:
        assert "UPDATED_GLOBAL" in resumed.context_manager._instructions
        assert "GLOBAL_SENTINEL" not in resumed.context_manager._instructions
    finally:
        resumed.close()
        mcp.close()
