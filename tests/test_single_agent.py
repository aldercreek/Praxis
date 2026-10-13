from praxis_agent.agent import create_default_agent
from praxis_agent.config import ModelConfig
from praxis_agent.model import ModelResponse
from praxis_agent.multi_agent.tools import PARENT_INSTRUCTIONS
from praxis_agent.prompts import BASE_INSTRUCTIONS


def test_single_agent_removes_collaboration_but_preserves_tools_and_resume(tmp_path, monkeypatch):
    class Client:
        def __init__(self, config, instructions, **kwargs):
            self.instructions = instructions
            self.inputs = []

        def close(self):
            pass

        def create_response(self, input_items, tools, **kwargs):
            self.inputs.append(str(input_items))
            return ModelResponse(({"role": "assistant", "content": "done"},), "done", ())

    monkeypatch.setattr("praxis_agent.agent.OpenAIModelClient", Client)
    monkeypatch.setattr(ModelConfig, "from_env", classmethod(
        lambda cls, **kwargs: ModelConfig(api_key="test", model="test")))
    monkeypatch.setattr("praxis_agent.agent.discover_skills", lambda workspace: ())
    database = tmp_path / "sessions.sqlite"
    agent = create_default_agent(tmp_path, session_database_path=database, enable_multi_agent=False)
    try:
        assert agent.collaboration is None
        names = set(agent.tool_registry.names)
        assert not names.intersection({"spawn_agent", "send_messages", "wait_agents", "close_agent"})
        assert {"run_command", "run_code", "write_file", "edit_file", "load_skill"} <= names
        assert PARENT_INSTRUCTIONS.strip() not in agent.model_client.instructions
        assert agent.model_client.instructions.startswith(BASE_INSTRUCTIONS.rstrip())
        assert agent.run("Remember this task") == "done"
        assert "collaboration runtime" not in agent.model_client.inputs[-1]
        session_id = agent.session_id
    finally:
        agent.close()
    resumed = create_default_agent(tmp_path, session_database_path=database,
                                   resume_session_id=session_id, enable_multi_agent=False)
    try:
        assert resumed.run("Continue") == "done"
        assert "Remember this task" in resumed.model_client.inputs[-1]
        assert resumed.collaboration is None
        assert resumed.model_client.instructions.startswith(BASE_INSTRUCTIONS.rstrip())
    finally:
        resumed.close()
    multi = create_default_agent(tmp_path, session_database_path=database, enable_multi_agent=True)
    try:
        assert multi.model_client.instructions.replace(
            "\n\n" + PARENT_INSTRUCTIONS.strip(), "", 1,
        ).startswith(BASE_INSTRUCTIONS.rstrip())
        assert PARENT_INSTRUCTIONS.strip() in multi.model_client.instructions
    finally:
        multi.close()
