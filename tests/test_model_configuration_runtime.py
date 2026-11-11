
import pytest
import yaml

from praxis_agent.config import load_config
from praxis_agent.factory import create_default_agent
from praxis_agent.mcp import MCPConnections
from praxis_agent.model import ModelResponse
from praxis_agent.multi_agent.models import AgentStatus


def configuration(tmp_path, **options):
    models = {}
    for name, window in [('main', 128000), ('worker', 64000), ('summary', 32000)]:
        models[name] = dict(protocol='responses', base_url=f'https://{name}.example/v1',
                            model=f'upstream-{name}', context_window=window, max_output_tokens=4096)
    path = tmp_path / 'models.yaml'
    path.write_text(yaml.safe_dump(dict(default_model='main', models=models, **options)), encoding='utf-8')
    return load_config(path)


@pytest.mark.parametrize('independent', [True, False])
def test_factory_routes_main_worker_and_summary_with_own_budgets(tmp_path, monkeypatch, independent):
    created = []
    class Client:
        def __init__(self, config, instructions, **options):
            self.config, self.instructions = config, instructions
            self.closed = False
            created.append(self)
        def close(self):
            self.closed = True
        def create_response(self, *args, **kwargs):
            return ModelResponse(({'role': 'assistant', 'content': 'done'},), 'done', ())
    monkeypatch.setattr('praxis_agent.factory.OpenAIModelClient', Client)
    monkeypatch.setattr('praxis_agent.factory.discover_skills', lambda workspace: ())
    config = configuration(tmp_path, **({'compression': {'model': 'summary'},
                                        'subagents': {'model': 'worker', 'max_concurrent': 2}} if independent else {}))
    mcp = MCPConnections(tmp_path / 'missing-mcp.json')
    mcp.start()
    agent = create_default_agent(tmp_path, configuration=config,
                                session_database_path=tmp_path/'state.sqlite', mcp=mcp)
    try:
        assert created[0].config.model == 'upstream-main'
        assert created[1].config.model == ('upstream-summary' if independent else 'upstream-main')
        assert agent.context_manager._summarizer.model_limits.context_window_tokens == (32000 if independent else 128000)
        runtime = agent.collaboration
        spawned = runtime.spawn('Report done', working_directory=str(tmp_path))
        with runtime.condition:
            assert runtime.condition.wait_for(lambda: len(created) >= 4
                                              and runtime.handles[spawned['agent_id']].status == AgentStatus.IDLE, timeout=5)
        assert created[2].config.model == ('upstream-worker' if independent else 'upstream-main')
        assert created[3].config.model == ('upstream-summary' if independent else 'upstream-main')
        # Explicit spawn selection takes precedence over the worker default.
        second = runtime.spawn('Report done again', working_directory=str(tmp_path), model='main')
        with runtime.condition:
            assert runtime.condition.wait_for(lambda: len(created) >= 6
                                              and runtime.handles[second['agent_id']].status == AgentStatus.IDLE, timeout=5)
        assert created[4].config.model == 'upstream-main'
        assert created[5].config.model == ('upstream-summary' if independent else 'upstream-main')
    finally:
        agent.close()
        mcp.close()
    assert all(client.closed for client in created)


def test_service_configuration_reload_is_atomic_and_file_only(service, tmp_path):
    from praxis_agent.service.errors import ServiceError
    harness = service()
    old = harness.manager.config
    invalid = tmp_path/'invalid.yaml'
    invalid.write_text('models: {}')
    with pytest.raises(ServiceError):
        harness.manager.configure({'config_path': str(invalid)})
    assert harness.manager.config is old
    with pytest.raises(ServiceError):
        harness.manager.configure({'base_url': 'https://ignored.example'})
    new = configuration(tmp_path)
    harness.manager.configure({'config_path': str(new.path)})
    assert harness.manager.configuration()['models'] == ['main', 'summary', 'worker']
    assert harness.manager.configuration()['model'] == 'main'
    # Loading a file creates a snapshot; a later file edit does not change it.
    new.path.write_text('models: {}')
    assert harness.manager.config.resolve().model == 'upstream-main'
