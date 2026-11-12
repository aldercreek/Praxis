import sys
from types import SimpleNamespace

import pytest

from praxis_agent.config import ObservabilityConfig, load_config
from praxis_agent.events import ChildAgentEvent, TurnCompleted, TurnStarted
from praxis_agent.factory import create_default_agent
from praxis_agent.mcp import MCPConnections
from praxis_agent.model import ModelResponse
from praxis_agent.multi_agent.models import AgentStatus
from praxis_agent.observability import Observability, load_observability
from tests.support.service import wait_until
from tests.test_config import save


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    (tmp_path / "praxis_test_observer.py").write_text('''
instances = []
def create_sink(*, options):
    sink = Sink(options)
    instances.append(sink)
    return sink
class Sink:
    def __init__(self, options):
        self.options = options
        self.events = []
        self.closed = 0
    def emit(self, event):
        self.events.append(event)
        if self.options.get("fail"):
            raise RuntimeError("private plugin error")
    def close(self):
        self.closed += 1
''', encoding="utf-8")
    metadata = tmp_path / "praxis_test_observer-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: praxis-test-observer\nVersion: 1.0\n")
    (metadata / "entry_points.txt").write_text(
        "[praxis.observability]\ntest-observer = praxis_test_observer:create_sink\n"
        "unused = unavailable_package:create_sink\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "praxis_test_observer", raising=False)
    import praxis_test_observer
    return praxis_test_observer


@pytest.mark.parametrize("value", [{"enabled": "false"}, {"provider": ""}, {"provider": " a "},
                                   {"provider": "x", "options": []}, {"endpoint": "url"}])
def test_invalid_observability_config(tmp_path, value):
    with pytest.raises(ValueError, match="observability"):
        load_config(save(tmp_path, observability=value))


def test_optional_config_does_not_discover_plugins(tmp_path, monkeypatch):
    def unexpected(**kwargs):
        pytest.fail("Plugin discovery should be disabled")
    monkeypatch.setattr("praxis_agent.observability.entry_points", unexpected)
    assert load_observability(load_config(save(tmp_path, observability={"enabled": False})).observability) is None


def test_real_entry_point_loads_only_selected_provider_and_copies_options(tmp_path, plugin):
    config = load_config(save(tmp_path, observability={
        "provider": "test-observer", "options": {"nested": {"key": "value"}},
    })).observability
    sink = load_observability(config)
    instance = plugin.instances[0]
    instance.options["nested"]["key"] = "changed"
    assert config.options["nested"]["key"] == "value"
    event = TurnStarted("turn", "task")
    sink.emit(event)
    assert instance.events == [event]
    sink.close()
    assert instance.closed == 1 and instance.events == [event]


def test_missing_duplicate_and_invalid_plugins(monkeypatch):
    config = ObservabilityConfig("missing")
    monkeypatch.setattr("praxis_agent.observability.entry_points", lambda **kw: ())
    with pytest.raises(ValueError, match="not installed"):
        load_observability(config)
    monkeypatch.setattr("praxis_agent.observability.entry_points", lambda **kw: (object(), object()))
    with pytest.raises(ValueError, match="Multiple"):
        load_observability(config)
    for factory in (None, lambda **kw: object()):
        monkeypatch.setattr("praxis_agent.observability.entry_points",
                            lambda **kw: (SimpleNamespace(load=lambda: factory),))
        with pytest.raises(TypeError):
            load_observability(config)


@pytest.mark.parametrize("fail", [False, True])
def test_factory_wires_plugin_and_closes_after_workers(tmp_path, monkeypatch, plugin, fail):
    class Client:
        def __init__(self, *args, **kwargs):
            pass
        def create_response(self, *args, **kwargs):
            return ModelResponse(({"role": "assistant", "content": "done"},), "done", ())
        def close(self):
            pass
    monkeypatch.setattr("praxis_agent.factory.OpenAIModelClient", Client)
    monkeypatch.setattr("praxis_agent.factory.discover_skills", lambda workspace: ())
    config = load_config(save(tmp_path, observability={
        "provider": "test-observer", **({"options": {"fail": True}} if fail else {}),
    }))
    ui_events = []
    mcp = MCPConnections(tmp_path / "missing.json")
    mcp.start()
    observer = Observability(config.observability, tmp_path / "phoenix")
    agent = create_default_agent(tmp_path, configuration=config, mcp=mcp, observability=observer,
                                 session_database_path=tmp_path / "state.sqlite",
                                 event_sink=SimpleNamespace(emit=ui_events.append))
    try:
        assert agent.run("task") == "done"
        assert any(isinstance(e, TurnCompleted) for e in ui_events)
        runtime = agent.collaboration
        child = runtime.spawn("task", working_directory=str(tmp_path))["agent_id"]
        with runtime.condition:
            assert runtime.condition.wait_for(
                lambda: runtime.handles[child].status == AgentStatus.IDLE, timeout=5)
        agent._pump_worker_events()
        wait_until(lambda: observer.snapshot()["state"] == "failed" if fail else
                   len(plugin.instances) == 1 and len(plugin.instances[0].events) == len(ui_events))
        assert len(plugin.instances) == 1
        instance = plugin.instances[0]
        if fail:
            assert len(instance.events) == 1
        else:
            assert instance.options == {}
            assert instance.events == ui_events
            assert any(isinstance(e, ChildAgentEvent) for e in instance.events)
            assert len({id(e) for e in instance.events}) == len(instance.events)
    finally:
        agent.close()
        agent.close()
        observer.close()
        mcp.close()
    assert plugin.instances[0].closed == 1



def test_explicit_missing_provider_warns_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("praxis_agent.observability.entry_points", lambda **kw: ())
    observer = Observability(ObservabilityConfig("missing"), tmp_path / "phoenix")
    try:
        wait_until(lambda: observer.snapshot()["state"] == "failed")
        assert "not installed" in observer.snapshot()["message"]
        assert observer.process is None
        assert not (tmp_path / "phoenix").exists()
    finally:
        observer.close()


@pytest.mark.parametrize("settings", [None, {}, {"enabled": True}])
def test_missing_provider_defaults_to_enabled_phoenix(tmp_path, settings):
    config = load_config(save(tmp_path, **({"observability": settings} if settings is not None else {})))
    assert config.observability.enabled
    assert config.observability.provider is None


def test_disabled_skips_all_loading(tmp_path, monkeypatch):
    monkeypatch.setattr("praxis_agent.observability.entry_points", lambda **kw: pytest.fail("discovery"))
    observer = Observability(ObservabilityConfig("missing", enabled=False), tmp_path / "phoenix")
    observer.start()
    assert observer.snapshot()["state"] == "disabled"
    assert observer.thread is None and observer.process is None
    observer.close()


def test_default_startup_queues_events_and_shares_one_service(tmp_path, monkeypatch):
    from threading import Event
    entered, release = Event(), Event()
    services, sinks = [], []
    class Service:
        alive = True
        def __init__(self, directory):
            self.closed = 0
            services.append(self)
        def start(self, stopping):
            entered.set()
            assert release.wait(5)
            return "http://127.0.0.1:12345"
        def close(self):
            self.closed += 1
            self.alive = False
    class Sink:
        def __init__(self, url, on_error):
            self.events = []
            self.closed = 0
            sinks.append(self)
        def emit_for_session(self, session_id, event):
            self.events.append((session_id, event))
        def close(self):
            self.closed += 1
    monkeypatch.setattr("praxis_agent.phoenix_service.LocalPhoenix", Service)
    monkeypatch.setattr("praxis_agent.phoenix_sink.PhoenixSink", Sink)
    observer = Observability(ObservabilityConfig(), tmp_path)
    try:
        assert entered.wait(5)
        event = TurnStarted("turn", "task")
        first = observer.bind(lambda: "one")
        second = observer.bind(lambda: "two")
        first.emit(event)
        second.emit_live(event)
        assert observer.snapshot()["state"] == "starting"
        release.set()
        wait_until(lambda: sinks and len(sinks[0].events) == 2)
        assert sinks[0].events == [("one", event), ("two", event)]
        assert len(services) == 1
        services[0].alive = False
        wait_until(lambda: observer.snapshot()["state"] == "failed")
    finally:
        release.set()
        observer.close()
    assert sinks[0].closed == 1


def test_failed_selected_provider_can_retry_without_default(tmp_path, monkeypatch):
    calls = []
    sink = SimpleNamespace(emit=lambda e: None, close=lambda: calls.append("closed"))
    def load(config):
        calls.append(config.provider)
        if len(calls) == 1:
            raise RuntimeError("unavailable")
        return sink
    monkeypatch.setattr("praxis_agent.observability.load_observability", load)
    observer = Observability(ObservabilityConfig("trip"), tmp_path)
    try:
        wait_until(lambda: observer.snapshot()["state"] == "failed")
        observer.thread.join(2)
        observer.start()
        wait_until(lambda: observer.snapshot()["state"] == "ready")
        assert observer.process is None
        assert calls == ["trip", "trip"]
    finally:
        observer.close()
    assert calls[-1] == "closed"
