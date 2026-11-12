import yaml

from support.service import wait_until


def test_observability_status_retry_and_configuration_reload(service, model_configuration_file, monkeypatch):
    from praxis_agent import observability

    attempts = []

    def unavailable(config):
        attempts.append(config.provider)
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(observability, "load_observability", unavailable)
    h = service()
    client = h.client()
    assert client.request("GET", "/api/observability")["state"] == "disabled"
    assert client.request("POST", "/api/observability", {})["state"] == "disabled"
    config = yaml.safe_load(model_configuration_file.read_text(encoding="utf-8"))
    config["observability"] = {"provider": "custom"}
    model_configuration_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    h.manager.configure({"config_path": str(model_configuration_file)})
    wait_until(lambda: not h.manager.observability.thread.is_alive())
    status = client.request("GET", "/api/observability")
    assert status["state"] == "failed"
    assert status["provider"] == "custom"
    assert status["url"] == ""
    assert "provider unavailable" in status["message"]
    client.request("POST", "/api/observability", {})
    wait_until(lambda: len(attempts) == 2 and not h.manager.observability.thread.is_alive())
    assert attempts == ["custom", "custom"]
    session = h.create()
    h.submit(session)
    assert h.settled(session)["session"]["latest_turn"]["status"] == "completed"
    config["observability"] = {"enabled": False}
    model_configuration_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    h.manager.configure({"config_path": str(model_configuration_file)})
    assert client.request("GET", "/api/observability")["state"] == "disabled"
