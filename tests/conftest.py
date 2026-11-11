import asyncio

import pytest


@pytest.fixture
def terminal_app(monkeypatch):
    """A real UI with deterministic colors and no model, network or session writes."""
    from praxis_agent.terminal.app import PraxisApp
    from praxis_agent.terminal.views import ViewEventSink

    monkeypatch.delenv("NO_COLOR", raising=False)
    return PraxisApp(ViewEventSink())


@pytest.fixture
def wait_for_ui():
    async def wait(pilot, condition):
        async with asyncio.timeout(5):
            while not condition():
                await pilot.pause(0.02)
        await pilot.pause()

    return wait


@pytest.fixture
def drag_selection():
    async def drag(pilot, widget, start, end):
        await pilot.mouse_down(widget, offset=start)
        await pilot.hover(widget, offset=end)
        await pilot.mouse_up(widget, offset=end)
        await pilot.pause()

    return drag

pytest_plugins = ["support.service"]


@pytest.fixture(autouse=True)
def model_configuration_file(tmp_path_factory, monkeypatch):
    """Every test service uses an isolated file, never the user's gateway or keys."""
    import yaml
    path = tmp_path_factory.mktemp("model-config") / "praxis.yaml"
    path.write_text(yaml.safe_dump({"models": {"test": {
        "model": "test", "base_url": "https://example.test/v1", "protocol": "responses",
        "context_window": 128000, "max_output_tokens": 8192,
    }}}), encoding="utf-8")
    monkeypatch.setenv("PRAXIS_CONFIG", str(path))
    return path
