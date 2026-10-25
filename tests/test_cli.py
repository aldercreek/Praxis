import asyncio

import pytest

from praxis_agent.cli import main
from praxis_agent.terminal.app import PraxisApp


def test_cli_creates_session_and_submits_task_to_shared_service(service, monkeypatch):
    h = service()
    client = h.client()
    monkeypatch.setattr("praxis_agent.client.ServiceClient", lambda: client)
    monkeypatch.chdir(h.path)
    def run(app):
        async def exercise():
            async with app.run_test() as pilot:
                async with asyncio.timeout(5):
                    while not app.connection.summary.get("latest_turn") or app.busy:
                        await pilot.pause(0.05)
                assert app.store.views["main"].committed_blocks
        asyncio.run(exercise())
        return 0
    monkeypatch.setattr(PraxisApp, "run", run)
    assert main(["inspect", "repository"]) == 0
    assert h.agents[0].run_state.original_task == "inspect repository"


def test_cli_lists_sessions_without_loading_agent(service, monkeypatch, capsys):
    h = service()
    session = h.create()
    client = h.client()
    monkeypatch.setattr("praxis_agent.client.ServiceClient", lambda: client)
    monkeypatch.chdir(h.path)
    assert main(["--sessions"]) == 0
    assert session["id"][:8] in capsys.readouterr().out
    assert h.agents == []


def test_cli_rejects_unknown_option(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--plain"])
    assert error.value.code == 2
