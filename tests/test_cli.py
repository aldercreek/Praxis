import asyncio

import pytest
from textual.widgets import Input

from praxis_agent.cli import main
from praxis_agent.terminal.app import PraxisApp
from praxis_agent.terminal.views import ViewEventSink
from praxis_agent.events import (
    AssistantTextCompleted,
    AssistantTextDelta,
    EventSink,
    ModelRequestCompleted,
    ModelRequestStarted,
)


class FakeAgent:
    def __init__(self) -> None:
        self.tasks: list[str] = []
        self.clear_count = 0
        self.close_count = 0
        self.event_sink: EventSink | None = None

    def attach_event_sink(self, event_sink: EventSink) -> "FakeAgent":
        self.event_sink = event_sink
        return self

    def run(self, task: str) -> str:
        self.tasks.append(task)
        output_text = f"completed: {task}"
        if self.event_sink is not None:
            turn_id = f"turn-{len(self.tasks)}"
            self.event_sink.emit(ModelRequestStarted(turn_id, 1))
            self.event_sink.emit(
                AssistantTextDelta(turn_id, 1, "completed: ")
            )
            self.event_sink.emit(AssistantTextDelta(turn_id, 1, task))
            self.event_sink.emit(ModelRequestCompleted(turn_id, 1))
            self.event_sink.emit(
                AssistantTextCompleted(turn_id, output_text)
            )
        return output_text

    def clear_history(self) -> None:
        self.clear_count += 1

    def close(self) -> None:
        if not self.close_count:
            self.close_count += 1


def test_cli_runs_a_task(capsys, monkeypatch) -> None:
    agent = FakeAgent()
    monkeypatch.setattr(
        "praxis_agent.cli.create_default_agent",
        lambda workspace, **callbacks: agent.attach_event_sink(
            callbacks["event_sink"]
        ),
    )

    def run(app):
        async def exercise():
            async with app.run_test() as pilot:
                await pilot.pause(0.2)
                assert isinstance(app.sink, ViewEventSink)
                assert app.store.views["main"].committed_blocks
                assert not app.busy
        asyncio.run(exercise())
        return 0

    monkeypatch.setattr(PraxisApp, "run", run)
    exit_code = main(["inspect", "repository"])

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    assert agent.tasks == ["inspect repository"]
    assert agent.close_count == 1


def test_cli_runs_an_interactive_session(capsys, monkeypatch) -> None:
    agent = FakeAgent()
    monkeypatch.setattr(
        "praxis_agent.cli.create_default_agent",
        lambda workspace, **callbacks: agent.attach_event_sink(
            callbacks["event_sink"]
        ),
    )
    def run(app):
        async def exercise():
            async with app.run_test() as pilot:
                for task in ["first task", "/clear", "follow-up task", "/exit"]:
                    app.query_one("#task", Input).value = task
                    await pilot.press("enter")
                    await pilot.pause(0.2)
        asyncio.run(exercise())
        return 0

    monkeypatch.setattr(PraxisApp, "run", run)

    exit_code = main([])

    assert exit_code == 0
    assert agent.tasks == ["first task", "follow-up task"]
    assert agent.clear_count == 1
    assert agent.close_count == 1
    assert capsys.readouterr().out == ""


def test_cli_rejects_removed_plain_option(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--plain"])
    assert error.value.code == 2
    assert "unrecognized arguments: --plain" in capsys.readouterr().err
