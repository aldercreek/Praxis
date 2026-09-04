from praxis_agent.cli import main
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


def test_cli_runs_a_task(capsys, monkeypatch) -> None:
    agent = FakeAgent()
    monkeypatch.setattr(
        "praxis_agent.cli.create_default_agent",
        lambda workspace, **callbacks: agent.attach_event_sink(
            callbacks["event_sink"]
        ),
    )

    exit_code = main(["inspect", "repository"])

    assert exit_code == 0
    assert capsys.readouterr().out == "completed: inspect repository\n"
    assert agent.tasks == ["inspect repository"]


def test_cli_runs_an_interactive_session(capsys, monkeypatch) -> None:
    agent = FakeAgent()
    entered_tasks = iter(
        ["first task", "/clear", "follow-up task", "/exit"]
    )
    monkeypatch.setattr(
        "praxis_agent.cli.create_default_agent",
        lambda workspace, **callbacks: agent.attach_event_sink(
            callbacks["event_sink"]
        ),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: next(entered_tasks),
    )

    exit_code = main([])

    assert exit_code == 0
    assert agent.tasks == ["first task", "follow-up task"]
    assert agent.clear_count == 1
    assert capsys.readouterr().out == (
        "completed: first task\n"
        "Conversation cleared.\n"
        "completed: follow-up task\n"
    )
