from praxis_agent.cli import main


class FakeAgent:
    def __init__(self) -> None:
        self.tasks: list[str] = []
        self.clear_count = 0

    def run(self, task: str) -> str:
        self.tasks.append(task)
        return f"completed: {task}"

    def clear_history(self) -> None:
        self.clear_count += 1


def test_cli_runs_a_task(capsys, monkeypatch) -> None:
    agent = FakeAgent()
    monkeypatch.setattr(
        "praxis_agent.cli.create_default_agent",
        lambda workspace, **callbacks: agent,
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
        lambda workspace, **callbacks: agent,
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
