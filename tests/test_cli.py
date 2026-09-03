from praxis_agent.cli import main


class FakeAgent:
    def run(self, task: str) -> str:
        return f"completed: {task}"


def test_cli_runs_a_task(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        "praxis_agent.cli.create_default_agent",
        lambda workspace, **callbacks: FakeAgent(),
    )

    exit_code = main(["inspect", "repository"])

    assert exit_code == 0
    assert capsys.readouterr().out == "completed: inspect repository\n"
