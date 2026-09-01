from praxis_agent.cli import main


def test_cli_runs_a_task(capsys) -> None:
    exit_code = main(["inspect", "repository"])

    assert exit_code == 0
    assert capsys.readouterr().out == "Praxis received: inspect repository\n"

