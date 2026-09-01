import pytest

from praxis_agent.agent import Agent


def test_agent_accepts_a_task() -> None:
    result = Agent().run("inspect repository")

    assert result == "Praxis received: inspect repository"


def test_agent_rejects_an_empty_task() -> None:
    with pytest.raises(ValueError, match="task must not be empty"):
        Agent().run("   ")

