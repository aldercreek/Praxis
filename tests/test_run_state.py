import pytest

from praxis_agent.planning import (
    PlanStepStatus,
    RunState,
    RunStatus,
    replace_plan,
)


def test_run_state_replaces_a_valid_plan() -> None:
    state = RunState("implement a feature")

    replace_plan(
        state,
        [
            {"step": "Inspect the code", "status": "completed"},
            {"step": "Implement the feature", "status": "in_progress"},
            {"step": "Run tests", "status": "pending"},
        ]
    )

    assert [step.description for step in state.plan] == [
        "Inspect the code",
        "Implement the feature",
        "Run tests",
    ]
    assert state.plan[1].status is PlanStepStatus.IN_PROGRESS


def test_run_state_rejects_multiple_steps_in_progress_atomically() -> None:
    state = RunState("implement a feature")
    replace_plan(state, [{"step": "Inspect", "status": "pending"}])

    with pytest.raises(ValueError, match="multiple in_progress"):
        replace_plan(
            state,
            [
                {"step": "Implement", "status": "in_progress"},
                {"step": "Test", "status": "in_progress"},
            ]
        )

    assert [step.description for step in state.plan] == ["Inspect"]


def test_run_state_records_files_and_terminal_status() -> None:
    state = RunState("edit files")

    state.record_changed_file("src\\example.py")
    state.record_changed_file("src/example.py")
    state.complete()

    assert state.changed_files == {"src/example.py"}
    assert state.status is RunStatus.COMPLETED
    assert state.error is None

    error = RuntimeError("request failed")
    state.fail(error)

    assert state.status is RunStatus.FAILED
    assert state.error == "RuntimeError: request failed"
