from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from praxis_agent.planning.state import RunState


class PlanStepStatus(StrEnum):
    """表示计划中一个步骤的执行状态。"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class PlanStep:
    """保存模型通过 update_plan 提交的一个计划步骤。"""

    description: str
    status: PlanStepStatus


def replace_plan(
    run_state: RunState,
    items: list[Mapping[str, object]],
) -> None:
    """校验并原子替换指定运行状态中的计划。"""
    if not isinstance(items, list) or not items:
        raise ValueError("plan must be a non-empty list")

    updated_plan: list[PlanStep] = []
    in_progress_count = 0
    for index, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            raise TypeError(f"plan item {index} must be an object")
        if set(item) != {"step", "status"}:
            raise ValueError(
                f"plan item {index} must contain only step and status"
            )

        description = item["step"]
        encoded_status = item["status"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"plan item {index} step must be a non-empty string"
            )
        if not isinstance(encoded_status, str):
            raise TypeError(f"plan item {index} status must be a string")

        try:
            status = PlanStepStatus(encoded_status)
        except ValueError as error:
            allowed = ", ".join(value.value for value in PlanStepStatus)
            raise ValueError(
                f"plan item {index} status must be one of: {allowed}"
            ) from error

        if status is PlanStepStatus.IN_PROGRESS:
            in_progress_count += 1
        updated_plan.append(PlanStep(description.strip(), status))

    if in_progress_count > 1:
        raise ValueError("plan must not contain multiple in_progress steps")

    run_state.plan = updated_plan
