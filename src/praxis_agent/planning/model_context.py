import json

from praxis_agent.planning.state import RunState


def build_run_state_context(run_state: RunState) -> dict[str, object]:
    """构造只参与当前模型请求的任务状态快照。"""
    snapshot = {
        "type": "praxis_run_state",
        "plan": [
            {
                "step": step.description,
                "status": step.status.value,
            }
            for step in run_state.plan
        ],
        "changed_files": sorted(run_state.changed_files),
    }
    return {
        "role": "developer",
        "content": (
            "Current Praxis run state:\n"
            + json.dumps(
                snapshot,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ),
    }
