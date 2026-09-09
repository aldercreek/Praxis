from collections.abc import Mapping

from praxis_agent.planning.plan import replace_plan
from praxis_agent.tools.registry import Tool, ToolContext


_UPDATE_PLAN_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "array",
            "minItems": 1,
            "description": "The complete updated task plan in execution order.",
            "items": {
                "type": "object",
                "properties": {
                    "step": {
                        "type": "string",
                        "minLength": 1,
                        "description": "A concise, actionable task step.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": "The current status of this step.",
                    },
                },
                "required": ["step", "status"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["plan"],
    "additionalProperties": False,
}


def _update_plan(
    context: ToolContext,
    plan: list[Mapping[str, object]],
) -> str:
    """使用 Agent 注入的运行时状态替换当前计划。"""
    replace_plan(context.run_state, plan)
    return "Plan updated."


def create_update_plan_tool() -> Tool:
    """创建不绑定具体任务状态的计划工具定义。"""
    return Tool(
        name="update_plan",
        description=(
            "Replace the current task plan with an updated checklist. "
            "Use it for non-trivial tasks and keep at most one step in_progress."
        ),
        parameters=_UPDATE_PLAN_PARAMETERS,
        handler=_update_plan,
    )
