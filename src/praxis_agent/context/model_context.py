import json

from praxis_agent.context.capacity import ContextStatus


def build_context_status_context(
    status: ContextStatus,
) -> dict[str, object]:
    """构造只参与当前模型请求的上下文容量快照。"""
    snapshot = {
        "type": "praxis_context_status",
        "estimated_input_tokens": status.estimated_input_tokens,
        "effective_input_limit_tokens": (
            status.effective_input_limit_tokens
        ),
        "usage_percent": status.usage_percent,
        "pressure": status.pressure.value,
    }
    return {
        "role": "developer",
        "content": (
            "Current Praxis context capacity (estimated):\n"
            + json.dumps(
                snapshot,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ),
    }
