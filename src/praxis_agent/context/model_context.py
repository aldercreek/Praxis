from praxis_agent.context.capacity import ContextStatus


def build_context_status_context(
    status: ContextStatus,
) -> dict[str, object]:
    """构造只参与当前模型请求的上下文容量快照。"""
    return {
        "role": "developer",
        "content": f"Context usage: {status.usage_percent:g}%",
    }
