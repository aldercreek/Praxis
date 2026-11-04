from praxis_agent.context.capacity import ContextPolicy, ContextStatus


def build_context_status_context(
    status: ContextStatus, policy: ContextPolicy | None = None,
) -> dict[str, object]:
    """构造只参与当前模型请求的上下文容量快照。"""
    policy = policy or ContextPolicy()
    lines = ["<context_status>", "1. Usage", f"   Context usage: {status.usage_percent:g}%"]
    ratio = status.estimated_input_tokens / status.effective_input_limit_tokens
    if ratio >= policy.soft_ratio:
        urgency = (
            "Urgent: prioritize context management before further large outputs."
            if ratio >= policy.emergency_ratio else
            "Prioritize context management before further large outputs."
            if ratio >= policy.hard_ratio else
            "Review reusable history before further large outputs."
        )
        lines.extend([
            "2. Reminder",
            f"   {urgency}",
            f"   Automatic compaction starts at {policy.auto_compact_ratio * 100:g}%: tool bodies are omitted",
            "   and conversation history may be summarized. Use delete/compress for redundant material;",
            "   preserve active requirements, verified conclusions, unresolved work and source references.",
            "   Do not repeatedly compact unchanged or essential information.",
        ])
    lines.append("</context_status>")
    return {"role": "developer", "content": "\n".join(lines)}
