from __future__ import annotations

from praxis_agent.tools.registry import Tool, ToolContext


_MESSAGE_IDS_SCHEMA = {
    "type": "object",
    "properties": {
        "message_ids": {
            "type": "array", "minItems": 1,
            "items": {"type": "string", "minLength": 1},
            "description": (
                "IDs of the messages to replace. Copy exact message_id values from "
                "labels in the current context. All messages, including those created "
                "by compression, use the same message_id field. Changes appear next turn; "
                "do not target messages already changed in this turn."
            ),
        },
    },
    "required": ["message_ids"],
    "additionalProperties": False,
}


def _change(
    context: ToolContext, action: str, message_ids: list[str], instructions: str = "",
) -> str:
    manager = context.context_manager
    if manager is None:
        raise ValueError("context management is unavailable")
    return manager.change_messages(
        action, message_ids, instructions=instructions,
    )


def delete(context: ToolContext, message_ids: list[str]) -> str:
    return _change(context, "delete", message_ids)


def compress(
    context: ToolContext, message_ids: list[str], instructions: str,
) -> str:
    return _change(context, "compress", message_ids, instructions)


def create_context_tools() -> tuple[Tool, ...]:
    """只注册定义；当前会话和上下文管理器由执行时注入。"""
    return (
        Tool(
            name="delete",
            description=(
                "Replace selected user/assistant text messages, tool results or summaries "
                "with one hidden-content marker in future inputs. Use visible message_id labels, not positions or "
                "call_ids. Preserve current requirements and unresolved evidence; prefer "
                "compress when facts must remain available. Returns one replacement message_id "
                "and an operation_id. Does not delete original history or files."
            ),
            parameters=_MESSAGE_IDS_SCHEMA, handler=delete,
        ),
        Tool(
            name="compress",
            description=(
                "Summarize selected user/assistant text, tool results or earlier summaries "
                "in a separate model request. Specify message_ids and instructions describing "
                "what to preserve or omit. Replaces the selected messages with one message "
                "containing the summary. Returns its message_id and an operation_id. "
                "Original contents remain saved in complete history. Placeholders and "
                "context-management tool results can also be selected. Protocol calls "
                "and multimodal content cannot be selected."
            ),
            parameters={
                "type": "object",
                "properties": {
                    **_MESSAGE_IDS_SCHEMA["properties"],
                    "instructions": {
                        "type": "string", "minLength": 1,
                        "description": "What facts, constraints, evidence and unfinished work the summary must preserve; what may be omitted.",
                    },
                },
                "required": ["message_ids", "instructions"],
                "additionalProperties": False,
            },
            handler=compress,
        ),
    )
