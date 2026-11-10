"""Copy retained conversation data without an unfinished tool-call batch."""
from copy import deepcopy

from praxis_agent.tools import ToolContext


def inherited_context(context: ToolContext) -> tuple[object, ...]:
    if context.history is None:
        raise ValueError("context inheritance requires the parent's conversation history")
    history = context.history.snapshot()
    if context.context_manager is not None:
        items = [message["payload"] for message in context.context_manager.operations.project(history)]
    else:
        from praxis_agent.context.operations import item_payload
        items = [item_payload(item) for item in history]
    # The spawn tool itself may be in a partially saved parallel batch. Keep only
    # a prefix with complete call/result pairs, including after context projection.
    while True:
        calls = {item.get("call_id"): index for index, item in enumerate(items)
                 if item.get("type") == "function_call"}
        outputs = {item.get("call_id") for item in items
                   if item.get("type") == "function_call_output"}
        unfinished = [index for call, index in calls.items() if call not in outputs]
        if not unfinished:
            break
        items = items[:min(unfinished)]
    while items and items[-1].get("type") == "reasoning":
        items.pop()
    calls = {item.get("call_id") for item in items if item.get("type") == "function_call"}
    return tuple(deepcopy(item) for item in items
                 if item.get("role") not in {"system", "developer"}
                 and (item.get("type") != "function_call_output" or item.get("call_id") in calls))
