"""Trusted runtime presentation, kept separate from tool-owned result data."""


def execution_status(presentation):
    data = presentation or {}
    if not data.get("execution_status"):
        return None
    if data.get("blocked_by") == "hook":
        return "blocked"
    if data.get("blocked_by") == "approval":
        return "denied"
    if data.get("result_processing") == "error":
        return "processing_error"
    if data.get("cancelled"):
        return "cancelled"
    if data.get("command_status") in {"running", "cancelled"}:
        return data["command_status"]
    if data.get("command_exit_code") not in (None, 0):
        return "failed"
    return "completed" if data["execution_status"] == "success" else "failed"


def presentation_notes(presentation):
    data = presentation or {}
    notes = []
    if data.get("arguments_modified"):
        notes.append("Arguments modified by hook; final arguments shown.")
    if data.get("approval_hooks"):
        notes.append("Approval required by hook: " + ", ".join(data["approval_hooks"]))
    if data.get("blocked_by") == "hook":
        notes.append("Blocked by hook; tool was not executed.")
    if data.get("result_processing") == "modified":
        notes.append("Output processed by hook.")
    if data.get("result_processing") == "error":
        outcome = "succeeded" if data.get("execution_status") == "success" else "failed"
        notes.append(f"Result processing failed; tool execution {outcome}.")
    return notes
