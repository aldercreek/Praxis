"""Deterministic tool-body reduction; original messages remain in the history store."""

import json
from collections.abc import Sequence

def _excerpt(value: object, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[:limit // 2] + "\n[omitted]\n" + text[-limit // 2:]


def _result_summary(output: object, message_id: str, tool_name: str | None) -> str:
    from praxis_agent.mcp_results import decode_tool_output

    try:
        data = decode_tool_output(output)
    except (ValueError, TypeError, KeyError, IndexError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    if data.get("context_omitted") is True:
        return output
    body = data.get("output", "")
    if isinstance(body, str):
        try:
            decoded = json.loads(body)
            if isinstance(decoded, dict):
                body = decoded
        except ValueError:
            pass
    detail = body if isinstance(body, dict) else {}
    success = data.get("success")
    result = {
        "context_omitted": True, "source_message_id": message_id,
        "status": "success" if success is True else "failed" if success is False else "unknown",
    }
    if isinstance(success, bool):
        result["success"] = success
    is_command = tool_name in {"run_command", "wait_command"}
    if is_command:
        # Only command bodies contain execution metadata. Other bodies are data;
        # even for commands, a nonzero exit code does not change tool success.
        for key in ("status", "exit_code", "session_id", "pid"):
            if key in detail:
                result[key] = detail[key]
    if success is False or (is_command and (
        detail.get("error") or detail.get("exit_code") not in (None, 0)
        or detail.get("status") in {"failed", "spawn_failed", "cancelled", "interrupted"}
    )):
        errors = {key: detail[key] for key in ("error", "message", "stderr", "stdout") if detail.get(key)}
        result["error"] = _excerpt(errors or body or data, 1600)
    return json.dumps(result, ensure_ascii=False)


def trim_tool_messages(messages: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Omit all completed tool bodies, retaining protocol pairs, outcomes and IDs."""
    calls = {m["payload"]["call_id"]: m["payload"].get("name") for m in messages
             if m["payload"].get("type") == "function_call" and m["payload"].get("call_id")}
    finished = {m["payload"].get("call_id") for m in messages
                if m["payload"].get("type") == "function_call_output"}
    result = []
    for message in messages:
        payload = message["payload"]
        kind = payload.get("type")
        updated = dict(payload)
        if kind == "function_call_output":
            updated["output"] = _result_summary(
                payload.get("output"), message["message_id"], calls.get(payload.get("call_id")),
            )
        elif kind == "function_call" and payload.get("call_id") in finished:
            try:
                arguments = json.loads(payload.get("arguments", ""))
            except (ValueError, TypeError):
                arguments = None

            def reduce(value):
                if isinstance(value, str) and len(value) > 1024:
                    return f"[Body omitted; original arguments: read_history {message['message_id']}]"
                if isinstance(value, dict):
                    return {key: reduce(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [reduce(item) for item in value]
                return value

            reduced = reduce(arguments)
            if reduced != arguments:
                updated["arguments"] = json.dumps(reduced, ensure_ascii=False)
                updated.pop("id", None)
        result.append({**message, "payload": updated} if updated != payload else message)
    return result
