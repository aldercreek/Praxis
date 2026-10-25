"""Restore saved turns that have conversation items but no display journal."""

from __future__ import annotations

import json

from praxis_agent.mcp_results import decode_tool_output

from .transcript import content_text


def session_view_events(connection, session_id, journal):
    """Combine saved history and journal frames without inventing SSE sequence IDs."""
    first_event = {}
    started = set()
    settled = set()
    for index, frame in enumerate(journal):
        turn_id = frame["turn_id"]
        first_event.setdefault(turn_id, index)
        if frame["agent_id"] != "main":
            continue
        if frame["event"] == "TurnStarted":
            started.add(turn_id)
        if frame["event"] in {
            "TurnCompleted",
            "TurnFailed",
            "TurnCancelled",
            "TurnSettled",
        }:
            settled.add(turn_id)

    runs = connection.execute(
        "SELECT * FROM runs WHERE session_id = ? ORDER BY started_at, id", (session_id,)
    ).fetchall()
    # Insert each saved turn before its recovery events, or before the next turn.
    positions = {}
    position = len(journal)
    for run in reversed(runs):
        position = first_event.get(run["id"], position)
        positions[run["id"]] = position
    insertions = {}
    for run in runs:
        if run["id"] in started or run["status"] == "running":
            continue
        rows = connection.execute(
            "SELECT payload_json FROM conversation_items WHERE session_id = ? AND run_id = ? ORDER BY sequence",
            (session_id, run["id"]),
        ).fetchall()
        if rows:
            frames = saved_turn_events(
                session_id,
                run,
                [json.loads(row[0]) for row in rows],
                run["id"] in settled,
            )
            insertions.setdefault(positions[run["id"]], []).extend(frames)
    result = []
    for index in range(len(journal) + 1):
        result.extend(insertions.get(index, []))
        if index < len(journal):
            result.append(journal[index])
    return result


def saved_turn_events(session_id, run, items, has_terminal_event):
    def frame(event, **data):
        return dict(
            event=event,
            session_id=session_id,
            turn_id=run["id"],
            agent_id="main",
            data=data,
        )

    calls = {}
    first_user = True
    for number, item in enumerate(items, 1):
        role, kind = item.get("role"), item.get("type")
        if role in {"user", "assistant", "system"}:
            text = content_text(item.get("content"))
            if not text:
                continue
            if role == "user" and first_user:
                first_user = False
                yield frame("TurnStarted", task=text)
            elif role == "assistant":
                yield frame("ModelRequestStarted", request_number=number)
                yield frame(
                    "ModelRequestCompleted", request_number=number, output_text=text
                )
            else:
                yield frame(
                    "MessagesReceived",
                    items=[dict(role=role, content=text)],
                    source="user" if role == "user" else "agent",
                )
        elif kind == "function_call":
            call_id = item["call_id"]
            calls[call_id] = item.get("name", "")
            try:
                arguments = json.loads(item.get("arguments", "{}"))
            except (ValueError, TypeError):
                arguments = None
            if not isinstance(arguments, dict):
                arguments = {"raw": item.get("arguments", "")}
            yield frame(
                "ToolCallPrepared",
                call_id=call_id,
                tool_name=calls[call_id],
                arguments=arguments,
                requires_approval=False,
            )
            yield frame("ToolStarted", call_id=call_id, tool_name=calls[call_id])
        elif kind == "function_call_output" and item.get("call_id") in calls:
            call_id = item["call_id"]
            try:
                decoded = decode_tool_output(item.get("output", "{}"))
            except (ValueError, TypeError):
                decoded = dict(success=False, output=str(item.get("output", "")))
            if not isinstance(decoded, dict):
                decoded = dict(success=False, output=str(decoded))
            output = decoded.get("output", "")
            result = dict(
                success=bool(decoded.get("success")),
                output=output
                if isinstance(output, str)
                else json.dumps(output, ensure_ascii=False),
            )
            if isinstance(output, dict):
                result["structured_output"] = output
            for key in ("structured_output", "file_change"):
                if decoded.get(key):
                    result[key] = decoded[key]
            yield frame(
                "ToolFinished", call_id=call_id, tool_name=calls[call_id], result=result
            )
    plan = json.loads(run["plan_json"])
    if plan:
        yield frame(
            "PlanUpdated",
            plan=[
                dict(description=step["step"], status=step["status"]) for step in plan
            ],
        )
    if not has_terminal_event:
        status = run["status"]
        if status == "failed":
            yield frame(
                "TurnFailed",
                error_type="SavedTurnFailed",
                message=run["error"] or "Execution failed",
            )
        else:
            yield frame("TurnCancelled" if status == "cancelled" else "TurnCompleted")
