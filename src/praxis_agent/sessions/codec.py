from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Row

from praxis_agent.planning.state import RunState, RunStatus

from .models import SessionRecord, SessionStoreError, SessionSummary
from .schema import _INTERRUPTED_TOOL_OUTPUT


def _session_from_row(row: Row) -> SessionRecord:
    return SessionRecord(
        id=str(row["id"]),
        workspace=Path(row["workspace"]),
        model=str(row["model"]),
        title=str(row["title"]),
        created_at=_parse_time(row["created_at"]),
        updated_at=_parse_time(row["updated_at"]),
    )


def _session_summary_from_row(row: Row) -> SessionSummary:
    encoded_status = row["latest_run_status"]
    return SessionSummary(
        id=str(row["id"]),
        title=str(row["title"]),
        updated_at=_parse_time(row["updated_at"]),
        latest_run_status=(
            None if encoded_status is None else RunStatus(encoded_status)
        ),
    )


def _encode_item(item: object) -> tuple[str, str]:
    if isinstance(item, Mapping):
        payload = dict(item)
    else:
        model_dump = getattr(item, "model_dump", None)
        if not callable(model_dump):
            raise SessionStoreError("conversation item must be a mapping or SDK model")
        payload = model_dump(mode="json")

    if not isinstance(payload, Mapping):
        raise SessionStoreError("conversation item must encode to an object")

    item_type = payload.get("type")
    if not isinstance(item_type, str) or not item_type:
        if isinstance(payload.get("role"), str):
            item_type = "message"
        else:
            raise SessionStoreError("conversation item has no type")
    try:
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise SessionStoreError(
            f"conversation item is not JSON serializable: {error}"
        ) from error
    return item_type, payload_json


def _encode_plan(run_state: RunState) -> str:
    return json.dumps(
        [
            {"step": step.description, "status": step.status.value}
            for step in run_state.plan
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _encode_changed_files(run_state: RunState) -> str:
    return json.dumps(
        sorted(run_state.changed_files),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _interrupted_tool_item(call_id: str) -> dict[str, object]:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(
            {"success": False, "output": _INTERRUPTED_TOOL_OUTPUT},
            ensure_ascii=False,
        ),
    }


def _normalize_workspace(workspace: Path) -> str:
    return os.path.normcase(str(workspace.expanduser().resolve()))


def _title_from_task(task: str) -> str:
    return " ".join(task.split())[:80]


def _now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)
