from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from praxis_agent.planning.state import RunState, RunStatus
from praxis_agent.sessions.models import (
    SessionNotFoundError,
    SessionRecord,
    SessionStoreError,
    SessionWorkspaceMismatchError,
)

if TYPE_CHECKING:
    from sqlite3 import Connection, Row


_SCHEMA_VERSION = 1
_INTERRUPTED_RUN_ERROR = "Praxis stopped before this run completed."
_INTERRUPTED_TOOL_OUTPUT = (
    "Tool execution was interrupted. Its completion is unknown.\n"
    "Inspect the workspace before deciding whether to retry."
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    model TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS sessions_workspace_updated_idx
    ON sessions(workspace, updated_at DESC);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    original_task TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('running', 'completed', 'failed')
    ),
    error TEXT,
    plan_json TEXT NOT NULL,
    changed_files_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    UNIQUE(session_id, id)
);

CREATE INDEX IF NOT EXISTS runs_session_started_idx
    ON runs(session_id, started_at);

CREATE TABLE IF NOT EXISTS conversation_items (
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    item_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(session_id, sequence),
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY(session_id, run_id)
        REFERENCES runs(session_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS conversation_items_run_idx
    ON conversation_items(run_id);
"""


def default_database_path() -> Path:
    """返回 Praxis 默认的本地状态数据库路径。"""
    return Path.home() / ".praxis" / "state.sqlite3"


class SQLiteSessionStore:
    """使用单个 SQLite 数据库持久化会话、Run 和模型上下文。"""

    def __init__(self, database_path: Path | None = None) -> None:
        path = database_path or default_database_path()
        self.database_path = path.expanduser().resolve()
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SessionStoreError(
                f"cannot create session state directory: {error}"
            ) from error
        self._initialize()

    def create_session(
        self,
        workspace: Path,
        model: str,
        *,
        title: str = "",
    ) -> SessionRecord:
        """创建一段不包含任何 Run 的新会话。"""
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("model must not be empty")

        session_id = uuid4().hex
        workspace_key = _normalize_workspace(workspace)
        now = _now_text()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    id, workspace, model, title, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    workspace_key,
                    normalized_model,
                    title.strip(),
                    now,
                    now,
                ),
            )

        return SessionRecord(
            id=session_id,
            workspace=Path(workspace_key),
            model=normalized_model,
            title=title.strip(),
            created_at=_parse_time(now),
            updated_at=_parse_time(now),
        )

    def get_session(
        self,
        session_id: str,
        *,
        workspace: Path | None = None,
    ) -> SessionRecord:
        """读取指定会话，并可校验它是否属于当前工作区。"""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()

        if row is None:
            raise SessionNotFoundError(f"session not found: {session_id}")

        session = _session_from_row(row)
        if (
            workspace is not None
            and _normalize_workspace(workspace)
            != _normalize_workspace(session.workspace)
        ):
            raise SessionWorkspaceMismatchError(
                "session workspace does not match the current workspace: "
                f"{session.workspace}"
            )
        return session

    def latest_session(self, workspace: Path) -> SessionRecord | None:
        """返回工作区最后更新的会话；不存在时返回 None。"""
        workspace_key = _normalize_workspace(workspace)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM sessions
                WHERE workspace = ?
                ORDER BY updated_at DESC, created_at DESC, id DESC
                LIMIT 1
                """,
                (workspace_key,),
            ).fetchone()
        return None if row is None else _session_from_row(row)

    def start_run(
        self,
        session_id: str,
        run_id: str,
        run_state: RunState,
        user_item: object,
    ) -> None:
        """在一个事务中创建 Run 并保存启动它的用户消息。"""
        if run_state.status is not RunStatus.RUNNING:
            raise ValueError("a new run must have running status")

        encoded_item = _encode_item(user_item)
        now = _now_text()
        with self._transaction() as connection:
            self._require_session(connection, session_id)
            connection.execute(
                """
                INSERT INTO runs (
                    id, session_id, original_task, status, error,
                    plan_json, changed_files_json, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    run_id,
                    session_id,
                    run_state.original_task,
                    run_state.status.value,
                    run_state.error,
                    _encode_plan(run_state),
                    _encode_changed_files(run_state),
                    now,
                ),
            )
            self._append_encoded_items(
                connection,
                session_id,
                run_id,
                [encoded_item],
                created_at=now,
            )
            connection.execute(
                """
                UPDATE sessions
                SET title = CASE WHEN title = '' THEN ? ELSE title END,
                    updated_at = ?
                WHERE id = ?
                """,
                (_title_from_task(run_state.original_task), now, session_id),
            )

    def append_response_items(
        self,
        session_id: str,
        run_id: str,
        items: Sequence[object],
    ) -> None:
        """原样保存一次已完成模型响应中的全部 output items。"""
        encoded_items = [_encode_item(item) for item in items]
        if not encoded_items:
            return

        now = _now_text()
        with self._transaction() as connection:
            self._require_run(connection, session_id, run_id)
            self._append_encoded_items(
                connection,
                session_id,
                run_id,
                encoded_items,
                created_at=now,
            )
            self._touch_session(connection, session_id, now)

    def append_tool_result(
        self,
        session_id: str,
        run_id: str,
        item: object,
        run_state: RunState,
    ) -> None:
        """原子保存工具结果以及工具执行后的最新 RunState。"""
        encoded_item = _encode_item(item)
        now = _now_text()
        with self._transaction() as connection:
            self._require_run(connection, session_id, run_id)
            self._append_encoded_items(
                connection,
                session_id,
                run_id,
                [encoded_item],
                created_at=now,
            )
            self._update_run_state(
                connection,
                session_id,
                run_id,
                run_state,
            )
            self._touch_session(connection, session_id, now)

    def finish_run(
        self,
        session_id: str,
        run_id: str,
        run_state: RunState,
    ) -> None:
        """保存 Run 的 completed 或 failed 终态。"""
        if run_state.status is RunStatus.RUNNING:
            raise ValueError("cannot finish a run with running status")

        now = _now_text()
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = ?, error = ?, plan_json = ?,
                    changed_files_json = ?, finished_at = ?
                WHERE session_id = ? AND id = ?
                """,
                (
                    run_state.status.value,
                    run_state.error,
                    _encode_plan(run_state),
                    _encode_changed_files(run_state),
                    now,
                    session_id,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise SessionStoreError(f"run not found: {run_id}")
            self._touch_session(connection, session_id, now)

    def recover_interrupted_runs(self, session_id: str) -> int:
        """关闭遗留 running Run，并补齐未完成工具调用的协议结果。"""
        now = _now_text()
        with self._transaction() as connection:
            self._require_session(connection, session_id)
            run_rows = connection.execute(
                """
                SELECT id
                FROM runs
                WHERE session_id = ? AND status = 'running'
                ORDER BY started_at
                """,
                (session_id,),
            ).fetchall()

            for run_row in run_rows:
                run_id = str(run_row["id"])
                pending_call_ids = self._pending_call_ids(
                    connection,
                    session_id,
                    run_id,
                )
                interrupted_outputs = [
                    _encode_item(_interrupted_tool_item(call_id))
                    for call_id in pending_call_ids
                ]
                self._append_encoded_items(
                    connection,
                    session_id,
                    run_id,
                    interrupted_outputs,
                    created_at=now,
                )
                connection.execute(
                    """
                    UPDATE runs
                    SET status = 'failed', error = ?, finished_at = ?
                    WHERE session_id = ? AND id = ?
                    """,
                    (_INTERRUPTED_RUN_ERROR, now, session_id, run_id),
                )

            if run_rows:
                self._touch_session(connection, session_id, now)
            return len(run_rows)

    def load_conversation_items(self, session_id: str) -> list[object]:
        """按照权威 sequence 顺序恢复模型上下文。"""
        with self._connection() as connection:
            self._require_session(connection, session_id)
            rows = connection.execute(
                """
                SELECT sequence, payload_json
                FROM conversation_items
                WHERE session_id = ?
                ORDER BY sequence
                """,
                (session_id,),
            ).fetchall()

        items: list[object] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError as error:
                raise SessionStoreError(
                    "invalid conversation item JSON at sequence "
                    f"{row['sequence']}"
                ) from error
            if not isinstance(payload, Mapping):
                raise SessionStoreError(
                    "conversation item is not an object at sequence "
                    f"{row['sequence']}"
                )
            items.append(dict(payload))
        return items

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if version > _SCHEMA_VERSION:
                raise SessionStoreError(
                    "session database was created by a newer Praxis version"
                )
            connection.executescript(_SCHEMA)
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            connection.commit()

    @contextmanager
    def _connection(self) -> Iterator[Connection]:
        connection: Connection | None = None
        try:
            connection = sqlite3.connect(
                self.database_path,
                timeout=30.0,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 30000")
            yield connection
        except sqlite3.Error as error:
            raise SessionStoreError(
                f"session database error: {error}"
            ) from error
        finally:
            if connection is not None:
                connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    @staticmethod
    def _require_session(connection: Connection, session_id: str) -> None:
        row = connection.execute(
            "SELECT 1 FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"session not found: {session_id}")

    @staticmethod
    def _require_run(
        connection: Connection,
        session_id: str,
        run_id: str,
    ) -> None:
        row = connection.execute(
            "SELECT 1 FROM runs WHERE session_id = ? AND id = ?",
            (session_id, run_id),
        ).fetchone()
        if row is None:
            raise SessionStoreError(f"run not found: {run_id}")

    @staticmethod
    def _touch_session(
        connection: Connection,
        session_id: str,
        updated_at: str,
    ) -> None:
        connection.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (updated_at, session_id),
        )

    @staticmethod
    def _update_run_state(
        connection: Connection,
        session_id: str,
        run_id: str,
        run_state: RunState,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE runs
            SET status = ?, error = ?, plan_json = ?,
                changed_files_json = ?
            WHERE session_id = ? AND id = ?
            """,
            (
                run_state.status.value,
                run_state.error,
                _encode_plan(run_state),
                _encode_changed_files(run_state),
                session_id,
                run_id,
            ),
        )
        if cursor.rowcount != 1:
            raise SessionStoreError(f"run not found: {run_id}")

    @staticmethod
    def _append_encoded_items(
        connection: Connection,
        session_id: str,
        run_id: str,
        encoded_items: Sequence[tuple[str, str]],
        *,
        created_at: str,
    ) -> None:
        if not encoded_items:
            return

        next_sequence = int(
            connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) + 1
                FROM conversation_items
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()[0]
        )
        connection.executemany(
            """
            INSERT INTO conversation_items (
                session_id, sequence, run_id, item_type,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    session_id,
                    next_sequence + offset,
                    run_id,
                    item_type,
                    payload_json,
                    created_at,
                )
                for offset, (item_type, payload_json) in enumerate(
                    encoded_items
                )
            ],
        )

    @staticmethod
    def _pending_call_ids(
        connection: Connection,
        session_id: str,
        run_id: str,
    ) -> list[str]:
        rows = connection.execute(
            """
            SELECT sequence, payload_json
            FROM conversation_items
            WHERE session_id = ? AND run_id = ?
            ORDER BY sequence
            """,
            (session_id, run_id),
        ).fetchall()
        pending: dict[str, None] = {}
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError as error:
                raise SessionStoreError(
                    "invalid conversation item JSON at sequence "
                    f"{row['sequence']}"
                ) from error
            if not isinstance(payload, Mapping):
                continue
            call_id = payload.get("call_id")
            if not isinstance(call_id, str):
                continue
            if payload.get("type") == "function_call":
                pending[call_id] = None
            elif payload.get("type") == "function_call_output":
                pending.pop(call_id, None)
        return list(pending)


def _session_from_row(row: Row) -> SessionRecord:
    return SessionRecord(
        id=str(row["id"]),
        workspace=Path(row["workspace"]),
        model=str(row["model"]),
        title=str(row["title"]),
        created_at=_parse_time(row["created_at"]),
        updated_at=_parse_time(row["updated_at"]),
    )


def _encode_item(item: object) -> tuple[str, str]:
    if isinstance(item, Mapping):
        payload = dict(item)
    else:
        model_dump = getattr(item, "model_dump", None)
        if not callable(model_dump):
            raise SessionStoreError(
                "conversation item must be a mapping or SDK model"
            )
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
