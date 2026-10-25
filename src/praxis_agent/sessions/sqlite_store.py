from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from praxis_agent.context.operations import ContextOperation
from praxis_agent.planning.state import RunState, RunStatus
from praxis_agent.sessions.models import (
    AmbiguousSessionIdError,
    SessionNotFoundError,
    SessionRecord,
    SessionStoreError,
    SessionSummary,
    SessionWorkspaceMismatchError,
)

if TYPE_CHECKING:
    from sqlite3 import Connection


from .codec import (
    _encode_changed_files,
    _encode_item,
    _encode_plan,
    _interrupted_tool_item,
    _normalize_workspace,
    _now_text,
    _parse_time,
    _session_from_row,
    _session_summary_from_row,
    _title_from_task,
)
from .database import Database, default_database_path
from .schema import _INTERRUPTED_RUN_ERROR, _SCHEMA, _SCHEMA_VERSION


class SQLiteSessionStore:
    """使用单个 SQLite 数据库持久化会话、Run 和模型上下文。"""

    def __init__(self, database_path: Path | None = None, *, database: Database | None = None):
        self.database = database or Database(database_path or default_database_path())
        self.database_path = self.database.path
        self._initialize()

    def _connection(self):
        return self.database.connection()

    def _transaction(self):
        return self.database.transaction()

    atomic = _transaction

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

    def resolve_session(
        self,
        reference: str,
        *,
        workspace: Path,
    ) -> SessionRecord:
        """使用完整 ID 或唯一 ID 前缀解析当前会话。"""
        normalized_reference = reference.strip()
        if not normalized_reference:
            raise ValueError("session ID must not be empty")

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM sessions
                WHERE substr(id, 1, length(?)) = ? COLLATE NOCASE
                ORDER BY updated_at DESC, created_at DESC, id DESC
                LIMIT 2
                """,
                (normalized_reference, normalized_reference),
            ).fetchall()

        if not rows:
            raise SessionNotFoundError(
                f"session not found: {normalized_reference}"
            )
        if len(rows) > 1:
            raise AmbiguousSessionIdError(
                "session ID prefix is ambiguous: "
                f"{normalized_reference}"
            )

        session = _session_from_row(rows[0])
        if _normalize_workspace(workspace) != _normalize_workspace(
            session.workspace
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

    def list_sessions(
        self,
        workspace: Path,
        *,
        limit: int = 20,
    ) -> list[SessionSummary]:
        """按照最近更新时间列出工作区的会话摘要。"""
        if limit < 1:
            raise ValueError("session list limit must be at least 1")

        workspace_key = _normalize_workspace(workspace)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    sessions.id,
                    sessions.title,
                    sessions.updated_at,
                    (
                        SELECT runs.status
                        FROM runs
                        WHERE runs.session_id = sessions.id
                        ORDER BY runs.started_at DESC, runs.id DESC
                        LIMIT 1
                    ) AS latest_run_status
                FROM sessions
                WHERE sessions.workspace = ?
                ORDER BY
                    sessions.updated_at DESC,
                    sessions.created_at DESC,
                    sessions.id DESC
                LIMIT ?
                """,
                (workspace_key, limit),
            ).fetchall()
        return [_session_summary_from_row(row) for row in rows]

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
                WHERE session_id = ? AND id = ? AND status = 'running'
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
            if row["sequence"] != len(items) + 1:
                raise SessionStoreError("conversation sequence is not contiguous")
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

    def append_context_operation(
        self, session_id: str, operation: ContextOperation,
    ) -> None:
        """追加视图操作，绝不覆盖 conversation_items 中的原始正文。"""
        now = _now_text()
        with self._transaction() as connection:
            self._require_session(connection, session_id)
            self._insert_context_operation(connection, session_id, operation, now)
            self._touch_session(connection, session_id, now)

    @staticmethod
    def _insert_context_operation(
        connection: Connection, session_id: str,
        operation: ContextOperation, created_at: str,
    ) -> None:
        sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM context_view_operations "
            "WHERE session_id = ?", (session_id,),
        ).fetchone()[0]
        if operation.operation_id != f"op{sequence}":
            raise SessionStoreError("context operations changed; resume the session again")
        connection.execute(
            "INSERT INTO context_view_operations "
            "(session_id, sequence, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (session_id, sequence, json.dumps(operation.to_dict(), ensure_ascii=False), created_at),
        )

    def load_context_operations(self, session_id: str) -> list[ContextOperation]:
        """按追加顺序重放操作，使恢复会话后继续使用相同视图。"""
        with self._connection() as connection:
            self._require_session(connection, session_id)
            rows = connection.execute(
                "SELECT sequence, payload_json FROM context_view_operations "
                "WHERE session_id = ? ORDER BY sequence", (session_id,),
            ).fetchall()
        operations = []
        for row in rows:
            try:
                operation = ContextOperation.from_dict(json.loads(row["payload_json"]))
            except (TypeError, ValueError) as error:
                raise SessionStoreError("invalid context operation JSON") from error
            if row["sequence"] != len(operations) + 1 or operation.operation_id != f"op{row['sequence']}":
                raise SessionStoreError("invalid context operation sequence")
            operations.append(operation)
        return operations

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
            "SELECT 1 FROM runs WHERE session_id = ? AND id = ? AND status = 'running'",
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
            SET error = ?, plan_json = ?,
                changed_files_json = ?
            WHERE session_id = ? AND id = ? AND status = 'running'
            """,
            (
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
