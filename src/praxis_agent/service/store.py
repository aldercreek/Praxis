"""Durable commands and event journal; no network or Agent lifecycle here."""

from __future__ import annotations

import json

from praxis_agent.sessions import SQLiteSessionStore

from .errors import ServiceError
from .history import session_view_events
from .schema import SCHEMA
from .transcript import Transcript


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class ServiceStore:
    def __init__(self, sessions: SQLiteSessionStore):
        self.sessions = sessions
        self.db = sessions.database
        with self.db.connection() as connection:
            connection.executescript(SCHEMA)
            connection.commit()

    def command(self, command_id, kind, payload, operation):
        """Dedupe the mutation and its reply in the same transaction."""
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM commands WHERE id = ?", (command_id,)
            ).fetchone()
            if row:
                if row["kind"] != kind or row["payload"] != encode(payload):
                    raise ServiceError(
                        "command_conflict",
                        "Command ID was used for different content",
                        409,
                    )
                return json.loads(row["result"]), False
            # The FK from inputs requires the command to exist before operation().
            connection.execute(
                "INSERT INTO commands VALUES (?, ?, ?, '{}')",
                (command_id, kind, encode(payload)),
            )
            result = operation(connection)
            connection.execute(
                "UPDATE commands SET result = ? WHERE id = ?",
                (encode(result), command_id),
            )
            return result, True

    def append_event(self, session_id, event, turn_id="", agent_id="main", data=None):
        with self.db.transaction() as connection:
            data = data or {}
            self._record_tool(connection, session_id, event, turn_id, agent_id, data)
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM events WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            frame = dict(
                event=event,
                session_id=session_id,
                sequence=sequence,
                turn_id=turn_id,
                agent_id=agent_id,
                data=data or {},
            )
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?)",
                (session_id, sequence, encode(frame)),
            )
            return frame

    @staticmethod
    def _record_tool(connection, session_id, event, turn_id, agent_id, data):
        if event == "TurnSettled" and data.get("status") in {"cancelled", "failed"}:
            connection.execute(
                "UPDATE tool_calls SET status = ?, result = ? WHERE session_id = ? AND turn_id = ? "
                "AND status IN ('pending', 'running')",
                (
                    data["status"],
                    encode(
                        {
                            "success": False,
                            "output": "Execution ended before a result was recorded; completion is unknown.",
                        }
                    ),
                    session_id,
                    turn_id,
                ),
            )
            return
        if event not in {
            "ToolCallPrepared",
            "ToolStarted",
            "ToolFinished",
            "ApprovalResolved",
        }:
            return
        call_id = data["call_id"]
        status = {
            "ToolCallPrepared": "pending",
            "ToolStarted": "running",
            "ApprovalResolved": "pending" if data.get("approved") else "denied",
        }.get(event)
        if event == "ToolFinished":
            status = "completed" if data["result"].get("success") else "failed"
        connection.execute(
            "INSERT INTO tool_calls(session_id, agent_id, call_id, turn_id, tool_name, arguments, status, result) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(session_id, agent_id, call_id) DO UPDATE SET "
            "status = excluded.status, result = COALESCE(excluded.result, tool_calls.result), "
            "arguments = COALESCE(excluded.arguments, tool_calls.arguments)",
            (
                session_id,
                agent_id,
                call_id,
                turn_id,
                data.get("tool_name", ""),
                encode(data["arguments"]) if "arguments" in data else None,
                status,
                encode(data["result"]) if "result" in data else None,
            ),
        )

    def events(self, session_id, after=0, limit=256):
        with self.db.connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM events WHERE session_id = ? AND sequence > ? ORDER BY sequence LIMIT ?",
                (session_id, after, limit),
            ).fetchall()
            return [json.loads(row[0]) for row in rows]

    def pending(self, session_id):
        with self.db.connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT command_id, content FROM inputs WHERE session_id = ? "
                    "AND consumed_turn_id IS NULL AND deleted = 0 ORDER BY rowid",
                    (session_id,),
                )
            ]

    def pending_changed(self, session_id):
        self.append_event(
            session_id, "InputsChanged", data={"inputs": self.pending(session_id)}
        )

    def summary(self, session_id):
        record = self.sessions.get_session(session_id)
        with self.db.connection() as connection:
            active = connection.execute(
                "SELECT * FROM active_turns WHERE session_id = ?", (session_id,)
            ).fetchone()
            latest = connection.execute(
                "SELECT id, status, error FROM runs WHERE session_id = ? ORDER BY started_at DESC, id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return dict(
            id=record.id,
            title=record.title,
            workspace=str(record.workspace),
            model=record.model,
            updated_at=record.updated_at.isoformat(),
            active_turn=dict(
                id=active["turn_id"], cancel_requested=bool(active["cancel_requested"])
            )
            if active
            else None,
            latest_turn=dict(latest) if latest else None,
        )

    def updated(self, session_id):
        self.append_event(
            session_id, "SessionUpdated", data={"session": self.summary(session_id)}
        )

    def snapshot(self, session_id):
        # One short unit of work freezes the projection and its sequence together.
        # Replay is local SQL only; never hold this transaction while streaming.
        with self.db.transaction(write=False) as connection:
            summary = self.summary(session_id)
            frames = [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT payload FROM events WHERE session_id = ? ORDER BY sequence",
                    (session_id,),
                )
            ]
            view_events = session_view_events(connection, session_id, frames)
            transcript = Transcript()
            for frame in view_events:
                transcript.consume(
                    frame["event"], frame["turn_id"], frame["agent_id"], frame["data"]
                )
            approvals = [
                self.approval_dict(row)
                for row in connection.execute(
                    "SELECT * FROM approvals WHERE session_id = ? AND status = 'pending'",
                    (session_id,),
                )
            ]
            return dict(
                session=summary,
                messages=transcript.messages,
                approvals=approvals,
                pending_inputs=self.pending(session_id),
                sequence=frames[-1]["sequence"] if frames else 0,
                # Typed terminal presentation is reconstructed without creating an Agent.
                view_events=view_events,
            )

    @staticmethod
    def approval_dict(row):
        return dict(
            approval_id=row["id"],
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            agent_id=row["agent_id"],
            tool_name=row["tool_name"],
            arguments=json.loads(row["arguments"]),
            status=row["status"],
        )

    def recover(self):
        """Called only during startup, after acquiring the process ownership lock."""
        with self.db.transaction() as connection:
            rows = connection.execute(
                "SELECT session_id, turn_id FROM active_turns UNION "
                "SELECT session_id, id AS turn_id FROM runs WHERE status = 'running'",
            ).fetchall()
            for row in rows:
                session_id, turn_id = row["session_id"], row["turn_id"]
                saved = connection.execute(
                    "SELECT status FROM runs WHERE id = ?", (turn_id,)
                ).fetchone()
                unfinished = saved is None or saved["status"] == "running"
                status = "failed" if unfinished else saved["status"]
                if unfinished:
                    self.sessions.recover_interrupted_runs(session_id)
                # A crash while initializing the Agent may precede start_run().
                connection.execute(
                    "INSERT OR IGNORE INTO runs VALUES (?, ?, '', 'failed', ?, '[]', '[]', "
                    "strftime('%Y-%m-%dT%H:%M:%f+00:00','now'), strftime('%Y-%m-%dT%H:%M:%f+00:00','now'))",
                    (turn_id, session_id, "Backend exited before this turn completed."),
                )
                approvals = connection.execute(
                    "SELECT id FROM approvals WHERE session_id = ? AND status = 'pending'",
                    (session_id,),
                ).fetchall()
                connection.execute(
                    "UPDATE approvals SET status = 'cancelled' WHERE session_id = ? AND status = 'pending'",
                    (session_id,),
                )
                for approval in approvals:
                    self.append_event(
                        session_id,
                        "ApprovalClosed",
                        turn_id,
                        data={"approval_id": approval["id"], "status": "cancelled"},
                    )
                if unfinished:
                    self.append_event(
                        session_id,
                        "TurnFailed",
                        turn_id,
                        data={
                            "error_type": "BackendExited",
                            "message": "Backend exited. Unfinished tool outcomes are unknown; inspect before retrying.",
                        },
                    )
                connection.execute(
                    "DELETE FROM active_turns WHERE session_id = ?", (session_id,)
                )
                self.append_event(
                    session_id, "TurnSettled", turn_id, data={"status": status}
                )
                self.pending_changed(session_id)
                self.updated(session_id)
