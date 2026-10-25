"""Own one main execution flow; HTTP connections never own or cancel it."""

from __future__ import annotations

import logging
from threading import RLock, Thread

from praxis_agent.execution import ExecutionStopped, Run

from .approvals import ApprovalBroker
from .errors import ServiceError
from .events import ServiceEventSink
from .inputs import TurnInputs

logger = logging.getLogger(__name__)


class SessionRuntime:
    def __init__(self, record, store, agent_factory, bypass_approvals=False):
        self.id = record.id
        self.record = record
        self.store = store
        self.factory = agent_factory
        self.bypass_approvals = bypass_approvals
        self.lock = RLock()
        self.run = None
        self.worker = None
        self.agent = None
        self.closed = False
        self.approvals = ApprovalBroker(self)

    def publish(self, event, turn_id="", agent_id="main", data=None):
        return self.store.append_event(self.id, event, turn_id, agent_id, data)

    def submit(self, command_id, text):
        with self.lock:
            if self.closed:
                raise ServiceError("closing", "Service is shutting down", 503)

            def accept(connection):
                connection.execute(
                    "INSERT INTO inputs(command_id, session_id, content) VALUES (?, ?, ?)",
                    (command_id, self.id, text),
                )
                self.store.pending_changed(self.id)
                return {"command_id": command_id, "accepted": True}

            result, created = self.store.command(
                command_id, "input", {"session_id": self.id, "text": text}, accept
            )
            if created and self.run is None:
                self._start_pending()
            return result

    def resume(self, command_id):
        with self.lock:
            if self.closed:
                raise ServiceError("closing", "Service is shutting down", 503)
            result, created = self.store.command(
                command_id,
                "resume",
                {"session_id": self.id},
                lambda _: {"command_id": command_id, "accepted": True},
            )
            if created and self.run is None:
                self._start_pending()
            return result

    def delete_input(self, command_id):
        with self.lock, self.store.db.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM inputs WHERE command_id = ? AND session_id = ?",
                (command_id, self.id),
            ).fetchone()
            if row is None:
                raise ServiceError("input_not_found", "Input not found", 404)
            if row["consumed_turn_id"] is not None or (
                self.run and getattr(self, "first_id", None) == command_id
            ):
                raise ServiceError(
                    "input_consumed", "Input is already being processed", 409
                )
            if not row["deleted"]:
                connection.execute(
                    "UPDATE inputs SET deleted = 1 WHERE command_id = ?", (command_id,)
                )
                self.store.pending_changed(self.id)
            return {"deleted": True}

    def _start_pending(self):
        pending = self.store.pending(self.id)
        if not pending or self.closed:
            return
        scope = Run("session")
        self.run = scope
        self.first_id = pending[0]["command_id"]
        with self.store.db.transaction() as connection:
            connection.execute(
                "INSERT INTO active_turns(session_id, turn_id) VALUES (?, ?)",
                (self.id, scope.id),
            )
            self.store.updated(self.id)
        self.worker = Thread(
            target=self._execute,
            args=(scope, pending[0]),
            name=f"praxis-{self.id[:8]}",
            daemon=True,
        )
        self.worker.start()

    def _execute(self, scope, first):
        status, error = "completed", None
        try:
            if self.agent is None:
                self.agent = self.factory(
                    self.record.workspace,
                    event_sink=ServiceEventSink(self),
                    approve_tool_call=self.approvals.request,
                    bypass_approvals=self.bypass_approvals,
                    resume_session_id=self.id,
                    session_store=self.store.sessions,
                )
            self.agent.input_source = TurnInputs(self, first["command_id"])
            scope.execute(lambda: self.agent.run(first["content"]))
        except ExecutionStopped:
            status = "cancelled"
        except Exception as failure:
            status, error = "failed", str(failure)
            logger.exception("Session %s failed", self.id)
            if self.agent is not None:
                try:
                    self.agent.close()
                except Exception:
                    logger.exception("Could not close failed session %s", self.id)
                self.agent = None
        finally:
            with self.lock:
                self.run = None
                with self.store.db.transaction() as connection:
                    # Initialization failures also need a durable turn result.
                    connection.execute(
                        "INSERT OR IGNORE INTO runs VALUES (?, ?, ?, ?, ?, '[]', '[]', "
                        "strftime('%Y-%m-%dT%H:%M:%f+00:00','now'), strftime('%Y-%m-%dT%H:%M:%f+00:00','now'))",
                        (scope.id, self.id, first["content"], status, error),
                    )
                    # Also settle a run whose normal finalization failed before
                    # committing. The next Agent reloads durable history.
                    connection.execute(
                        "UPDATE runs SET status = ?, error = ?, finished_at = "
                        "strftime('%Y-%m-%dT%H:%M:%f+00:00','now') WHERE id = ? AND status = 'running'",
                        (status, error, scope.id),
                    )
                    connection.execute(
                        "DELETE FROM active_turns WHERE session_id = ? AND turn_id = ?",
                        (self.id, scope.id),
                    )
                    self.publish(
                        "TurnSettled", scope.id, data={"status": status, "error": error}
                    )
                    self.store.pending_changed(self.id)
                    self.store.updated(self.id)
                # Inputs admitted after the final check are picked up here. A stop
                # or failure always retains them for an explicit user action.
                if status == "completed":
                    self._start_pending()

    def cancel(self, turn_id):
        with self.lock:
            scope = self.run
            if scope is None or scope.id != turn_id:
                with self.store.db.connection() as connection:
                    row = connection.execute(
                        "SELECT status FROM runs WHERE session_id = ? AND id = ?",
                        (self.id, turn_id),
                    ).fetchone()
                if row is None:
                    raise ServiceError("turn_not_found", "Turn does not exist", 404)
                return {"turn_id": turn_id, "status": row["status"]}
            if not scope.cancelled.is_set():
                with self.store.db.transaction() as connection:
                    connection.execute(
                        "UPDATE active_turns SET cancel_requested = 1 WHERE session_id = ?",
                        (self.id,),
                    )
                    self.store.updated(self.id)
                scope.cancel()
            return {"turn_id": turn_id, "cancel_requested": True}

    def close(self):
        with self.lock:
            self.closed = True
            if self.run:
                self.cancel(self.run.id)
            worker = self.worker
        if worker:
            worker.join()
        if self.agent:
            self.agent.close()
