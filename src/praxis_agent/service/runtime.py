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
        self.deleting = False
        self.restart_input_id = None
        self.approvals = ApprovalBroker(self)

    def publish(self, event, turn_id="", agent_id="main", data=None):
        return self.store.append_event(self.id, event, turn_id, agent_id, data)

    def ensure_open(self):
        if self.deleting:
            raise ServiceError("session_deleted", "Session is being deleted", 410)
        if self.closed:
            raise ServiceError("closing", "Service is shutting down", 503)

    def submit(self, command_id, text):
        with self.lock:
            self.ensure_open()

            def accept(connection):
                connection.execute(
                    "INSERT INTO inputs(command_id, session_id, content) VALUES (?, ?, ?)",
                    (command_id, self.id, text),
                )
                connection.execute(
                    "UPDATE inputs SET position = ("
                    "SELECT COALESCE(MAX(COALESCE(position, rowid)), 0) + 1 "
                    "FROM inputs WHERE session_id = ?) WHERE command_id = ?",
                    (self.id, command_id),
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
            self.ensure_open()
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
            self.ensure_open()
            row = connection.execute(
                "SELECT * FROM inputs WHERE command_id = ? AND session_id = ?",
                (command_id, self.id),
            ).fetchone()
            if row is None:
                raise ServiceError("input_not_found", "Input not found", 404)
            if (
                row["consumed_turn_id"] is not None
                or command_id == self.store.sending_input(self.id)
                or (self.run and command_id == self.first_id)
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

    def _editable_input(self, connection, input_id):
        row = connection.execute(
            "SELECT * FROM inputs WHERE command_id = ? AND session_id = ? AND deleted = 0",
            (input_id, self.id),
        ).fetchone()
        if row is None:
            raise ServiceError("input_not_found", "This queued message is no longer available.", 404)
        if (
            row["consumed_turn_id"] is not None
            or input_id == self.store.sending_input(self.id)
            or (self.run and input_id == self.first_id)
        ):
            raise ServiceError(
                "input_consumed", "This message has already been received by the agent.", 409
            )
        return row

    def edit_input(self, command_id, input_id):
        """Move a queued message back to the caller's draft atomically."""
        with self.lock:
            self.ensure_open()

            def edit(connection):
                row = self._editable_input(connection, input_id)
                connection.execute("UPDATE inputs SET deleted = 1 WHERE command_id = ?", (input_id,))
                self.store.pending_changed(self.id)
                return {"content": row["content"]}

            return self.store.command(
                command_id, "edit_input", {"session_id": self.id, "input_id": input_id}, edit
            )[0]

    def reorder_inputs(self, command_id, input_ids):
        with self.lock:
            self.ensure_open()

            def reorder(connection):
                sending = self.store.sending_input(self.id)
                pending = [
                    item["command_id"] for item in self.store.pending(self.id)
                    if item["command_id"] != sending
                ]
                if len(input_ids) != len(pending) or set(input_ids) != set(pending):
                    raise ServiceError("queue_changed", "The message queue changed. Please try again.", 409)
                connection.executemany("UPDATE inputs SET position = ? WHERE command_id = ?", enumerate(input_ids, 1))
                self.store.pending_changed(self.id)
                return {"reordered": True}

            return self.store.command(
                command_id, "reorder_inputs", {"session_id": self.id, "input_ids": input_ids}, reorder
            )[0]

    def send_input_now(self, command_id, input_id):
        """Prioritize one input, then hand over only after the active run settles."""
        with self.lock:
            self.ensure_open()

            def prioritize(connection):
                self._editable_input(connection, input_id)
                connection.execute(
                    "UPDATE inputs SET position = ("
                    "SELECT COALESCE(MIN(COALESCE(position, rowid)), 0) - 1 "
                    "FROM inputs WHERE session_id = ?) WHERE command_id = ?",
                    (self.id, input_id),
                )
                if self.run:
                    connection.execute("UPDATE active_turns SET input_id = ? WHERE session_id = ?", (input_id, self.id))
                self.store.pending_changed(self.id)
                return {"accepted": True}

            result, created = self.store.command(
                command_id, "send_input_now", {"session_id": self.id, "input_id": input_id}, prioritize
            )
            if created:
                if self.run:
                    self.cancel(self.run.id)
                    self.restart_input_id = input_id
                else:
                    self._start_pending()
            return result

    def _start_pending(self):
        pending = self.store.pending(self.id)
        if not pending or self.closed:
            return
        scope = Run("session")
        self.run = scope
        self.first_id = pending[0]["command_id"]
        with self.store.db.transaction() as connection:
            connection.execute(
                "INSERT INTO active_turns(session_id, turn_id, input_id) VALUES (?, ?, ?)",
                (self.id, scope.id, self.first_id),
            )
            self.store.pending_changed(self.id)
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
                    # Keep ownership until deletion/shutdown can retry cleanup.
                    with self.lock:
                        self.closed = True
                else:
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
                restart = self.restart_input_id is not None
                self.restart_input_id = None
                if status == "completed" or restart:
                    self._start_pending()

    def cancel(self, turn_id, *, closing=False):
        with self.lock:
            if not closing:
                self.ensure_open()
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
            self.restart_input_id = None
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
                self.cancel(self.run.id, closing=True)
            worker = self.worker
        if worker:
            worker.join()
        if self.agent:
            self.agent.close()
