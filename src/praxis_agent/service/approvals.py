from concurrent.futures import CancelledError, Future
from uuid import uuid4

from praxis_agent.execution import ExecutionStopped, current_handle, run_operation

from .errors import ServiceError
from .store import encode


class ApprovalBroker:
    """One authoritative decision shared by every client watching a session."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.waiters = {}

    def request(self, tool_name, arguments):
        handle = current_handle()
        if handle is None:
            return run_operation(
                "approval", lambda _: self.request(tool_name, arguments)
            )
        runtime, approval_id, future = self.runtime, uuid4().hex, Future()
        with runtime.lock:
            if runtime.run is None or runtime.closed:
                raise ExecutionStopped("approval cancelled")
            scope = runtime.run
            scope.check()
            turn_id, agent_id = scope.id, str(arguments.get("agent_id", "main"))
            data = dict(
                approval_id=approval_id,
                tool_name=tool_name,
                arguments=dict(arguments),
                turn_id=turn_id,
                agent_id=agent_id,
                status="pending",
            )
            with runtime.store.db.transaction() as connection:
                connection.execute(
                    "INSERT INTO approvals VALUES (?, ?, ?, ?, ?, ?, 'pending')",
                    (
                        approval_id,
                        runtime.id,
                        turn_id,
                        agent_id,
                        tool_name,
                        encode(arguments),
                    ),
                )
                runtime.publish("ApprovalPending", turn_id, agent_id, data)
            self.waiters[approval_id] = future
        unsubscribe = handle.on_cancel(future.cancel)
        unsubscribe_root = scope.on_cancel(future.cancel)
        try:
            result = future.result()
            scope.check()
            handle.check()
            return result
        except CancelledError as error:
            raise ExecutionStopped("approval cancelled") from error
        finally:
            unsubscribe()
            unsubscribe_root()
            with runtime.lock:
                self.waiters.pop(approval_id, None)
                self._resolve(approval_id, "cancelled")

    def _resolve(self, approval_id, status):
        runtime = self.runtime
        with runtime.store.db.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM approvals WHERE id = ? AND session_id = ?",
                (approval_id, runtime.id),
            ).fetchone()
            if row is None:
                raise ServiceError("approval_not_found", "Approval does not exist", 404)
            if row["status"] == "pending":
                connection.execute(
                    "UPDATE approvals SET status = ? WHERE id = ? AND status = 'pending'",
                    (status, approval_id),
                )
                runtime.publish(
                    "ApprovalClosed",
                    row["turn_id"],
                    row["agent_id"],
                    {"approval_id": approval_id, "status": status},
                )
            else:
                status = row["status"]
            return {"approval_id": approval_id, "status": status}

    def decide(self, approval_id, approved):
        with self.runtime.lock:
            self.runtime.ensure_open()
            scope = self.runtime.run
            desired = "approved" if approved else "denied"
            if scope is None or scope.cancelled.is_set():
                desired = "cancelled"
            result = self._resolve(approval_id, desired)
            future = self.waiters.get(approval_id)
            if future is not None and not future.done():
                if result["status"] == "cancelled":
                    future.cancel()
                else:
                    future.set_result(result["status"] == "approved")
            return result
