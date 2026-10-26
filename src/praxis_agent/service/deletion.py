"""One deletion per session, with durable intent and startup-only recovery."""

import logging
from concurrent.futures import Future
from contextlib import nullcontext
from threading import Thread

from .errors import ServiceError

logger = logging.getLogger(__name__)


class SessionDeletions:
    def __init__(self, manager):
        self.manager = manager
        # Loaded before the service accepts requests. Never restore these Agents.
        self.marked = set(manager.store.deleting_sessions())
        self.pending = {}
        self.recovery = None

    def start_recovery(self):
        if self.marked:
            self.recovery = Thread(target=self._recover, name="praxis-delete-recovery", daemon=True)
            self.recovery.start()

    def _recover(self):
        for session_id in tuple(self.marked):
            try:
                self.delete(session_id)
            except Exception:
                logger.exception("Could not finish deleting session %s", session_id)

    def delete(self, session_id):
        manager = self.manager
        with manager.lock:
            if manager.closed:
                raise ServiceError("closing", "Service is shutting down", 503)
            future = self.pending.get(session_id)
            owner = future is None
            if owner:
                runtime = manager.runtimes.get(session_id)
                # Admission and the durable marker change under the same session lock.
                with runtime.lock if runtime else nullcontext():
                    if not manager.store.mark_for_deletion(session_id):
                        return {"deleted": True, "session_id": session_id}
                    self.marked.add(session_id)
                    if runtime:
                        runtime.deleting = runtime.closed = True
                    future = Future()
                    self.pending[session_id] = future
        if not owner:
            return future.result()
        try:
            # No manager/session lock while cancellation joins workers and commands.
            if runtime:
                runtime.close()
            manager.store.delete_session(session_id)
            with manager.lock:
                manager.runtimes.pop(session_id, None)
                self.marked.discard(session_id)
            result = {"deleted": True, "session_id": session_id}
            future.set_result(result)
            return result
        except Exception as error:
            # Keep the marker and closed runtime so a later request can retry cleanup.
            logger.exception("Could not finish deleting session %s", session_id)
            future.set_exception(error)
            raise
        finally:
            with manager.lock:
                self.pending.pop(session_id, None)

    def wait(self):
        if self.recovery:
            self.recovery.join()
        with self.manager.lock:
            pending = list(self.pending.values())
        for future in pending:
            try:
                future.result()
            except Exception:
                pass  # The durable marker remains available on the next startup.
