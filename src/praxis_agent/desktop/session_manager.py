from __future__ import annotations

import logging
import os
from concurrent.futures import CancelledError, Future, InvalidStateError
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Lock, RLock, Thread
from uuid import uuid4

from praxis_agent.agent import create_default_agent
from praxis_agent.config import ModelConfig
from praxis_agent.execution import ExecutionStopped, Run, current_handle, current_run, run_operation
from praxis_agent.sessions import SQLiteSessionStore

from .event_adapter import DesktopEventSink, Transcript, json_value
from .protocol import ProtocolError, required_string

logger = logging.getLogger(__name__)


def workspace_path(params: dict, *, optional=False) -> Path | None:
    if optional and params.get("workspace") is None:
        return None
    path = Path(required_string(params, "workspace")).expanduser().resolve()
    if not path.is_dir():
        raise ProtocolError("invalid_workspace", f"Workspace is not a directory: {path}")
    return path


class SessionRuntime:
    def __init__(self, record, send):
        self.record = record
        self.send = send
        self.lock = RLock()
        self.agent = None
        self.run: Run | None = None
        self.worker: Thread | None = None
        self.input: SimpleQueue[str] = SimpleQueue()
        self.paused_inputs: list[str] = []
        self.approvals: dict[str, tuple[dict, Future]] = {}
        self.transcript = Transcript()
        self.sequence = 0
        self.status = "idle"
        self.title = record.title
        self.updated_at = record.updated_at.isoformat()
        self.closed = False

    def summary(self):
        return {"id": self.record.id, "title": self.title, "workspace": str(self.record.workspace),
                "model": self.record.model, "updated_at": self.updated_at, "status": self.status}

    def publish(self, event, turn_id="", agent_id="main", data=None):
        with self.lock:
            data = {} if data is None else data
            self.sequence += 1
            self.transcript.consume(event, turn_id, agent_id, data)
            self.send({"event": event, "session_id": self.record.id, "turn_id": turn_id,
                       "agent_id": agent_id, "sequence": self.sequence, "data": data})

    def updated(self):
        self.publish("SessionUpdated", self.run.id if self.run else "", data={"session": self.summary()})

    def snapshot(self):
        with self.lock:
            return {"session": self.summary(), "messages": deepcopy(self.transcript.messages),
                    "approvals": [deepcopy(data) for data, _ in self.approvals.values()],
                    "pending_inputs": list(self.paused_inputs),
                    "sequence": self.sequence}

    def approve(self, tool_name, arguments):
        if current_handle() is None:
            return run_operation("approval", lambda handle: self.approve(tool_name, arguments))
        handle = current_handle()
        approval_id = uuid4().hex
        future = Future()
        scope = current_run()
        with self.lock:
            if self.closed or self.run is None or self.run.cancelled.is_set():
                raise ExecutionStopped("approval cancelled")
            turn_id = self.run.id
            agent_id = str(arguments.get("agent_id", "main"))
            data = {"approval_id": approval_id, "tool_name": tool_name,
                    "arguments": json_value(dict(arguments)), "turn_id": turn_id, "agent_id": agent_id}
            self.approvals[approval_id] = (data, future)
            self.publish("ApprovalPending", turn_id, agent_id, data)
        unsubscribe = handle.on_cancel(future.cancel)
        # Also bind to the owning root Run in case an adapter invokes approval without a child handle.
        unsubscribe_root = self.run.on_cancel(future.cancel) if self.run is not scope else lambda: None
        try:
            result = future.result()
            handle.check()
            return result
        except CancelledError as error:
            raise ExecutionStopped("approval cancelled") from error
        finally:
            unsubscribe()
            unsubscribe_root()
            with self.lock:
                self.approvals.pop(approval_id, None)
                self.publish("ApprovalClosed", turn_id, agent_id, {"approval_id": approval_id})

    def respond(self, params):
        approval_id = required_string(params, "approval_id")
        approved = params.get("approved")
        if not isinstance(approved, bool):
            raise ProtocolError("invalid_params", "approved must be a boolean")
        with self.lock:
            entry = self.approvals.get(approval_id)
            if entry is None:
                raise ProtocolError("approval_not_found", "Approval is no longer pending")
            try:
                entry[1].set_result(approved)
            except InvalidStateError as error:
                raise ProtocolError("approval_not_found", "Approval is no longer pending") from error
        return {}

    def start(self, text: str, accepted):
        with self.lock:
            if self.closed:
                raise ProtocolError("closing", "Session is closing")
            if self.run is not None:
                raise ProtocolError("session_busy", "This session already has an active turn")
            scope = Run("desktop-session")
            self.run = scope
            self.status = "running"
            self.updated_at = datetime.now(UTC).isoformat()
            if not self.title:
                self.title = " ".join(text.split())[:80]
            self.agent.user_input = self.input
            if text in self.paused_inputs:
                self.paused_inputs.remove(text)
            self.worker = Thread(target=self._execute, args=(scope, text),
                                 name=f"praxis-session-{self.record.id[:8]}", daemon=True)
            # Register first, enqueue acceptance second, and only then dispatch execution.
            accepted({"turn_id": scope.id})
            self.updated()
            self.worker.start()

    def _execute(self, scope, text):
        status, error_message = "idle", None
        try:
            scope.execute(lambda: self.agent.run(text))
        except ExecutionStopped:
            status = "cancelled"
        except Exception as error:
            status, error_message = "failed", str(error)
            logger.exception("Desktop turn failed for session %s", self.record.id)
        finally:
            # Run.execute returns only after all owned handles have completed cleanup.
            with self.lock:
                self.run = None
                self.status = status
                self.updated_at = datetime.now(UTC).isoformat()
                paused = []
                while True:
                    try:
                        paused.append(self.input.get_nowait())
                    except Empty:
                        break
                if paused:
                    self.paused_inputs.extend(paused)
                    self.publish("InputsPaused", scope.id, data={"texts": list(self.paused_inputs)})
                data = {"status": status}
                if error_message:
                    data["error"] = error_message
                self.publish("TurnSettled", scope.id, data=data)
                self.updated()

    def enqueue(self, text):
        with self.lock:
            if self.run is None or self.status != "running":
                raise ProtocolError("session_not_running", "Start a turn before sending supplementary input")
            self.input.put(text)
            if text in self.paused_inputs:
                self.paused_inputs.remove(text)
        return {"queued": True}

    def stop(self):
        with self.lock:
            scope = self.run
            if scope is None:
                return {"stopping": False}
            self.status = "stopping"
            self.updated()
        scope.cancel()
        return {"stopping": True}

    def cancel_for_shutdown(self):
        with self.lock:
            self.closed = True
        self.stop()

    def close(self):
        self.cancel_for_shutdown()
        worker = self.worker
        if worker is not None:
            worker.join()
        if self.agent is not None:
            self.agent.close()


class SessionManager:
    def __init__(self, send, *, database_path=None, agent_factory=create_default_agent):
        self.send = send
        self.store = SQLiteSessionStore(database_path)
        self.agent_factory = agent_factory
        self.lock = Lock()
        self.sessions: dict[str, SessionRuntime] = {}
        self.loading: dict[str, Future] = {}
        self.closing = False

    def initialize(self):
        try:
            config = ModelConfig.from_env()
            return {"protocol_version": 1, "model": config.model, "configured": True}
        except ValueError as error:
            return {"protocol_version": 1, "model": os.environ.get("ADA_MODEL", "").strip() or "auto(free)",
                    "configured": False, "configuration_error": str(error)}

    def _load(self, record):
        with self.lock:
            if self.closing:
                raise ProtocolError("closing", "Server is shutting down")
            if record.id in self.sessions:
                return self.sessions[record.id]
            future = self.loading.get(record.id)
            owner = future is None
            if owner:
                future = Future()
                self.loading[record.id] = future
        if not owner:
            return future.result()
        runtime = SessionRuntime(record, self.send)
        try:
            runtime.agent = self.agent_factory(
                record.workspace, event_sink=DesktopEventSink(runtime), approve_tool_call=runtime.approve,
                resume_session_id=record.id, session_database_path=self.store.database_path,
            )
            runtime.agent.user_input = runtime.input
            runtime.transcript = Transcript(self.store.load_conversation_items(record.id))
            latest = next((item for item in self.store.list_sessions(record.workspace, limit=1000)
                           if item.id == record.id), None)
            if latest and latest.latest_run_status and latest.latest_run_status.value in {"failed", "cancelled"}:
                runtime.status = latest.latest_run_status.value
            with self.lock:
                if self.closing:
                    raise ProtocolError("closing", "Server is shutting down")
                self.sessions[record.id] = runtime
            future.set_result(runtime)
            return runtime
        except BaseException as error:
            try:
                runtime.close()
            except Exception:
                logger.exception("Could not close failed session initialization for %s", record.id)
            finally:
                # Every concurrent opener must be released even when cleanup also fails.
                future.set_exception(error)
            raise
        finally:
            with self.lock:
                self.loading.pop(record.id, None)

    def create(self, params):
        workspace = workspace_path(params)
        if self.agent_factory is create_default_agent:
            # Invalid configuration should not leave an empty session record behind.
            ModelConfig.from_env()
        model = self.initialize()["model"]
        record = self.store.create_session(workspace, model)
        runtime = self._load(record)
        with runtime.lock:
            runtime.updated()
            return {"session": runtime.summary()}

    def open(self, params):
        session_id = required_string(params, "session_id")
        workspace = workspace_path(params, optional=True)
        record = self.store.get_session(session_id, workspace=workspace)
        return self._load(record).snapshot()

    def list(self, params):
        workspace = workspace_path(params)
        sessions = []
        for item in self.store.list_sessions(workspace, limit=200):
            with self.lock:
                runtime = self.sessions.get(item.id)
            if runtime is not None:
                with runtime.lock:
                    sessions.append(runtime.summary())
            else:
                record = self.store.get_session(item.id)
                status = item.latest_run_status.value if item.latest_run_status else "idle"
                # Persisted running means interrupted until the session is recovered on open.
                if status == "running":
                    status = "failed"
                if status == "completed":
                    status = "idle"
                sessions.append({"id": record.id, "title": record.title, "workspace": str(record.workspace),
                                 "model": record.model, "updated_at": record.updated_at.isoformat(), "status": status})
        return {"sessions": sessions}

    def require(self, params):
        session_id = required_string(params, "session_id")
        with self.lock:
            if self.closing:
                raise ProtocolError("closing", "Server is shutting down")
            runtime = self.sessions.get(session_id)
        if runtime is None:
            raise ProtocolError("session_not_open", "Open the session before sending commands")
        return runtime

    def begin_shutdown(self):
        with self.lock:
            self.closing = True
            sessions = tuple(self.sessions.values())
        for runtime in sessions:
            runtime.cancel_for_shutdown()

    def close(self):
        self.begin_shutdown()
        with self.lock:
            sessions = tuple(self.sessions.values())
        for runtime in sessions:
            runtime.close()
