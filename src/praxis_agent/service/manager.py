from __future__ import annotations

import os
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from threading import RLock

from praxis_agent.config import load_config
from praxis_agent.factory import create_default_agent
from praxis_agent.observability import Observability

from .errors import ServiceError, required_text
from .deletion import SessionDeletions
from .runtime import SessionRuntime


class SessionManager:
    def __init__(self, store, *, agent_factory=create_default_agent, mcp=None, config_path=None):
        self.store = store
        self.mcp = mcp
        self.config = load_config(config_path)
        self.observability = Observability(self.config.observability, store.db.path.parent / "phoenix")
        self.agent_factory = agent_factory
        self.factory = partial(agent_factory, mcp=mcp, configuration=self.config, observability=self.observability)
        self.lock = RLock()
        self.runtimes = {}
        self.closed = False
        self.bypass_approvals = os.environ.get("PRAXIS_BYPASS_APPROVALS") == "1"
        self.deletions = SessionDeletions(self)
        self.deletions.start_recovery()

    def delete(self, session_id):
        return self.deletions.delete(session_id)

    def mcp_status(self):
        return self.mcp.snapshot() if self.mcp is not None else {"servers": []}

    def configuration(self):
        return {
            "model": self.config.default_model,
            "models": list(self.config.models),
            "config_path": str(self.config.path),
            "configured": True,
            "bypass_approvals": self.bypass_approvals,
            "observability": self.observability.snapshot(),
        }

    def create(self, data):
        workspace = Path(required_text(data, "workspace")).expanduser().resolve()
        if not workspace.is_dir():
            raise ServiceError("invalid_workspace", "Workspace is not a directory")
        model = data.get("model") or self.configuration()["model"]
        if not isinstance(model, str) or not model.strip():
            raise ServiceError("invalid_model", "Model must be a nonempty string")
        try:
            selected = self.config.resolve(model)
            if not selected.tool_calling:
                raise ValueError("Agent models require tool_calling")
        except ValueError as error:
            raise ServiceError("invalid_model", str(error)) from error
        payload = {"workspace": os.path.normcase(str(workspace)), "model": model}

        def create(_):
            record = self.store.sessions.create_session(workspace, model)
            self.store.updated(record.id)
            return {"session": self.store.summary(record.id)}

        return self.store.command(
            required_text(data, "command_id"), "create", payload, create
        )[0]

    def list(self, workspace=None):
        with self.store.db.transaction(write=False) as connection:
            if workspace:
                rows = connection.execute(
                    "SELECT id FROM sessions WHERE workspace = ? AND delete_requested_at IS NULL ORDER BY updated_at DESC",
                    (os.path.normcase(str(Path(workspace).expanduser().resolve())),),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT id FROM sessions WHERE delete_requested_at IS NULL ORDER BY updated_at DESC"
                ).fetchall()
            return {"sessions": [self.store.summary(row["id"]) for row in rows]}

    def runtime(self, session_id):
        with self.lock:
            if self.closed:
                raise ServiceError("closing", "Service is shutting down", 503)
            if session_id in self.deletions.marked:
                raise ServiceError("session_deleted", "Session is being deleted", 410)
            self.store.require_available(session_id)
            if session_id not in self.runtimes:
                record = self.store.sessions.get_session(session_id)
                self.runtimes[session_id] = SessionRuntime(
                    record, self.store, self.factory, self.bypass_approvals
                )
            return self.runtimes[session_id]

    def decide(self, approval_id, approved):
        if not isinstance(approved, bool):
            raise ServiceError("invalid_params", "approved must be a boolean")
        with self.store.db.connection() as connection:
            row = connection.execute(
                "SELECT session_id FROM approvals WHERE id = ?", (approval_id,)
            ).fetchone()
        if row is None:
            raise ServiceError("approval_not_found", "Approval does not exist", 404)
        return self.runtime(row["session_id"]).approvals.decide(approval_id, approved)

    def configure(self, data):
        if set(data) - {"config_path", "bypass_approvals"}:
            raise ServiceError("invalid_params", "Settings accept config_path and bypass_approvals only")
        if "bypass_approvals" in data and not isinstance(data["bypass_approvals"], bool):
            raise ServiceError("invalid_params", "bypass_approvals must be a boolean")
        config = self.config
        if "config_path" in data:
            try:
                config = load_config(required_text(data, "config_path"))
            except ValueError as error:
                raise ServiceError("invalid_configuration", str(error)) from error
        bypass = data.get("bypass_approvals", self.bypass_approvals)
        with self.lock, ExitStack() as locks:
            runtimes = [runtime for runtime in self.runtimes.values() if not runtime.closed]
            for runtime in runtimes:
                locks.enter_context(runtime.lock)
            if any(runtime.run is not None for runtime in runtimes):
                raise ServiceError("sessions_running", "Stop running sessions before changing shared service settings", 409)
            for runtime in runtimes:
                if runtime.agent:
                    runtime.agent.close()
                    runtime.agent = None
            self.config = config
            if "config_path" in data:
                self.observability.close()
                self.observability = Observability(config.observability, self.store.db.path.parent / "phoenix")
            self.factory = partial(self.agent_factory, mcp=self.mcp, configuration=config,
                                   observability=self.observability)
            self.bypass_approvals = bypass
            for runtime in runtimes:
                runtime.factory = self.factory
                runtime.bypass_approvals = bypass
        return self.configuration()

    def close(self):
        with self.lock:
            self.closed = True
        self.deletions.wait()
        with self.lock:
            runtimes = list(self.runtimes.values())
        try:
            for runtime in runtimes:
                runtime.close()
        finally:
            try:
                self.observability.close()
            finally:
                if self.mcp is not None:
                    self.mcp.close()
