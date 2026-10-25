from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from threading import RLock

from praxis_agent.config import ModelConfig
from praxis_agent.factory import create_default_agent

from .errors import ServiceError, required_text
from .runtime import SessionRuntime


class SessionManager:
    def __init__(self, store, *, agent_factory=create_default_agent):
        self.store = store
        self.factory = agent_factory
        self.lock = RLock()
        self.runtimes = {}
        self.closed = False
        self.bypass_approvals = os.environ.get("PRAXIS_BYPASS_APPROVALS") == "1"

    def configuration(self):
        try:
            config = ModelConfig.from_env()
            return {
                "model": config.model,
                "base_url": config.base_url,
                "configured": True,
                "bypass_approvals": self.bypass_approvals,
            }
        except ValueError as error:
            return {
                "model": os.environ.get("ADA_MODEL", "auto(free)"),
                "configured": False,
                "configuration_error": str(error),
                "bypass_approvals": self.bypass_approvals,
            }

    def create(self, data):
        workspace = Path(required_text(data, "workspace")).expanduser().resolve()
        if not workspace.is_dir():
            raise ServiceError("invalid_workspace", "Workspace is not a directory")
        model = data.get("model") or self.configuration()["model"]
        if not isinstance(model, str) or not model.strip():
            raise ServiceError("invalid_model", "Model must be a nonempty string")
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
                    "SELECT id FROM sessions WHERE workspace = ? ORDER BY updated_at DESC",
                    (os.path.normcase(str(Path(workspace).expanduser().resolve())),),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT id FROM sessions ORDER BY updated_at DESC"
                ).fetchall()
            return {"sessions": [self.store.summary(row["id"]) for row in rows]}

    def runtime(self, session_id):
        with self.lock:
            if self.closed:
                raise ServiceError("closing", "Service is shutting down", 503)
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
        keys = {
            "model": "ADA_MODEL",
            "base_url": "ADA_BASE_URL",
            "api_key": "ADA_API_KEY",
        }
        if set(data) - (set(keys) | {"bypass_approvals"}):
            raise ServiceError("invalid_params", "Unknown configuration field")
        for key in keys:
            if key in data and (
                not isinstance(data[key], str) or len(data[key]) > 4096
            ):
                raise ServiceError("invalid_params", f"Invalid {key}")
        if "bypass_approvals" in data and not isinstance(
            data["bypass_approvals"], bool
        ):
            raise ServiceError("invalid_params", "bypass_approvals must be a boolean")
        changes = {
            key: value
            for key, value in data.items()
            if (
                value != self.bypass_approvals
                if key == "bypass_approvals"
                else value != os.environ.get(keys[key], "")
            )
        }
        if not changes:
            return self.configuration()
        with self.lock, ExitStack() as locks:
            for runtime in self.runtimes.values():
                locks.enter_context(runtime.lock)
            if any(runtime.run is not None for runtime in self.runtimes.values()):
                raise ServiceError(
                    "sessions_running",
                    "Stop running sessions before changing shared service settings",
                    409,
                )
            for runtime in self.runtimes.values():
                if runtime.agent:
                    runtime.agent.close()
                    runtime.agent = None
            for key, value in changes.items():
                if key == "bypass_approvals":
                    self.bypass_approvals = value
                else:
                    os.environ[keys[key]] = value
            for runtime in self.runtimes.values():
                runtime.bypass_approvals = self.bypass_approvals
        return self.configuration()

    def close(self):
        with self.lock:
            self.closed = True
            runtimes = list(self.runtimes.values())
        for runtime in runtimes:
            runtime.close()
