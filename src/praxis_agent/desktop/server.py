from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

from praxis_agent.sessions import SessionNotFoundError

from .protocol import OutputChannel, ProtocolError, parse_request, required_string
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class DesktopServer:
    """The calling thread receives stdin; slow setup and Agent runs never occupy it."""

    def __init__(self, stdin, stdout, *, database_path=None, agent_factory=None, bypass_approvals=False):
        self.stdin = stdin
        self.stopping = Event()
        self.output = OutputChannel(stdout, on_failure=self.stopping.set)
        options = {} if agent_factory is None else {"agent_factory": agent_factory}
        self.manager = SessionManager(self.output.send, database_path=database_path,
                                      bypass_approvals=bypass_approvals, **options)
        self.management = ThreadPoolExecutor(max_workers=4, thread_name_prefix="praxis-session-setup")

    def _error(self, request_id, error):
        if isinstance(error, ProtocolError):
            code = error.code
        elif isinstance(error, SessionNotFoundError):
            code = "session_not_found"
        elif isinstance(error, ValueError):
            code = "invalid_params"
        else:
            code = "internal_error"
            logger.exception("Desktop request failed", exc_info=error)
        self.output.send({"id": request_id, "error": {"code": code, "message": str(error)}})

    def handle(self, request):
        request_id, method, params = request["id"], request["method"], request["params"]
        respond = lambda result: self.output.send({"id": request_id, "result": result})
        try:
            if method == "initialize":
                result = self.manager.initialize()
            elif method == "session.create":
                result = self.manager.create(params)
            elif method == "session.open":
                result = self.manager.open(params)
            elif method == "session.list":
                result = self.manager.list(params)
            elif method == "turn.start":
                self.manager.require(params).start(required_string(params, "text"), respond)
                return
            elif method == "turn.input":
                result = self.manager.require(params).enqueue(required_string(params, "text"))
            elif method == "turn.stop":
                result = self.manager.require(params).stop()
            elif method == "command.list":
                result = self.manager.require(params).commands()
            elif method == "command.stop":
                result = self.manager.require(params).stop_command(required_string(params, "command_id"))
            elif method == "approval.respond":
                result = self.manager.require(params).respond(params)
            elif method == "shutdown":
                result = {}
            else:
                raise ProtocolError("method_not_found", f"Unknown method: {method}")
            respond(result)
        except Exception as error:
            self._error(request_id, error)

    def _receive(self):
        try:
            for line in self.stdin:
                if self.stopping.is_set():
                    break
                if not line.strip():
                    continue
                try:
                    request = parse_request(line)
                except ProtocolError as error:
                    try:
                        parsed = json.loads(line)
                        request_id = parsed.get("id") if isinstance(parsed, dict) else None
                        if not isinstance(request_id, str):
                            request_id = None
                    except (ValueError, RecursionError):
                        request_id = None
                    self._error(request_id, error)
                    continue
                if request["method"] in {"session.create", "session.open", "session.list", "command.stop"}:
                    self.management.submit(self.handle, request)
                else:
                    self.handle(request)
                if request["method"] == "shutdown":
                    break
        finally:
            self.stopping.set()

    def run(self):
        # A dedicated reader may block on stdin. The owner can still shut down after stdout fails.
        receiver = Thread(target=self._receive, name="praxis-stdin", daemon=True)
        receiver.start()
        try:
            self.stopping.wait()
        finally:
            self.manager.begin_shutdown()
            self.management.shutdown(wait=True, cancel_futures=False)
            self.manager.close()
            self.output.close()
