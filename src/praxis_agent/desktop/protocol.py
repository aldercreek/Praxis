from __future__ import annotations

import json
import logging
from queue import Full, Queue
from threading import Event, Thread
from time import monotonic
from typing import TextIO


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def parse_request(line: str) -> dict:
    try:
        request = json.loads(line)
    except (ValueError, RecursionError) as error:
        raise ProtocolError("parse_error", "Request must be valid JSON") from error
    if not isinstance(request, dict):
        raise ProtocolError("invalid_request", "Request must be an object")
    if not isinstance(request.get("id"), str) or not request["id"]:
        raise ProtocolError("invalid_request", "Request id must be a non-empty string")
    if not isinstance(request.get("method"), str) or not request["method"]:
        raise ProtocolError("invalid_request", "Request method must be a non-empty string")
    if not isinstance(request.get("params", {}), dict):
        raise ProtocolError("invalid_params", "Request params must be an object")
    return {**request, "params": request.get("params", {})}


def required_string(params: dict, name: str) -> str:
    value = params.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError("invalid_params", f"{name} must be a non-empty string")
    return value.strip()


class OutputChannel:
    """A single writer keeps NDJSON messages intact and applies bounded backpressure."""

    def __init__(self, stream: TextIO, *, capacity: int = 2048, send_timeout: float = 1.0,
                 on_failure=None):
        self.stream = stream
        self.queue: Queue[dict | None] = Queue(maxsize=capacity)
        self.failed = Event()
        self.send_timeout = send_timeout
        self.on_failure = on_failure
        self.thread = Thread(target=self._write, name="praxis-stdout", daemon=True)
        self.thread.start()

    def _fail(self, reason):
        if not self.failed.is_set():
            self.failed.set()
            logging.getLogger(__name__).error("Desktop transport closed: %s", reason)
            if self.on_failure is not None:
                self.on_failure()

    def send(self, message: dict | None) -> bool:
        deadline = monotonic() + self.send_timeout
        while not self.failed.is_set():
            try:
                self.queue.put(message, timeout=min(0.1, max(0, deadline - monotonic())))
                return True
            except Full:
                if monotonic() >= deadline:
                    # A stalled desktop must not hold session locks or cancellation indefinitely.
                    # Fail the connection as a whole: critical messages are never silently skipped.
                    self._fail("stdout consumer stopped reading (output queue full)")
                    return False
        return False

    def _write(self) -> None:
        try:
            while not self.failed.is_set():
                message = self.queue.get()
                if message is None:
                    return
                self.stream.write(json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n")
                self.stream.flush()
        except Exception as error:
            self._fail(f"stdout write failed: {error}")

    def close(self) -> None:
        self.send(None)
        self.thread.join(timeout=self.send_timeout)
        self.failed.set()
