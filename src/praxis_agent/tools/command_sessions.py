"""Agent-owned commands with file-backed output, independent of tool/turn waits."""
from __future__ import annotations

import atexit
import logging
import os
import subprocess
import tempfile
from contextlib import ExitStack
from threading import Event, Lock, RLock, Thread
from time import monotonic
from uuid import uuid4

from praxis_agent.execution import ExecutionStopped, current_handle, current_run
from praxis_agent.tools.command import _OutputDecoder, _open_output_reader
from praxis_agent.tools.processes import CommandProcess


def validate_yield(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("yield_time_ms must be an integer")
    if not 0 <= value <= 600000:
        raise ValueError("yield_time_ms must be between 0 and 600000")


class _OutputFile:
    def __init__(self, resources):
        self.writer = resources.enter_context(tempfile.NamedTemporaryFile(buffering=0))
        self.model = resources.enter_context(_open_output_reader(self.writer.name))
        self.display = resources.enter_context(_open_output_reader(self.writer.name))
        self.decoders = {"model": _OutputDecoder(), "display": _OutputDecoder()}

    def size(self):
        return os.fstat(self.writer.fileno()).st_size

    def read(self, consumer, end, final=False):
        reader = getattr(self, consumer)
        parts = []
        while reader.tell() < end:
            chunk = reader.read(min(65536, end - reader.tell()))
            if not chunk:
                break
            parts.append(self.decoders[consumer].decode(chunk))
        if final:
            parts.append(self.decoders[consumer].decode(b"", final=True))
        return "".join(parts)


class _Session:
    def __init__(self, argv, cwd):
        self.id = "cmd_" + uuid4().hex
        self.command = list(argv)
        self.lock = RLock()
        self.operation = Lock()
        self.resources = ExitStack()
        self.state = "running"
        self.error = None
        self.exit_code = None
        self.completed_at = None
        self.cleaned = False
        self.stop_requested = Event()
        self.pending = None
        self.unsubscribe = lambda: None
        try:
            self.outputs = {name: _OutputFile(self.resources) for name in ("stdout", "stderr")}
            self.process = CommandProcess(argv, cwd=cwd, stdin=subprocess.DEVNULL,
                                          stdout=self.outputs["stdout"].writer,
                                          stderr=self.outputs["stderr"].writer, shell=False)
        except BaseException:
            self.resources.close()
            raise

    def finish(self, status="exited", error=None):
        with self.lock:
            if self.state != "running":
                return
            # Stopping descendants also closes their output writers. Do not announce
            # completion until cleanup has returned, even if the root already exited.
            try:
                self.process.close()
                self.cleaned = True
            except Exception as failure:
                status, error = "failed", f"Command cleanup failed: {failure}"
            self.exit_code = self.process.process.returncode
            self.state, self.error = status, error
            if status == "exited" and self.exit_code != 0:
                self.error = (f"The command ran and exited with code {self.exit_code}. "
                              "The meaning of a nonzero exit code depends on the program.")
            self.completed_at = monotonic()
            self.unsubscribe()

    def stop(self):
        self.request_stop()
        self.finish("cancelled", "Command and its child processes were stopped.")

    def request_stop(self):
        self.stop_requested.set()
        self.process.request_stop()


# One maintenance thread for the host, never one output reader per stream/command.
_service_lock = Lock()
_managers = set()
_service_stop = Event()
_service_thread = None


def _maintain():
    while not _service_stop.wait(0.2):
        with _service_lock:
            managers = tuple(_managers)
        for manager in managers:
            try:
                manager.maintain()
            except Exception:
                logging.getLogger(__name__).exception("Could not maintain command sessions")


def _register(manager):
    global _service_thread
    with _service_lock:
        _managers.add(manager)
        if _service_thread is None:
            _service_thread = Thread(target=_maintain, name="praxis-command-maintenance", daemon=True)
            _service_thread.start()


def _shutdown():
    _service_stop.set()
    with _service_lock:
        managers = tuple(_managers)
    for manager in managers:
        try:
            manager.close()
        except Exception:
            logging.getLogger(__name__).exception("Could not close command sessions")
    if _service_thread is not None:
        _service_thread.join(timeout=2)


atexit.register(_shutdown)


class CommandSessionManager:
    def __init__(self, *, max_output_bytes=16 * 1024 * 1024, max_total_bytes=128 * 1024 * 1024,
                 retention_seconds=900, max_sessions=32):
        self.max_output_bytes = max_output_bytes
        self.max_total_bytes = max_total_bytes
        self.retention_seconds = retention_seconds
        self.max_sessions = max_sessions
        self._lock = RLock()
        self._sessions = {}
        self._finished = {}
        self._closed = False

    def start(self, argv, cwd):
        # Serialize spawn with close so no command can escape shutdown registration.
        with self._lock:
            if self._closed:
                raise ExecutionStopped("command manager is closed")
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError("Too many command sessions; finish or stop existing commands first")
            run = current_run()
            if run is not None:
                run.check()
            session = _Session(argv, cwd)
            self._sessions[session.id] = session
            if run is not None:
                # This callback survives the startup tool, but is removed when the
                # turn settles normally. No process future is added to Run.handles.
                remove_cleanup = run.on_cancel_cleanup(session.stop)
                remove_request = run.on_cancel(session.request_stop)
                session.unsubscribe = lambda: (remove_request(), remove_cleanup())
            _register(self)
            return session.id

    def _get(self, session_id):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError("Command session is unavailable or expired; it may belong to another agent")
            return session

    def _refresh(self, session):
        with session.lock:
            if session.state != "running":
                if not session.cleaned:
                    try:
                        session.process.close()
                    except Exception:
                        return
                    session.cleaned = True
                return
            size = sum(output.size() for output in session.outputs.values())
            if session.stop_requested.is_set():
                session.finish("cancelled", "Command and its child processes were stopped.")
            elif size > self.max_output_bytes:
                session.finish("failed", "Command output exceeded the temporary storage limit; output was truncated.")
            elif session.process.process.poll() is not None:
                session.finish()

    def _ends(self, session):
        # Snapshot EOF, bounded even if a child writes faster than quota sampling.
        remaining = self.max_output_bytes
        ends = {}
        for name, output in session.outputs.items():
            ends[name] = min(output.size(), remaining)
            remaining -= ends[name]
        return ends

    def _display(self, session, callback, *, all_available=False):
        if callback is None:
            return
        with session.lock:
            ends = self._ends(session)
            chunks = []
            for name, output in session.outputs.items():
                end = ends[name] if all_available else min(ends[name], output.display.tell() + 65536)
                final = session.state != "running" and end == ends[name]
                text = output.read("display", end, final)
                if text:
                    chunks.append((name, text))
        for name, text in chunks:
            callback(name, text)

    def wait(self, session_id, yield_time_ms=10000, *, check_execution=None,
             on_output=None, defer_ack=False):
        validate_yield(yield_time_ms)
        with self._lock:
            if session_id in self._finished:
                return dict(self._finished[session_id][1])
        session = self._get(session_id)
        if not session.operation.acquire(blocking=False):
            raise RuntimeError("Command session is busy; wait for its current operation to return")
        handle = current_handle()
        unsubscribe = handle.on_cancel(session.request_stop) if handle is not None else lambda: None
        try:
            # A batch remains replayable until its tool result is durably accepted.
            with session.lock:
                pending = session.pending
            if pending is not None:
                result = dict(pending)
            else:
                deadline = monotonic() + yield_time_ms / 1000
                while True:
                    self._refresh(session)
                    self._display(session, on_output)
                    try:
                        if check_execution is not None:
                            check_execution()
                        if handle is not None:
                            handle.check()
                    except ExecutionStopped:
                        session.stop()
                    if session.state != "running" or monotonic() >= deadline:
                        break
                    try:
                        session.process.process.wait(timeout=min(.05, max(.001, deadline - monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
                self._refresh(session)
                self._display(session, on_output, all_available=True)
                with session.lock:
                    ends = self._ends(session)
                    result = {"session_id": session.id, "status": session.state,
                              "exit_code": session.exit_code, "error": session.error}
                    for name, output in session.outputs.items():
                        text = output.read("model", ends[name], session.state != "running")
                        result[name] = text.replace("\r\n", "\n").replace("\r", "\n")
                    session.pending = dict(result)
            if not defer_ack:
                self.acknowledge(session_id, result)
            return result
        finally:
            unsubscribe()
            session.operation.release()

    def acknowledge(self, session_id, result):
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            with session.lock:
                if session.pending != result:
                    return
                session.pending = None
                if result["status"] != "running" and session.cleaned:
                    self._dispose(session)
                    self._finished[session_id] = (monotonic(), {**result, "stdout": "", "stderr": ""})

    def stop(self, session_id):
        session = self._get(session_id)
        session.stop()

    def _dispose(self, session):
        session.unsubscribe()
        session.resources.close()
        self._sessions.pop(session.id, None)

    def maintain(self):
        with self._lock:
            sessions = tuple(self._sessions.values())
            total = sum(output.size() for session in sessions for output in session.outputs.values())
        if total > self.max_total_bytes:
            for session in sessions:
                session.finish("failed", "Agent command temporary storage limit exceeded.")
        for session in sessions:
            # Tools/close may own the files; never wait on a long tool operation.
            if not session.operation.acquire(blocking=False):
                continue
            try:
                with self._lock:
                    if session.id not in self._sessions:
                        continue
                    self._refresh(session)
                    if (session.completed_at is not None and
                            monotonic() - session.completed_at >= self.retention_seconds):
                        # Retry cleanup failures before releasing output files.
                        if not session.cleaned:
                            session.process.close()
                        self._dispose(session)
            finally:
                session.operation.release()
        with self._lock:
            self._finished = {key: value for key, value in self._finished.items()
                              if monotonic() - value[0] < self.retention_seconds}
            if not self._sessions and not self._finished:
                with _service_lock:
                    _managers.discard(self)

    def summaries(self):
        with self._lock:
            return [{"session_id": session.id, "status": session.state, "argv": session.command}
                    for session in self._sessions.values()]

    def close(self):
        with self._lock:
            self._closed = True
            sessions = tuple(self._sessions.values())
        for session in sessions:
            session.stop()
        for session in sessions:
            with session.operation:
                with self._lock:
                    if session.id in self._sessions:
                        if not session.cleaned:
                            session.process.close()
                        self._dispose(session)
        with _service_lock:
            _managers.discard(self)


_default = None


def default_manager():
    global _default
    if _default is None:
        _default = CommandSessionManager()
    return _default
