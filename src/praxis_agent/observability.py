"""Application-owned observability, with bounded delivery and explicit status."""
from __future__ import annotations

from copy import deepcopy
from importlib.metadata import entry_points
import logging
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread
from typing import Protocol

from praxis_agent.config import ObservabilityConfig
from praxis_agent.events import AgentEvent, EventSink
from praxis_agent.sessions.database import data_directory

logger = logging.getLogger(__name__)


class ObservabilitySink(EventSink, Protocol):
    def close(self) -> None:
        """Flush pending records and release resources owned by this instance."""
        ...


def load_observability(config: ObservabilityConfig) -> ObservabilitySink | None:
    """Load only the explicitly selected entry point; never fall back."""
    if not config.enabled:
        return None
    matches = tuple(entry_points(group="praxis.observability", name=config.provider))
    if not matches:
        raise ValueError(f"Observability provider {config.provider!r} is not installed in this Python environment")
    if len(matches) != 1:
        raise ValueError(f"Multiple observability plugins register provider {config.provider!r}")
    factory = matches[0].load()
    if not callable(factory):
        raise TypeError("Observability entry point must be a callable factory")
    sink = factory(options=deepcopy(dict(config.options)))
    if not callable(getattr(sink, "emit", None)) or not callable(getattr(sink, "close", None)):
        if callable(getattr(sink, "close", None)):
            sink.close()
        raise TypeError("Observability factory must return a sink with emit() and close()")
    return sink


class _SessionSink:
    def __init__(self, owner, session_id):
        self.owner, self.session_id = owner, session_id

    def emit(self, event):
        # The callback follows clear_history() when it creates a new session.
        self.owner.emit(self.session_id(), event)

    emit_live = emit


class Observability:
    def __init__(self, config: ObservabilityConfig, directory: Path | None = None):
        self.config = config
        self.local_phoenix = config.provider in (None, "phoenix")
        self.directory = directory or data_directory() / "phoenix"
        self.lock = RLock()
        self.stopping = Event()
        self.queue = Queue(maxsize=2048)
        self.thread = None
        self.process = None
        self.state = "disabled" if not config.enabled else "starting"
        self.message = ""
        self.url = ""
        self.start()

    def start(self):
        with self.lock:
            if not self.config.enabled or self.stopping.is_set():
                return
            if self.thread is not None and self.thread.is_alive():
                return
            self.queue = Queue(maxsize=2048)
            self.state, self.message, self.url = "starting", "", ""
            self.thread = Thread(target=self._work, name="praxis-observability", daemon=True)
            self.thread.start()

    def bind(self, session_id):
        return _SessionSink(self, session_id)

    def snapshot(self):
        with self.lock:
            return dict(provider=self.config.provider or "phoenix", state=self.state,
                        message=self.message, url=self.url)

    def fail(self, message):
        with self.lock:
            if self.state == "failed":
                return
            self.state = "failed"
            self.message = message
        logger.warning("Observability: %s", message)

    def emit(self, session_id, event: AgentEvent):
        with self.lock:
            if self.state not in {"starting", "ready"} or self.stopping.is_set():
                return
            try:
                self.queue.put_nowait((session_id, event))
            except Full:
                self.fail("Observation queue is full. Tracing is disabled; agent execution continues.")

    def _work(self):
        sink = None
        try:
            if self.local_phoenix:
                if self.config.options:
                    raise ValueError("Managed Phoenix does not accept plugin options")
                from praxis_agent.phoenix_service import LocalPhoenix
                from praxis_agent.phoenix_sink import PhoenixSink
                self.process = LocalPhoenix(self.directory)
                url = self.process.start(self.stopping)
                sink = PhoenixSink(url, on_error=self.fail)
                with self.lock:
                    self.url = url
            else:
                sink = load_observability(self.config)
            with self.lock:
                if self.state != "failed":
                    self.state = "ready"
            while self.state != "failed":
                try:
                    item = self.queue.get(timeout=0.2)
                except Empty:
                    if self.stopping.is_set():
                        break
                    if self.process is not None and not self.process.alive:
                        raise RuntimeError("Local Phoenix stopped unexpectedly")
                    continue
                session_id, event = item
                if self.local_phoenix:
                    sink.emit_for_session(session_id, event)
                else:
                    sink.emit(event)
        except Exception as error:
            if not self.stopping.is_set():
                self.fail(f"{self.config.provider or 'phoenix'}: {type(error).__name__}: {error}. "
                          "Tracing is unavailable; agent execution continues. No fallback was loaded.")
        finally:
            if sink is not None:
                try:
                    sink.close()
                except Exception:
                    logger.warning("Could not flush/close observability provider", exc_info=True)
            if self.process is not None:
                self.process.close()
            # Release queued payloads after failure or shutdown.
            while True:
                try:
                    self.queue.get_nowait()
                except Empty:
                    break

    def close(self):
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=15)
        if self.process is not None:
            self.process.close()
