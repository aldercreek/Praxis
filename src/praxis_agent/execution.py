"""Per-turn execution trees. Cancellation requests are not completion receipts."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future, wait
from contextvars import ContextVar
from threading import Event, Lock
from uuid import uuid4


class ExecutionStopped(RuntimeError):
    """The user cancelled this execution."""


_run: ContextVar[Run | None] = ContextVar("praxis_run", default=None)
_operation: ContextVar[Handle | None] = ContextVar("praxis_operation", default=None)


def current_run():
    return _run.get()


def current_handle():
    return _operation.get()


class Handle:
    """The owner completes future only AFTER its operation and cleanup return."""

    def __init__(self, name: str, *, future: Future | None = None):
        self.name = name
        self.future = future if future is not None else Future()
        self.cancelled = Event()
        self._lock = Lock()
        self._callbacks = {}
        self.cancel_errors = []

    def on_cancel(self, callback):
        key = object()
        with self._lock:
            invoke = self.cancelled.is_set() and not self.future.done()
            if not invoke and not self.future.done():
                self._callbacks[key] = callback
        if invoke:
            self._invoke(callback)

        def unsubscribe():
            with self._lock:
                self._callbacks.pop(key, None)
        return unsubscribe

    def _invoke(self, callback):
        try:
            callback()
        except Exception as error:
            with self._lock:
                self.cancel_errors.append(error)
            logging.getLogger(__name__).warning("Could not request cancellation of %s", self.name, exc_info=True)

    def cancel(self):
        with self._lock:
            if self.future.done() or self.cancelled.is_set():
                return
            self.cancelled.set()
            callbacks = list(self._callbacks.values())
            self._callbacks.clear()
        for callback in callbacks:
            self._invoke(callback)

    def check(self):
        if self.cancelled.is_set():
            raise ExecutionStopped(f"{self.name} cancelled")

    def execute(self, operation):
        token = _operation.set(self)
        try:
            self.check()
            result = operation(self)
        except BaseException as error:
            self.future.set_exception(error)
            raise
        else:
            self.future.set_result(result)
            return result
        finally:
            _operation.reset(token)
            with self._lock:
                self._callbacks.clear()


class Run(Handle):
    """Own direct children; each child Run owns and drains its own subtree."""

    def __init__(self, name="agent"):
        super().__init__(name)
        self.id = uuid4().hex
        self.finished = self.future
        self.handles = []
        self.accepting = True
        self._settled = False

    def add(self, handle):
        with self._lock:
            if not self.accepting or self.cancelled.is_set():
                raise ExecutionStopped("run no longer accepts work")
            self.handles.append(handle)
        return handle

    def add_child(self, child):
        handle = Handle(child.name, future=child.finished)
        handle.child_run = child
        handle.on_cancel(child.cancel)
        return self.add(handle)

    def cancel(self):
        with self._lock:
            if self._settled or self.finished.done() or self.cancelled.is_set():
                return
            self.cancelled.set()
            self.accepting = False
            children = tuple(self.handles)
            callbacks = tuple(self._callbacks.values())
            self._callbacks.clear()
        for callback in callbacks:
            self._invoke(callback)
        for handle in children:
            handle.cancel()

    def settle(self, error=None):
        """Drain before persistence; freeze cancellation before choosing terminal state."""
        with self._lock:
            self.accepting = False
            children = tuple(self.handles)
        if error is not None:
            self.cancel()
        if children:
            wait([handle.future for handle in children])
        with self._lock:
            self._settled = True
        if error is None:
            self.check()

    def execute(self, operation):
        token = _run.set(self)
        operation_token = _operation.set(None)
        try:
            # The owner must run even when cancelled before dispatch, to finalize
            # queued worker state. Its own first checkpoint prevents actual work.
            result = operation()
            self.settle()
        except BaseException as error:
            self.settle(error)
            self.finished.set_exception(error)
            raise
        else:
            self.finished.set_result(result)
            return result
        finally:
            _operation.reset(operation_token)
            _run.reset(token)
            with self._lock:
                self._callbacks.clear()


def run_operation(name, operation):
    handle = Handle(name)
    scope = current_run()
    if scope is not None:
        scope.add(handle)
    return handle.execute(operation)


async def await_cancellable(awaitable, handle):
    """Cancel on the owning loop; await task cleanup, not a canceled proxy Future."""
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(awaitable)
    unsubscribe = handle.on_cancel(lambda: loop.call_soon_threadsafe(task.cancel))
    try:
        try:
            return await task
        except asyncio.CancelledError as error:
            handle.check()
            raise RuntimeError("request cancelled outside its execution handle") from error
    finally:
        unsubscribe()


class ExecutionControl:
    """Stable service binding; per-turn cancellation lives on a fresh Run."""

    def __init__(self):
        self.cancelled = Event()  # Permanent shutdown, never reset for another run.
        self.run = None

    def check(self):
        if self.cancelled.is_set():
            raise ExecutionStopped("agent closed")
        if self.run is not None:
            self.run.check()
        handle = current_handle()
        if handle is not None:
            handle.check()

    def cancel(self):
        self.cancelled.set()
        if self.run is not None:
            self.run.cancel()
