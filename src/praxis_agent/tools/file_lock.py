"""Process-wide file transactions with notification-driven cancellation."""

import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition, Lock

from praxis_agent.execution import current_handle, current_run


@dataclass
class _FileState:
    condition: Condition = field(default_factory=Condition)
    occupied: bool = False
    users: int = 0

    def wake(self) -> None:
        with self.condition:
            self.condition.notify_all()


_states: dict[str, _FileState] = {}
_states_lock = Lock()


@contextmanager
def file_transaction(path: Path) -> Iterator[Callable[[], None]]:
    """Serialize read/modify/write for a resolved path; yield a cancellation check.

    All waiters subscribe to their run as well as their current operation, so a
    whole-turn cancellation wakes them even before cancellation reaches each
    individual operation. A save already in progress is allowed to finish.
    """
    key = os.path.normcase(str(path.resolve()))
    with _states_lock:
        state = _states.setdefault(key, _FileState())
        state.users += 1

    scopes = tuple(scope for scope in (current_run(), current_handle()) if scope is not None)

    def check_cancelled() -> None:
        for scope in scopes:
            scope.check()

    acquired = False
    try:
        with ExitStack() as subscriptions:
            # Registration may synchronously invoke wake when already cancelled.
            # Do it outside the condition's critical section.
            for scope in scopes:
                subscriptions.callback(scope.on_cancel(state.wake))
            try:
                with state.condition:
                    while True:
                        check_cancelled()
                        if not state.occupied:
                            state.occupied = True
                            acquired = True
                            break
                        state.condition.wait()

                check_cancelled()
                yield check_cancelled
            finally:
                if acquired:
                    with state.condition:
                        state.occupied = False
                        state.condition.notify_all()
    finally:
        # Count holders AND waiters; removing a state with remaining waiters
        # would let a new caller create a second lock for the same file.
        with _states_lock:
            state.users -= 1
            if state.users == 0:
                del _states[key]
