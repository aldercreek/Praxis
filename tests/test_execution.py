from threading import Event, Thread

import pytest

from praxis_agent.execution import ExecutionStopped, Handle, Run, run_operation


def launch(scope, operation):
    def execute():
        try:
            scope.execute(operation)
        except BaseException:
            pass  # The scope's Future is the authoritative result.
    thread = Thread(target=execute, daemon=True)
    thread.start()
    return thread


def test_tree_cancellation_waits_for_leaf_cleanup_and_run_persistence():
    root, child, grandchild = Run("root"), Run("child"), Run("grandchild")
    child_handle = root.add_child(child)
    grandchild_handle = child.add_child(grandchild)
    started, wake, cleaning, release, persisted = (Event() for _ in range(5))

    def leaf(handle):
        handle.on_cancel(wake.set)
        started.set()
        try:
            assert wake.wait(5)
            handle.check()
        finally:
            cleaning.set()
            assert release.wait(5)

    def execute_leaf():
        try:
            run_operation("command", leaf)
        finally:
            grandchild.settle(ExecutionStopped())
            persisted.set()

    threads = [launch(grandchild, execute_leaf),
               launch(child, lambda: grandchild_handle.future.result()),
               launch(root, lambda: child_handle.future.result())]
    try:
        assert started.wait(2)
        root.cancel()
        root.cancel()  # Repeated Esc must not complete or interrupt cleanup again.
        assert cleaning.wait(2)
        assert not root.finished.done()
        assert not child_handle.future.done()
        assert not grandchild_handle.future.done()
        assert not grandchild.handles[0].future.done()
        assert not persisted.is_set()
        release.set()
        with pytest.raises(ExecutionStopped):
            root.finished.result(timeout=3)
        assert persisted.is_set()
        assert grandchild.handles[0].future.done()
        assert child_handle.future is child.finished
    finally:
        release.set()
        for thread in threads:
            thread.join(3)
            assert not thread.is_alive()


def test_cancel_before_callback_registration_and_stop_new_children():
    scope = Run()
    handle = scope.add(Handle("request"))
    scope.cancel()
    called = []
    handle.on_cancel(lambda: called.append("abort"))
    handle.cancel()
    assert called == ["abort"]
    assert not handle.future.done()
    with pytest.raises(ExecutionStopped):
        scope.add(Handle("too late"))
    with pytest.raises(ExecutionStopped):
        handle.execute(lambda h: pytest.fail("cancelled operation started"))
    assert isinstance(handle.future.exception(), ExecutionStopped)


def test_success_and_failure_futures_are_completed_by_owner():
    scope = Run()
    assert scope.execute(lambda: run_operation("read", lambda h: "contents")) == "contents"
    assert scope.finished.result() == "contents"
    assert scope.handles[0].future.result() == "contents"
    scope.cancel()
    assert not scope.cancelled.is_set()

    failed = Run()
    def broken(handle):
        raise ValueError("bad input")
    with pytest.raises(ValueError, match="bad input"):
        failed.execute(lambda: run_operation("broken", broken))
    assert isinstance(failed.finished.exception(), ValueError)
    assert isinstance(failed.handles[0].future.exception(), ValueError)


def test_one_failed_cancel_callback_does_not_skip_other_tasks():
    scope = Run()
    first, second = scope.add(Handle("first")), scope.add(Handle("second"))
    def broken():
        raise OSError("abort failed")
    first.on_cancel(broken)
    notified = Event()
    second.on_cancel(notified.set)
    scope.cancel()
    assert notified.is_set()
    assert isinstance(first.cancel_errors[0], OSError)
    assert not first.future.done() and not second.future.done()


def test_unsubscribed_callback_is_not_called_by_later_cancellation():
    handle = Handle("request")
    callback = Event()
    unsubscribe = handle.on_cancel(callback.set)
    unsubscribe()
    handle.cancel()
    assert not callback.is_set()


def test_cancel_cleanup_failure_does_not_skip_remaining_resources():
    scope = Run()
    completed = []

    def broken():
        raise OSError("cleanup failed")

    scope.on_cancel_cleanup(broken)
    scope.on_cancel_cleanup(lambda: completed.append("second"))
    scope.cancel()
    with pytest.raises(ExceptionGroup, match="clean up"):
        scope.settle()
    assert completed == ["second"]
