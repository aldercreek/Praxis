import asyncio
import json
from io import StringIO
from threading import Event, Thread
from types import SimpleNamespace

from rich.console import Console
from rich.text import Text
from textual.containers import ScrollableContainer, VerticalScroll
from textual.widgets import Static, TextArea

from praxis_agent.context.capacity import ContextPressure, ContextStatus
from praxis_agent.events import (
    AssistantTextCompleted,
    AssistantTextDelta,
    ChildAgentEvent,
    ChildAgentMessage,
    ChildAgentStatus,
    CompositeEventSink,
    ContextCompaction,
    MessagesReceived,
    ModelRequestCompleted,
    ModelRequestStarted,
    PlanUpdated,
    ToolCallPrepared,
    ToolFinished,
    ToolOutputDelta,
    ToolStarted,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
)
from praxis_agent.model import ModelResponse
from praxis_agent.multi_agent.models import MultiAgentConfig
from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.planning.plan import PlanStep, PlanStepStatus
from praxis_agent.terminal.app import AgentTab, ApprovalPanel, DetailsScreen, PraxisApp
from praxis_agent.terminal.views import AgentView, ToolBlock, ViewEventSink, ViewStore
from praxis_agent.tools import ToolResult


def widget_text(widget):
    capture = StringIO()
    Console(file=capture, width=100, color_system=None).print(widget.renderable)
    return capture.getvalue()


def text_of(view):
    capture = StringIO()
    console = Console(file=capture, width=100, color_system=None)
    for block in view.committed_blocks:
        console.print(block.output if isinstance(block, ToolBlock) else block)
    return capture.getvalue()


def test_views_keep_background_streams_isolated_and_commit_once():
    store = ViewStore()
    for agent_id, text in (("a", "alpha\n\nunfinished"), ("b", "beta")):
        for event in (TurnStarted(agent_id, f"task-{agent_id}"), ModelRequestStarted(agent_id, 1),
                      AssistantTextDelta(agent_id, 1, text)):
            store.accept(ChildAgentEvent("parent", agent_id, event))
    assert "alpha" in text_of(store.views["a"])
    assert "beta" not in text_of(store.views["a"])
    assert "beta" in text_of(store.views["b"])
    assert not store.views["main"].committed_blocks
    response = ModelResponse((), "alpha\n\nunfinished", ())
    for event in (ModelRequestCompleted("a", 1, response),
                  AssistantTextCompleted("a", response.output_text), TurnCompleted("a")):
        store.accept(ChildAgentEvent("parent", "a", event))
    result = text_of(store.views["a"])
    assert result.count("alpha") == result.count("unfinished") == 1
    # Non-streaming and corrected responses are also retained exactly once.
    view = store.views["b"]
    view.consume(ModelRequestCompleted("b", 1, ModelResponse((), "corrected beta", ())))
    view.consume(AssistantTextCompleted("b", "corrected beta"))
    assert text_of(view).count("beta") == 1
    assert "corrected beta" in text_of(view)


def test_parallel_tools_keep_remaining_activity_and_full_results():
    view = AgentView("a", "A")
    view.consume(TurnStarted("t", "task", occurred_at=10))
    for call_id in ("one", "two"):
        view.consume(ToolCallPrepared("t", call_id, "read_file", {"relative_path": call_id}, False))
        view.consume(ToolStarted("t", call_id, "read_file"))
    assert view.phase == "执行 2 个工具"
    output = "full contents\n" * 100
    view.consume(ToolFinished("t", "one", "read_file", ToolResult(True, output)))
    assert view.phase == "读取文件"
    assert view.committed_blocks[1].output == output
    assert view.committed_blocks[2].state == "running"
    view.consume(ToolFinished("t", "two", "read_file", ToolResult(False, "missing file")))
    assert view.state == "running"  # A tool error is not an agent failure.
    view.consume(TurnFailed("t", "RuntimeError", "stopped", occurred_at=35))
    view.set_status(ChildAgentStatus("parent", "a", "idle", occurred_at=90))
    assert view.phase == "执行出错"
    assert view.elapsed(now=200) == 25


def test_command_preview_is_removed_when_full_result_is_committed():
    store = ViewStore()
    store.accept(ToolOutputDelta("t", "cmd", "stdout", "partial", agent_id="a"))
    view = store.views["a"]
    assert "partial" in text_of(view)
    result = {"status": "exited", "exit_code": 0, "stdout": "partial", "stderr": "", "error": None}
    store.accept(ChildAgentEvent("parent", "a", ToolFinished(
        "t", "cmd", "run_command", ToolResult(True, json.dumps(result), structured_output=result))))
    assert text_of(view).count("partial") == 1


def test_background_report_does_not_clear_or_get_deleted_by_root_stream():
    store = ViewStore()
    store.accept(ModelRequestStarted("main", 1))
    store.accept(AssistantTextDelta("main", 1, "draft paragraph\n\nstill writing"))
    store.accept(ChildAgentMessage("main", "a", "report"))
    assert "still writing" in text_of(store.views["main"])
    store.accept(ModelRequestCompleted("main", 1, ModelResponse((), "corrected answer", ())))
    text = text_of(store.views["main"])
    assert "收到新消息" in text
    assert "corrected answer" in text
    assert "draft paragraph" not in text


def test_page_preserves_plan_capacity_compaction_and_partial_answer_on_failure():
    view = AgentView("main", "主 agent")
    status = ContextStatus(73000, 100000, 73, ContextPressure.SOFT)
    view.consume(TurnStarted("turn", "investigate"))
    view.consume(ModelRequestStarted("turn", 1, context_status=status))
    assert "73%" in view._capacity.plain
    view.consume(PlanUpdated("turn", (PlanStep("Inspect logs", PlanStepStatus.IN_PROGRESS),)))
    assert "Inspect logs" in text_of(view)
    view.consume(ContextCompaction("turn", "started", 110000))
    assert view.phase == "正在压缩上下文"
    view.consume(ContextCompaction("turn", "chunk_started", 110000,
                                   round_number=1, chunk_number=2, chunk_count=3))
    assert "2/3" in view.phase
    view.consume(ContextCompaction("turn", "completed", 110000, after_tokens=40000,
                                   request_count=3, elapsed_seconds=12))
    assert "110,000 → 40,000" in text_of(view)
    view.consume(MessagesReceived("turn", ({"role": "user", "content": "worker findings"},)))
    view.consume(ModelRequestStarted("turn", 2, context_status=status))
    # A response taller than the viewport remains readable after a mid-stream failure.
    code = "```python\n" + "print('evidence')\n" * 150
    view.consume(AssistantTextDelta("turn", 2, code))
    view.consume(TurnFailed("turn", "NetworkError", "disconnected"))
    rendered = text_of(view)
    assert rendered.count("print('evidence')") == 150
    assert "worker findings" in rendered
    assert "NetworkError: disconnected" in rendered
    assert "73%" in view._capacity.plain
    assert "73%" not in rendered  # Capacity belongs to the footer, not the conversation.
    view.consume(TurnStarted("next", "retry"))
    assert view._capacity is None
    assert view.phase == "等待模型响应"


def test_worker_stream_and_status_arrive_without_parent_pump_and_trace_is_not_duplicated(tmp_path):
    ready, release = Event(), Event()
    display, trace_events = ViewEventSink(), []
    sink = CompositeEventSink((display, SimpleNamespace(emit=trace_events.append)))

    def factory(agent_id, control, events):
        def run(task):
            events.emit(TurnStarted("child", task))
            events.emit(ModelRequestStarted("child", 1))
            events.emit(AssistantTextDelta("child", 1, "background progress"))
            ready.set()
            release.wait(5)
            events.emit(AssistantTextCompleted("child", "background progress"))
            events.emit(TurnCompleted("child"))
            return "background progress"
        return SimpleNamespace(run=run, close=lambda: None, context_status=None)

    runtime = AgentRuntime(factory, MultiAgentConfig(), workspace=tmp_path, live_output_sink=sink)
    try:
        info = runtime.spawn("task", "scope", working_directory=".")
        assert ready.wait(2)
        store = ViewStore()
        display.drain(store)
        view = store.views[info["agent_id"]]
        assert "background progress" in text_of(view)
        assert view.task == "scope"
        assert not trace_events
        before = text_of(view)
        runtime.pump_events(sink)
        display.drain(store)
        assert text_of(view) == before
        assert sum(isinstance(e, ChildAgentEvent) and isinstance(e.event, TurnStarted)
                   for e in trace_events) == 1
    finally:
        release.set()
        runtime.shutdown(lambda: runtime.pump_events(sink))


def test_app_switches_live_pages_and_preserves_full_tool_results():
    async def exercise():
        sink = ViewEventSink()
        app = PraxisApp(sink)
        async with app.run_test(size=(120, 35)) as pilot:
            sink.emit(TurnStarted("main", "root task"))
            sink.emit(ChildAgentStatus("main", "b", "running", "write tests"))
            for event in (TurnStarted("b", "child task"), ModelRequestStarted("b", 1),
                          AssistantTextDelta("b", 1, "child live output")):
                sink.emit(ChildAgentEvent("main", "b", event))
            await pilot.pause(0.2)
            assert app.selected_agent_id == "main"
            assert all("child live output" not in widget_text(widget)
                       for widget in app.query("#blocks Static"))
            await pilot.click("#agent-1")
            await pilot.pause(0.2)
            assert app.selected_agent_id == "b"
            assert any("child live output" in widget_text(widget)
                       for widget in app.query("#blocks Static"))
            sink.emit(ChildAgentEvent("main", "b", ToolFinished(
                "b", "read", "read_file", ToolResult(True, "entire tool result"))))
            await pilot.pause(0.2)
            await pilot.press("ctrl+t")
            await pilot.pause(0.2)
            assert isinstance(app.screen, DetailsScreen)
            assert "entire tool result" in widget_text(app.screen.query_one("#details-content", Static))
            await pilot.press("escape")
            await pilot.press("alt+left")
            await pilot.pause(0.2)
            await pilot.press("alt+right")
            await pilot.pause(0.2)
            assert app.selected_agent_id == "b"
            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            assert "entire tool result" in widget_text(app.screen.query_one("#details-content", Static))
            await pilot.press("escape")
            rows = list(app.query_one("#agents", ScrollableContainer).query(AgentTab))
            assert len({row.region.y for row in rows}) == 1
            assert all(row.size.height == 1 for row in rows)
            assert rows[1].has_class("selected")
            assert not rows[0].has_class("selected")
            rows[1].focus()
            await pilot.press("up")
            await pilot.pause()
            assert app.selected_agent_id == "main"
            await pilot.press("down")
            await pilot.pause()
            assert app.selected_agent_id == "b"
            app.query_one("#task", TextArea).focus()
            await pilot.pause()
            assert rows[1].has_class("selected")
            for index in range(8):
                sink.emit(ChildAgentStatus("main", f"extra-{index}", "running", "task"))
            await pilot.pause(0.2)
            assert app.query_one("#agents", ScrollableContainer).size.height == 1
            await pilot.resize_terminal(80, 35)
            await pilot.press("alt+left", "alt+left")
            await pilot.pause()
            assert app.selected_agent_id == "extra-7"
            assert app.query_one("#agents", ScrollableContainer).scroll_x > 0
    asyncio.run(exercise())


def test_app_restores_scroll_and_does_not_follow_output_when_reading_history():
    async def exercise():
        sink = ViewEventSink()
        app = PraxisApp(sink)
        for agent_id in ("main", "b"):
            view = app.store.get(agent_id)
            for i in range(60):
                view.committed_blocks.append(Text(f"{agent_id}: saved line {i}"))
        async with app.run_test(size=(90, 25)) as pilot:
            await pilot.pause(0.2)
            pane = app.query_one("#transcript", VerticalScroll)
            pane.scroll_to(y=7, animate=False, immediate=True)
            await pilot.pause(0.2)
            sink.emit(AssistantTextDelta("main", 1, "new text while reading"))
            await pilot.pause(0.2)
            assert pane.scroll_y == 7
            await pilot.press("alt+right")
            await pilot.pause(0.2)
            assert app.selected_agent_id == "b"
            pane.scroll_to(y=15, animate=False, immediate=True)
            await pilot.pause(0.2)
            await pilot.press("alt+left")
            await pilot.pause(0.2)
            assert pane.scroll_y == 7
            await pilot.press("alt+right")
            await pilot.pause(0.2)
            assert pane.scroll_y == 15
    asyncio.run(exercise())


def test_app_approval_works_while_viewing_another_agent():
    async def exercise():
        app = PraxisApp(ViewEventSink())
        result = []
        async with app.run_test(size=(100, 35)) as pilot:
            thread = Thread(target=lambda: result.append(app.approve_tool_call(
                "run_command", {"argv": ["pytest"], "agent_id": "b", "workspace": "demo"})))
            thread.start()
            try:
                await pilot.pause(0.2)
                assert app.query_one(ApprovalPanel).display
                await pilot.press("y")
                await pilot.pause(0.2)
                thread.join(2)
                assert result == [True]
                assert app.selected_agent_id == "main"
            finally:
                app.closing = True
                thread.join(2)
    asyncio.run(exercise())


def test_app_keeps_input_and_tabs_responsive_during_execution():
    async def exercise():
        sink, release = ViewEventSink(), Event()
        app = PraxisApp(sink, initial_task="work")

        def run(task):
            sink.emit(TurnStarted("main", task))
            sink.emit(ChildAgentStatus("main", "b", "running", "background task"))
            release.wait(5)
            sink.emit(TurnCompleted("main"))

        app.agent = SimpleNamespace(run=run, close=release.set)
        async with app.run_test(size=(60, 25)) as pilot:
            try:
                await pilot.pause(0.2)
                assert app.busy
                await pilot.press("alt+right")
                await pilot.pause(0.2)
                assert app.selected_agent_id == "b"
                release.set()
                await pilot.pause(0.2)
                assert not app.busy
                assert app.selected_agent_id == "b"
            finally:
                release.set()
                app._runner.join(2)
    asyncio.run(exercise())


def test_worker_approval_and_quit_do_not_require_parent_pump(tmp_path):
    async def exercise():
        sink = ViewEventSink()
        app = PraxisApp(sink)
        ready = Event()
        answers = []

        def factory(agent_id, control, events):
            def run(task):
                ready.set()
                answers.append(runtime.request_approval(agent_id, "run_command", {"argv": ["pytest"]}))
                return "stopped"
            return SimpleNamespace(run=run, close=lambda: None, context_status=None)

        runtime = AgentRuntime(factory, MultiAgentConfig(), workspace=tmp_path,
                               live_output_sink=sink, approve_tool_call=app.approve_tool_call)
        app.agent = SimpleNamespace(
            collaboration=runtime,
            close=lambda: runtime.shutdown(lambda: runtime.pump_events(sink)),
        )
        async with app.run_test(size=(100, 35)) as pilot:
            runtime.spawn("task", "scope", working_directory=".")
            assert ready.wait(2)
            await pilot.pause(0.2)
            assert app.query_one(ApprovalPanel).display
            await pilot.press("ctrl+q")
            await pilot.pause(0.3)
        assert answers == [False]
        assert all(h.status.value == "closed" for h in runtime.handles.values())
    asyncio.run(exercise())


def test_app_clear_removes_background_pages_and_pending_old_events():
    async def exercise():
        sink = ViewEventSink()
        app = PraxisApp(sink)

        def clear():
            # Final events from the stopped workers may exceed one UI tick's batch.
            for _ in range(1200):
                sink.emit(ChildAgentStatus("old", "child", "closed"))

        app.agent = SimpleNamespace(clear_history=clear)
        async with app.run_test(size=(80, 25)) as pilot:
            sink.emit(ChildAgentStatus("old", "child", "idle"))
            await pilot.pause(0.2)
            await pilot.press("alt+right")
            await pilot.pause(0.2)
            app._start("/clear")
            await pilot.pause(0.4)
            assert app.selected_agent_id == "main"
            assert list(app.store.views) == ["main"]
            assert not app.store.views["main"].committed_blocks
            assert not app.query_one("#agent-bar").display
    asyncio.run(exercise())


def test_app_quit_cancels_running_command(tmp_path):
    import sys

    from praxis_agent.agent import Agent
    from praxis_agent.model import FunctionCall
    from praxis_agent.multi_agent.models import ExecutionControl
    from praxis_agent.tools import create_default_registry

    async def exercise():
        sink = ViewEventSink()
        control = ExecutionControl()
        call = FunctionCall("cmd", "run_command", json.dumps({
            "argv": [sys.executable, "-u", "-c", "import time; print('ready', flush=True); time.sleep(30)"],
        }))
        client = SimpleNamespace(create_response=lambda *a, **kw: ModelResponse((), "", (call,)))
        app = PraxisApp(sink, initial_task="run command")
        app.agent = Agent(model_client=client, workspace=tmp_path,
                          tool_registry=create_default_registry(check_execution=control.check),
                          execution_control=control, event_sink=sink, bypass_approvals=True)
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(30):
                await pilot.pause(0.1)
                if "ready" in text_of(app.store.views["main"]):
                    break
            assert "ready" in text_of(app.store.views["main"])
            await pilot.press("ctrl+q")
            await pilot.pause(0.5)
        app._runner.join(2)
        assert not app._runner.is_alive()
        assert control.cancelled.is_set()
    asyncio.run(exercise())
