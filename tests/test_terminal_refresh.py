"""Verify submitted frames, not just the UI after it has settled."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from rich.text import Text
from textual import messages
from textual.containers import VerticalScroll
from textual.widgets import Static

from praxis_agent.events import (
    ChildAgentEvent,
    ChildAgentStatus,
    ToolCallPrepared,
    ToolFinished,
    ToolOutputDelta,
    ToolStarted,
    TurnCancelled,
    TurnStarted,
)
from praxis_agent.terminal.app import ToolRecord
from praxis_agent.terminal.views import AgentView, ToolBlock
from praxis_agent.tools import ToolResult


@pytest.fixture
def frames(terminal_app, monkeypatch):
    captured = []
    display = terminal_app._display

    def record(screen, renderable):
        if renderable is not None:
            pane = screen.query_one("#transcript", VerticalScroll)
            captured.append((
                [strip.text.rstrip() for strip in screen._compositor.render_strips()],
                pane.scroll_y, pane.max_scroll_y,
            ))
        display(screen, renderable)

    monkeypatch.setattr(terminal_app, "_display", record)
    return captured


def start_command(sink, call_id="cmd"):
    sink.emit(ToolCallPrepared("turn", call_id, "run_command", {"argv": ["python", "demo.py"]}, False))
    sink.emit(ToolStarted("turn", call_id, "run_command"))


def assert_title_marker_color(title, color):
    marker_seen = False
    for row in range(title.size.height):
        for segment in title.render_line(row):
            if not segment.text.strip():
                continue
            if "•" in segment.text:
                marker_seen = True
                assert segment.text.strip() == "•"
                assert segment.style.color.name == color
            else:
                assert segment.style.color == title.rich_style.color
            assert not segment.style.dim
    assert marker_seen


@pytest.mark.parametrize("running", [False, True])
def test_unchanged_ticks_submit_no_frames(terminal_app, frames, monkeypatch, running):
    async def exercise():
        app = terminal_app
        if running:
            app.sink.emit(TurnStarted("turn", "检查项目"))
            start_command(app.sink)
        # Keep the clock text constant; changing it is a legitimate repaint.
        monkeypatch.setattr(app.store.get("main"), "elapsed", lambda: 0)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(.2)
            frames.clear()
            for _ in range(5):
                await app._tick()
                await pilot.pause()
            assert frames == []
    asyncio.run(exercise())


@pytest.mark.parametrize("width", [40, 100])
def test_tool_title_is_correct_in_first_frame_and_completion_keeps_widget(
    terminal_app, frames, wait_for_ui, width,
):
    async def exercise():
        app = terminal_app
        async with app.run_test(size=(width, 24)) as pilot:
            await pilot.pause()
            frames.clear()
            app.sink.emit(TurnStarted("turn", "检查项目"))
            start_command(app.sink)
            await wait_for_ui(pilot, lambda: bool(app.query(ToolRecord)))
            record = app.query_one(ToolRecord)
            title = record.query_one(".tool-title", Static)
            assert frames
            assert all(any("• 执行 python demo.py" in row for row in rows) for rows, _, _ in frames)
            assert_title_marker_color(title, "#a0a0a0")
            app.sink.emit(ToolOutputDelta("turn", "cmd", "stdout", "line-1\nline-2\n"))
            await wait_for_ui(pilot, lambda: record.query_one(".tool-preview").size.height == 2)
            before = record.region
            frames.clear()
            result = {"status": "exited", "exit_code": 0, "stdout": "line-1\nline-2\n", "stderr": "", "error": None}
            app.sink.emit(ToolFinished("turn", "cmd", "run_command", ToolResult(
                True, json.dumps(result), structured_output=result,
            )))
            await wait_for_ui(pilot, lambda: record.calls[0].state == "completed")
            assert app.query_one(ToolRecord) is record
            assert record.query_one(".tool-title", Static) is title
            assert record.region == before
            assert frames
            for rows, _, _ in frames:
                assert sum("python demo.py" in row for row in rows) == 1
                assert any("• 执行 python demo.py" in row for row in rows)
                assert not any(row.strip() in {"•", "…"} for row in rows)
            assert_title_marker_color(title, "green")
    asyncio.run(exercise())


@pytest.mark.parametrize("follow", [False, True])
def test_output_growth_submits_only_the_final_scroll_position(terminal_app, frames, wait_for_ui, follow):
    async def exercise():
        app = terminal_app
        view = app.store.get("main")
        view.committed_blocks.append(Text("\n".join(f"history-{i}" for i in range(60))))
        async with app.run_test(size=(80, 18)) as pilot:
            start_command(app.sink)
            await wait_for_ui(pilot, lambda: bool(app.query(ToolRecord)))
            pane = app.query_one("#transcript", VerticalScroll)
            if not follow:
                pane.scroll_to(y=7, animate=False, immediate=True)
                await wait_for_ui(pilot, lambda: not view.follow_output)
            frames.clear()
            app.sink.emit(ToolOutputDelta("turn", "cmd", "stdout", "new-1\nnew-2\nnew-3\nnew-4\n"))
            await wait_for_ui(pilot, lambda: app.query_one(".tool-preview").size.height == 4)
            assert frames
            for rows, position, maximum in frames:
                if follow:
                    assert position == maximum
                    assert any("new-4" in row for row in rows)
                else:
                    assert position == 7
            assert view.follow_output == follow
    asyncio.run(exercise())


def test_parallel_commands_keep_order_when_output_and_completion_arrive_out_of_order(
    terminal_app, wait_for_ui,
):
    async def exercise():
        app = terminal_app
        async with app.run_test(size=(100, 30)) as pilot:
            for name in ("A", "B", "C"):
                app.sink.emit(ToolCallPrepared("turn", name, "run_command", {"argv": [name]}, False))
                app.sink.emit(ToolStarted("turn", name, "run_command"))
            await wait_for_ui(pilot, lambda: len(app.query(ToolRecord)) == 3)
            records = list(app.query(ToolRecord))
            for name in ("C", "B", "A"):
                app.sink.emit(ToolOutputDelta("turn", name, "stdout", f"{name}-output\n"))
                app.sink.emit(ToolFinished("turn", name, "run_command", ToolResult(True, f"{name}-result")))
                await wait_for_ui(pilot, lambda name=name: any(r.calls[0].output == f"{name}-result" for r in records))
                assert list(app.query(ToolRecord)) == records
                assert [record.calls[0].detail for record in records] == ["A", "B", "C"]
    asyncio.run(exercise())


def test_exploration_keeps_its_group_through_running_failure_and_cancellation(terminal_app, wait_for_ui):
    async def exercise():
        app = terminal_app
        async with app.run_test(size=(80, 25)) as pilot:
            for name in ("a.py", "b.py"):
                app.sink.emit(ToolCallPrepared("turn", name, "read_file", {"relative_path": name}, False))
                app.sink.emit(ToolStarted("turn", name, "read_file"))
            await wait_for_ui(pilot, lambda: bool(app.query(ToolRecord)))
            record = app.query_one(ToolRecord)
            assert len(record.calls) == 2
            height = record.size.height
            app.sink.emit(ToolFinished("turn", "b.py", "read_file", ToolResult(False, "permission denied")))
            await wait_for_ui(pilot, lambda: record.calls[1].failed)
            assert list(app.query(ToolRecord)) == [record]
            assert record.size.height == height
            assert record.calls[0].state == "running"
            app.sink.emit(TurnCancelled("turn"))
            await wait_for_ui(pilot, lambda: record.calls[0].state == "cancelled")
            assert list(app.query(ToolRecord)) == [record]
            assert not app.store.get("main").has_active_tools
            assert record.calls[1].output == "permission denied"
            stopped = app.query_one("#record-1", Static)
            assert stopped.render_line(0).text.strip() == "• 已停止本轮执行"
            assert stopped.content_region.x == record.query_one(".tool-title", Static).content_region.x
            for segment in stopped.render_line(0):
                if "•" in segment.text:
                    assert segment.text.strip() == "•"
                    assert segment.style.color.name == "red" and not segment.style.dim
                elif segment.text.strip():
                    assert segment.style.color == stopped.rich_style.color and segment.style.dim
    asyncio.run(exercise())


def test_end_of_stream_does_not_remove_preview_before_tool_result():
    view = AgentView("main", "主 Agent")
    view.consume(ToolOutputDelta("turn", "cmd", "stdout", "last output", command="demo"))
    block, = view.committed_blocks
    assert block.detail == "demo"
    view.consume(ToolOutputDelta("turn", "cmd", "", "", finished=True))
    assert view.committed_blocks == [block]
    view.consume(ToolFinished("turn", "cmd", "run_command", ToolResult(True, "final output")))
    assert len(view.committed_blocks) == 1
    assert view.committed_blocks[0].output == "final output"
    assert view.committed_blocks[0].detail == "demo"
    assert not view.has_active_tools and not view._command_output.outputs


def test_child_history_titles_survive_switching_scrolling_and_resize(terminal_app, frames, wait_for_ui):
    async def exercise():
        app = terminal_app
        app.sink.emit(ChildAgentStatus("root", "child", "running", "检查工具"))
        for i in range(12):
            for event in (
                ToolCallPrepared("child-turn", str(i), "run_command", {"argv": [f"command-{i}"]}, False),
                ToolFinished("child-turn", str(i), "run_command", ToolResult(True, f"output-{i}")),
            ):
                app.sink.emit(ChildAgentEvent("root", "child", event))
        async with app.run_test(size=(80, 18)) as pilot:
            await wait_for_ui(pilot, lambda: bool(app.query("#agent-1")))
            frames.clear()
            await pilot.press("alt+right")
            await wait_for_ui(pilot, lambda: app.selected_agent_id == "child" and len(app.query(ToolRecord)) == 12)
            assert any("• 执行 command-11" in row for row in frames[-1][0])
            await pilot.press("ctrl+home")
            await pilot.pause()
            assert any("• 执行 command-0" in row for row in frames[-1][0])
            await pilot.resize_terminal(44, 22)
            await pilot.press("alt+left", "alt+right")
            await pilot.pause()
            assert any("• 执行 command-0" in row for row in frames[-1][0])
            for rows, _, _ in frames:
                assert not any(row.strip() in {"•", "…"} for row in rows)
    asyncio.run(exercise())


def test_clock_update_does_not_resend_tool_text(terminal_app, monkeypatch):
    async def exercise():
        app = terminal_app
        app.sink.emit(TurnStarted("turn", "检查项目"))
        start_command(app.sink)
        elapsed = [0]
        monkeypatch.setattr(app.store.get("main"), "elapsed", lambda: elapsed[0])
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(.2)
            writes = []
            display = app._display

            def capture(screen, renderable):
                if renderable is not None:
                    writes.append(renderable.render_segments(app.console))
                display(screen, renderable)

            monkeypatch.setattr(app, "_display", capture)
            elapsed[0] = 10
            await app._tick()
            await pilot.pause()
            assert writes and any("0:10" in text for text in writes)
            assert not any("demo.py" in text for text in writes)
    asyncio.run(exercise())


@pytest.mark.parametrize("state,color", [
    ("completed", "green"), ("failed", "red"), ("running", "#a0a0a0"), ("cancelled", "#a0a0a0"),
])
@pytest.mark.parametrize("width", [44, 100])
def test_tool_status_color_stays_on_marker_after_wrapping(terminal_app, state, color, width):
    async def exercise():
        app = terminal_app
        app.store.get("main").committed_blocks.append(ToolBlock(
            "执行", "{}", "plain output", state, "run_command",
            "python demo.py --argument " + "long-command-argument " * 10,
        ))
        async with app.run_test(size=(width, 20)) as pilot:
            await pilot.pause()
            title = app.query_one(".tool-title", Static)
            assert title.size.height == 2
            assert title.render_line(0).text.startswith("• 执行 python demo.py")
            assert_title_marker_color(title, color)
            preview = app.query_one(".tool-preview", Static)
            assert all(segment.style.dim for segment in preview.render_line(0) if segment.text.strip())
    asyncio.run(exercise())


def test_group_failure_only_colors_the_marker(terminal_app):
    async def exercise():
        app = terminal_app
        app.store.get("main").committed_blocks.extend([
            ToolBlock("读取", "{}", "contents", name="read_file", detail="a.py"),
            ToolBlock("读取", "{}", "missing", "failed", "read_file", "b.py"),
        ])
        async with app.run_test(size=(100, 20)) as pilot:
            await pilot.pause()
            title = app.query_one(".tool-title", Static)
            assert title.render_line(0).text.strip() == "• 查看代码 · 2 次操作"
            assert_title_marker_color(title, "red")
            preview = app.query_one(".tool-preview", Static)
            for row, path in ((0, "a.py"), (1, "b.py")):
                assert f"读取 {path}" in preview.render_line(row).text
                assert all(segment.style.color == preview.rich_style.color and segment.style.dim
                           for segment in preview.render_line(row) if segment.text.strip())
    asyncio.run(exercise())


def test_cursor_changes_stay_inside_synchronized_frame_and_style_is_not_resent(terminal_app):
    app = terminal_app
    writes = []
    app._driver = SimpleNamespace(write=writes.append, is_inline=False)
    app._on_terminal_supports_synchronized_output(messages.TerminalSupportsSynchronizedOutput())
    app._show_terminal_cursor = True
    app._begin_update()
    writes.append("frame-1")
    app._end_update()
    assert writes == ["\x1b[?2026h", "frame-1", "\x1b[6 q", "\x1b[?25h", "\x1b[?2026l"]
    writes.clear()
    app._begin_update()
    writes.append("frame-2")
    app._end_update()
    assert writes == ["\x1b[?2026h", "\x1b[?25l", "frame-2", "\x1b[?25h", "\x1b[?2026l"]
    writes.clear()
    app._show_terminal_cursor = False
    app._begin_update()
    app._end_update()
    assert writes == ["\x1b[?2026h", "\x1b[?25l", "\x1b[?2026l"]
