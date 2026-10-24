import asyncio
import json
from concurrent.futures import Future
from io import StringIO

import pytest
from rich.console import Console
from textual.containers import VerticalScroll
from textual.widgets import OptionList, Static

from praxis_agent.events import (
    AssistantTextCompleted,
    AssistantTextDelta,
    ChildAgentEvent,
    ChildAgentMessage,
    ModelRequestCompleted,
    ModelRequestStarted,
    ToolCallPrepared,
    ToolFinished,
    ToolStarted,
    TurnCompleted,
    TurnStarted,
)
from praxis_agent.model import ModelResponse
from praxis_agent.terminal.app import (
    Approval,
    ApprovalPanel,
    AgentTab,
    DetailsScreen,
    PraxisApp,
    ToolRecord,
)
from praxis_agent.terminal.cells import MessageBlock, display_command
from praxis_agent.terminal.views import AgentView, ToolBlock, ViewEventSink
from praxis_agent.tools import ToolResult


def render(content, width=80):
    output = StringIO()
    Console(file=output, width=width, color_system=None).print(content)
    return output.getvalue()


def finish_tool(sink, call_id, name, arguments, output, success=True):
    sink.emit(ToolCallPrepared("turn", call_id, name, arguments, False))
    result = ToolResult(success, json.dumps(output, ensure_ascii=False), structured_output=output) if isinstance(output, dict) else ToolResult(success, output)
    sink.emit(ToolFinished("turn", call_id, name, result))


def test_multiline_command_heading_does_not_claim_missing_output(wait_for_ui):
    async def exercise():
        sink = ViewEventSink()
        app = PraxisApp(sink)
        script = "\n".join(f"print('marker-{i}')" for i in range(24))
        arguments = {"argv": ["python", "-c", script]}
        async with app.run_test(size=(60, 28)) as pilot:
            sink.emit(ToolCallPrepared("turn", "cmd", "run_command", arguments, False))
            sink.emit(ToolStarted("turn", "cmd", "run_command"))
            await wait_for_ui(pilot, lambda: bool(app.query(ToolRecord)))
            record = app.query_one(ToolRecord)
            title = record.query_one(".tool-title", Static)
            shown = "\n".join(title.render_line(y).text for y in range(title.size.height))
            assert "omitted" not in shown and "marker-0" in shown and "…" in shown
            assert title.size.height <= 2
            assert not record.query_one(".tool-preview", Static).display
            assert json.loads(record.calls[0].arguments) == arguments
            sink.emit(ToolFinished("turn", "cmd", "run_command", ToolResult(True, "\n".join(map(str, range(10))))))
            await wait_for_ui(pilot, lambda: record.calls[0].state == "completed")
            assert "6 lines omitted" in render(record.query_one(".tool-preview", Static).renderable)
            await pilot.press("ctrl+t")
            await pilot.pause()
            details = render(app.screen.query_one("#details-content", Static).renderable, 200)
            assert "marker-23" in details
    asyncio.run(exercise())


@pytest.mark.parametrize("width", [48, 100])
def test_streamed_markdown_keeps_one_message_and_matches_completed_layout(width):
    view = AgentView("main", "主 agent")
    view.consume(ModelRequestStarted("turn", 1))
    source = "第一段 **重点**。\n\n第二段文字。\n\n- 第一项\n- 第二项\n\n```python\nprint('ok')\n```"
    for start in range(0, len(source), 7):
        view.consume(AssistantTextDelta("turn", 1, source[start:start + 7]))
    message, = view.committed_blocks
    assert isinstance(message, MessageBlock)
    before = render(message, width)
    assert sum(line.startswith("•") for line in before.splitlines()) == 1
    assert "**" not in before
    assert "```" not in before
    assert "print('ok')" in before
    view.consume(ModelRequestCompleted("turn", 1, ModelResponse((), source, ())))
    view.consume(AssistantTextCompleted("turn", source))
    view.consume(TurnCompleted("turn"))
    assert view.committed_blocks == [message]
    assert render(message, width) == before


def test_grouped_reads_update_in_place_and_keep_full_details():
    async def exercise():
        sink = ViewEventSink()
        app = PraxisApp(sink)
        async with app.run_test(size=(70, 28)) as pilot:
            sink.emit(TurnStarted("turn", "检查文件"))
            finish_tool(sink, "a", "read_file", {"relative_path": "src/[red].py"}, "first full result")
            await pilot.pause(0.15)
            record = app.query_one(ToolRecord)
            assert record.size.height == 1
            title = record.query_one(".tool-title", Static)
            assert "[red].py" in render(title.renderable)
            assert render(title.renderable).startswith("• ")
            await pilot.click(title)
            await pilot.pause(0.1)
            assert record.size.height == 1
            finish_tool(sink, "b", "read_file", {"relative_path": "src/b.py"}, "second full result")
            await pilot.pause(0.15)
            assert app.query_one(ToolRecord) is record
            assert "2 operations" in render(title.renderable)
            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            assert isinstance(app.screen, DetailsScreen)
            details = render(app.screen.query_one("#details-content", Static).renderable)
            assert all(value in details for value in ("src/[red].py", "src/b.py", "first full result", "second full result"))
            # Grouping is presentation-only: original results are still separate and intact.
            original = [block for block in app.store.views["main"].committed_blocks if isinstance(block, ToolBlock)]
            assert [block.output for block in original] == ["first full result", "second full result"]
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert record.size.height == 3
            assert record.query_one(".tool-preview", Static).display
            finish_tool(sink, "c", "read_file", {"relative_path": "missing.py"}, "file missing", success=False)
            await pilot.pause(0.15)
            # A failure updates its existing group without moving other records.
            assert len(app.query(ToolRecord)) == 1
            assert render(title.renderable).strip() == "• Explore code · 3 operations"
            assert "".join(segment.text for segment in title.render_line(0)
                           if segment.style.color.name == "red").strip() == "•"
            assert "Read missing.py" in render(record.query_one(".tool-preview", Static).renderable)
    asyncio.run(exercise())


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_long_read_title_keeps_filename_on_resize_and_full_path_in_details(separator):
    async def exercise():
        sink = ViewEventSink()
        app = PraxisApp(sink)
        filename = "AbstractFltFilSegmentSearchTokenStrategy.java"
        path = separator.join(("src", "main", "java", "com", "ctrip", "flight", "搜索模块",
                               "aggregatorsearch", "strategy", filename))
        async with app.run_test(size=(60, 28)) as pilot:
            finish_tool(sink, "read", "read_file", {"relative_path": path}, "complete file contents")
            await pilot.pause(0.15)
            record = app.query_one(ToolRecord)
            title = record.query_one(".tool-title", Static)
            for width in (38, 60, 100, 180):
                await pilot.resize_terminal(width, 28)
                await pilot.pause()
                shown = "\n".join(title.render_line(y).text.rstrip() for y in range(title.size.height))
                assert shown.startswith("• Read")
                assert "Read completed" not in shown
                if width == 38:
                    assert "AbstractFlt" in shown and ".java" in shown
                else:
                    assert filename in shown
                assert "omitted" not in shown
                assert record.size.height <= 2
                if width == 180:
                    assert path in shown
            await pilot.press("ctrl+t")
            await pilot.pause()
            details = render(app.screen.query_one("#details-content", Static).renderable, 200)
            assert json.dumps(path, ensure_ascii=False)[1:-1] in details
            assert "complete file contents" in details
    asyncio.run(exercise())


@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("refresh_first", [False, True])
def test_switching_agent_during_refresh_keeps_records_and_cache_consistent(clear, refresh_first):
    async def exercise():
        sink = ViewEventSink()
        app = PraxisApp(sink)
        app.store.get("main").committed_blocks.append(MessageBlock("Main page", user=True))
        child = app.store.get("child")
        child.committed_blocks.extend(MessageBlock(f"Child record {i}") for i in range(31))
        async with app.run_test(size=(100, 30)) as pilot:
            await app._update_agents()
            await pilot.pause()
            sink.emit(AssistantTextDelta("root", 1, "Main update"))
            sink.emit(ChildAgentEvent("root", "child", AssistantTextDelta("child_turn", 1, "New text")))
            if clear:
                app._results.put((None, "clear", ""))
            operations = [app.on_agent_selected(AgentTab.Selected("child")), app._tick()]
            await asyncio.gather(*(reversed(operations) if refresh_first else operations))
            await pilot.pause()
            assert app.selected_agent_id == ("main" if clear else "child")
            widgets = list(app.query_one("#blocks").children)
            expected = app.store.get(app.selected_agent_id).display_blocks()
            assert len(widgets) == len(expected) == (0 if clear else 32)
            assert [widget.id for widget in widgets] == [f"record-{i}" for i in range(len(expected))]
            assert [widget.renderable for widget in widgets] == expected
            assert app._mounted_blocks == expected
            assert app._mounted_agent == app.selected_agent_id
            assert len(app._mounted_revisions) == len(expected)
            if clear:
                assert len(app.query(AgentTab)) == 1
                await app.on_agent_selected(AgentTab.Selected("child"))
                assert app.selected_agent_id == "main"
    asyncio.run(exercise())


def test_command_preview_is_bounded_and_full_output_is_in_transcript():
    async def exercise():
        sink = ViewEventSink()
        app = PraxisApp(sink)
        argv = ["powershell", "-NoProfile", "-Command", "python -m pytest tests -q"]
        output = {"status": "exited", "exit_code": 0, "stdout": "\n".join(f"result {i}" for i in range(10)),
                  "stderr": "", "error": None}
        async with app.run_test(size=(70, 28)) as pilot:
            finish_tool(sink, "cmd", "run_command", {"argv": argv}, output)
            await pilot.pause(0.15)
            record = app.query_one(ToolRecord)
            assert record.size.height == 6
            preview = render(record.query_one(".tool-preview", Static).renderable)
            assert all(f"result {i}" in preview for i in (0, 1, 8, 9))
            assert "6 lines omitted" in preview
            assert "result 6" not in preview
            assert "stdout:" not in preview and "<empty>" not in preview
            assert "-NoProfile" not in render(record.query_one(".tool-title", Static).renderable)
            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            assert isinstance(app.screen, DetailsScreen)
            full = render(app.screen.query_one("#details-content", Static).renderable)
            assert "-NoProfile" in full and "result 0" in full and "result 9" in full
            finish_tool(sink, "failed", "run_command", {"argv": ["pytest"]},
                        {"status": "exited", "exit_code": 1, "stdout": "", "stderr": "AssertionError: expected 2, got 1",
                         "error": "命令已退出，退出码为 1。"})
            await pilot.pause(0.15)
            assert "AssertionError" in render(app.screen.query_one("#details-content", Static).renderable)
            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            failed = list(app.query(ToolRecord))[-1]
            title = failed.query_one(".tool-title", Static)
            assert render(title.renderable).strip() == "• Run pytest"
            assert "".join(segment.text for segment in title.render_line(0)
                           if segment.style.color.name == "red").strip() == "•"
            assert "AssertionError" in render(failed.query_one(".tool-preview", Static).renderable)
    asyncio.run(exercise())


def test_live_message_survives_correction_resize_and_reading_history():
    async def exercise():
        sink = ViewEventSink()
        app = PraxisApp(sink)
        async with app.run_test(size=(80, 28)) as pilot:
            sink.emit(ModelRequestStarted("turn", 1))
            source = "开始检查。\n\n```python\n" + "print('evidence')\n" * 150
            sink.emit(AssistantTextDelta("turn", 1, source))
            await pilot.pause(0.2)
            message = app.query_one("#record-0", Static)
            pane = app.query_one("#transcript", VerticalScroll)
            assert message.size.height > 150
            assert pane.scrollbar_size_vertical == 0
            assert pane.scrollbar_size_horizontal == 0
            assert app.screen.max_scroll_y == 0
            assert render(message.renderable).count("print('evidence')") == 150
            pane.scroll_to(y=12, animate=False, immediate=True)
            await pilot.pause(0.1)
            sink.emit(AssistantTextDelta("turn", 1, "print('last line')\n```"))
            await pilot.pause(0.15)
            assert app.query_one("#record-0", Static) is message
            assert pane.scroll_y == 12
            await pilot.resize_terminal(48, 25)
            await pilot.pause(0.1)
            assert render(message.renderable, 44).count("print('evidence')") == 150
            sink.emit(ChildAgentMessage("turn", "worker", "report"))
            sink.emit(ModelRequestCompleted("turn", 1, ModelResponse((), "最终 **结论**。", ())))
            await pilot.pause(0.15)
            assert app.query_one("#record-0", Static) is message
            assert "最终 结论" in render(message.renderable)
            assert any("new message" in render(widget.renderable) for widget in app.query("#blocks Static"))
            assert pane.scroll_y <= pane.max_scroll_y
    asyncio.run(exercise())


def test_shell_display_only_unwraps_known_script_arguments():
    assert display_command({"argv": ["bash", "-lc", "pytest -q"]}) == "pytest -q"
    assert display_command({"argv": ["python", "-c", "print(1)"]}).startswith("python -c ")
    assert display_command({"argv": ["pwsh.exe", "-Command", "pytest", "--version"]}).startswith("pwsh.exe ")


@pytest.mark.parametrize("keys,allowed", [(["y"], True), (["1"], True), (["n"], False),
                                        (["2"], False), (["escape"], False), (["down", "enter"], False)])
def test_inline_approval_uses_existing_boolean_decision(keys, allowed):
    async def exercise():
        app = PraxisApp(ViewEventSink())
        request = Approval("run_command", {"argv": ["cmd", "/c", "echo hello"],
                           "workdir": ".", "yield_time_ms": 30000}, Future())
        async with app.run_test(size=(70, 28)) as pilot:
            app._approvals.put(request)
            await pilot.pause(0.15)
            panel = app.query_one(ApprovalPanel)
            assert panel.display
            assert app.screen is app._main_screen
            assert not app.query_one("#composer").display
            assert panel.region.bottom <= app.screen.size.height
            content = render(panel.query_one("#approval-content", Static).renderable)
            assert "$ echo hello" in content and "argv" not in content
            assert "Directory: ." in content
            await pilot.press(*keys)
            await pilot.pause(0.1)
            assert request.future.result(timeout=0) is allowed
            assert not panel.display
            assert app.query_one("#composer").display
    asyncio.run(exercise())


def test_approval_details_do_not_resolve_request_and_queue_resets_selection():
    async def exercise():
        app = PraxisApp(ViewEventSink())
        first = Approval("run_command", {"argv": ["cmd", "/c", "echo " + "long " * 60]}, Future())
        second = Approval("write_file", {"relative_path": "test.txt", "content": "hello"}, Future())
        async with app.run_test(size=(48, 25)) as pilot:
            app._approvals.put(first)
            app._approvals.put(second)
            await pilot.pause(0.15)
            panel = app.query_one(ApprovalPanel)
            assert panel.query_one(OptionList).region.bottom <= panel.region.bottom
            assert panel.query_one("#approval-hint").region.bottom <= panel.region.bottom
            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            assert isinstance(app.screen, DetailsScreen)
            assert '"argv"' in render(app.screen.query_one("#details-content", Static).renderable)
            await pilot.press("y")
            assert not first.future.done()
            assert not second.future.done()
            await pilot.press("escape")
            await pilot.press("down", "enter")
            await pilot.pause(0.2)
            assert first.future.result(timeout=0) is False
            assert app._approval is second
            assert app.query_one(OptionList).highlighted == 0
            assert app.query_one(ApprovalPanel).region.bottom <= 25
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert second.future.result(timeout=0) is True
    asyncio.run(exercise())


def test_cancelled_approval_closes_parameter_view_without_approving():
    async def exercise():
        app = PraxisApp(ViewEventSink())
        request = Approval("run_command", {"argv": ["echo", "hello"]}, Future())
        async with app.run_test(size=(70, 28)) as pilot:
            app._approvals.put(request)
            await pilot.pause(0.15)
            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            request.future.cancel()
            await pilot.pause(0.2)
            assert request.future.cancelled()
            assert app.screen is app._main_screen
            assert not app.query_one(ApprovalPanel).display
    asyncio.run(exercise())
