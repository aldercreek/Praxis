"""Regression coverage for the terminal's layout, context color and agent navigation."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual import events
from textual.containers import VerticalScroll
from textual.widgets import Static

from praxis_agent import __version__
from praxis_agent.context.capacity import ContextPressure, ContextStatus
from praxis_agent.events import (
    ChildAgentEvent,
    ChildAgentStatus,
    MessagesReceived,
    ModelRequestStarted,
    TurnCancelled,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
)
from praxis_agent.execution import Run
from praxis_agent.terminal.app import AgentStrip, AgentTab
from praxis_agent.terminal.cells import MessageBlock
from praxis_agent.terminal.composer import Composer
from praxis_agent.terminal.context import capacity_footer
from praxis_agent.terminal.views import AgentView


def capacity(percent):
    return ContextStatus(int(percent * 10), 1000, percent, ContextPressure.NORMAL)


@pytest.mark.parametrize("percent,expected", [(0, "0%"), (18.3, "18.3%"), (100, "100%"), (101.2, "101.2%")])
def test_capacity_emphasizes_percentage_only(percent, expected):
    text = capacity_footer(capacity(percent))
    assert text.plain == "上下文 ≈ " + expected
    assert len(text.spans) == 1
    span = text.spans[0]
    assert text.plain[span.start:span.end] == expected
    assert span.style == "cyan"


def test_missing_capacity_has_no_placeholder():
    assert capacity_footer(None) is None


@pytest.mark.parametrize("width", [100, 40])
def test_welcome_keeps_large_wordmark_model_and_shortened_directory(terminal_app, width):
    async def exercise():
        app = terminal_app
        app.agent = SimpleNamespace(workspace=Path("D:/Projects/long-parent-directory/Praxis"),
                                    session_model="auto(free)")
        async with app.run_test(size=(width, 20)) as pilot:
            await pilot.pause()
            welcome = app.query_one("#welcome", Static)
            lines = [welcome.render_line(y).text.rstrip() for y in range(welcome.size.height)]
            assert lines[:3] == [
                "█▀▀█ █▀▀█ ▄▀▀▄ ▀▄ ▄▀ ▀█▀ ▄▀▀▀",
                "█▄▄█ █▄▄▀ █▄▄█   █    █  ▀▀▀▄",
                "█    █  █ █  █ ▄▀ ▀▄ ▄█▄ ▄▄▄▀",
            ]
            assert lines[3] == ""
            assert lines[4] == f"v{__version__} · 模型 auto(free)"
            assert lines[5].startswith("目录 ") and lines[5].endswith("Praxis")
            assert ("…" in lines[5]) == (width == 40)
            assert app.query_one("#composer").region.y < 10
    asyncio.run(exercise())


@pytest.mark.parametrize("size", [(24, 20), (100, 9)])
def test_welcome_only_compacts_when_large_title_cannot_fit(terminal_app, size):
    async def exercise():
        app = terminal_app
        app.agent = SimpleNamespace(workspace=Path("D:/Projects/Praxis"), session_model="auto(free)")
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            welcome = app.query_one("#welcome", Static)
            assert welcome.render_line(0).text.rstrip() == "Praxis"
            assert app.query_one("#task").is_on_screen
            assert app.query_one("#hint").region.bottom <= size[1]
            await pilot.resize_terminal(100, 25)
            await pilot.pause()
            assert welcome.render_line(0).text.startswith("█▀▀█ █▀▀█")
            assert app.query_one("#task").is_on_screen
    asyncio.run(exercise())


@pytest.mark.parametrize("width,rows", [(100, 1), (38, 2)])
def test_context_color_survives_rendering_updates_and_agent_switches(terminal_app, wait_for_ui, width, rows):
    async def exercise():
        app = terminal_app
        async with app.run_test(size=(width, 28)) as pilot:
            app.sink.emit(TurnStarted("root", "检查项目"))
            app.sink.emit(ModelRequestStarted("root", 1, context_status=capacity(18.3)))
            app.sink.emit(ChildAgentStatus("root", "child", "running", "检查测试"))
            app.sink.emit(ChildAgentEvent("root", "child", ModelRequestStarted(
                "child", 1, context_status=capacity(42.5))))
            status = app.query_one("#status", Static)
            await wait_for_ui(pilot, lambda: "18.3%" in str(status.renderable))
            assert status.size.height == rows

            def assert_rendered_color(expected):
                segments = [segment for y in range(status.size.height) for segment in status.render_line(y)]
                colored = [s for s in segments if s.style and s.style.color and s.style.color.name == "cyan"]
                assert "".join(s.text for s in colored) == expected
                assert not any(s.style.dim for s in colored)
                assert status.region.bottom <= app.query_one("#composer").region.y

            assert_rendered_color("18.3%")
            app.sink.emit(ModelRequestStarted("root", 2, context_status=capacity(26.7)))
            await wait_for_ui(pilot, lambda: "26.7%" in str(status.renderable))
            assert_rendered_color("26.7%")
            await pilot.press("alt+right")
            await wait_for_ui(pilot, lambda: app.selected_agent_id == "child")
            assert "42.5%" in status.renderable.plain and "26.7%" not in status.renderable.plain
            colored = [s.text for y in range(status.size.height) for s in status.render_line(y)
                       if s.style and s.style.color and s.style.color.name == "cyan"]
            assert "".join(colored) == "42.5%"
            await pilot.press("alt+left")
            await wait_for_ui(pilot, lambda: app.selected_agent_id == "main")
            assert_rendered_color("26.7%")
    asyncio.run(exercise())


def test_agent_list_is_compact_text_only_and_preserves_draft(terminal_app, wait_for_ui):
    async def exercise():
        app = terminal_app
        async with app.run_test(size=(100, 30)) as pilot:
            editor = app.query_one(Composer)
            editor.load_draft("未发送的草稿\n继续说明")
            editor.move_cursor((0, 4))
            selection = editor.selection
            app.sink.emit(TurnStarted("root", "分析项目"))
            app.sink.emit(ChildAgentStatus("root", "child", "running", "分析测试"))
            await wait_for_ui(pilot, lambda: len(app.query(AgentTab)) == 2)
            rows = list(app.query(AgentTab))
            assert rows[0].renderable.plain[1:] == " 主 Agent"
            assert rows[1].renderable.plain[1:] == " Agent 1"
            assert rows[1].region.y == rows[0].region.y
            assert rows[1].region.x >= rows[0].region.right + 2
            assert all(row.tooltip is None for row in rows)
            assert not app.query("#agent-toggle, #return-main, #readonly, #agent-picker")
            assert all(binding.key != "ctrl+g" for binding in app.BINDINGS)

            await pilot.click(rows[1])
            await wait_for_ui(pilot, lambda: app.selected_agent_id == "child")
            assert rows[1].styles.background.a == 0
            assert rows[1].styles.text_style.bold
            assert rows[1].styles.color.hex == "#8AB8C5"
            assert not rows[0].styles.text_style.bold
            assert editor.read_only and not app.query_one("#composer").display
            app.query_one("#transcript", VerticalScroll).focus()
            await pilot.pause()
            assert rows[1].styles.text_style.bold  # Selection must survive loss of focus.
            assert rows[1].styles.background.a == 0

            await pilot.click(rows[0])
            await wait_for_ui(pilot, lambda: app.selected_agent_id == "main")
            assert app.query_one(Composer) is editor
            assert editor.text == "未发送的草稿\n继续说明"
            assert editor.selection == selection
            assert not editor.read_only and app.query_one("#composer").display
            await pilot.press("down", "up")
            assert app.selected_agent_id == "main"
    asyncio.run(exercise())


@pytest.mark.parametrize("size", [(100, 30), (36, 18)])
def test_agent_status_and_input_align_with_one_blank_row_around_editor(terminal_app, wait_for_ui, size):
    async def exercise():
        app = terminal_app
        async with app.run_test(size=size) as pilot:
            app.sink.emit(TurnStarted("main", "检查项目"))
            app.sink.emit(ChildAgentStatus("main", "child", "running", "检查测试"))
            await wait_for_ui(pilot, lambda: bool(app.query("#agent-1")))
            editor = app.query_one(Composer)
            editor.load_draft("第一行输入\n第二行输入")
            await wait_for_ui(pilot, lambda: editor.size.height == 2)
            status, prompt = app.query_one("#status"), app.query_one("#prompt")
            first = app.query_one("#agent-0")
            assert first.region.x == status.content_region.x == prompt.region.x == 1
            assert editor.region.x == first.region.x + 2
            assert editor.region.y == status.region.bottom + 1
            assert app.query_one("#hint").region.y == editor.region.bottom + 1
            assert app.query_one("#agent-bar").size.height == 1
            assert app.query_one("#transcript").size.height >= 1
            await pilot.press("alt+right")
            await wait_for_ui(pilot, lambda: app.selected_agent_id == "child")
            assert first.region.x == status.content_region.x
            assert not app.query_one("#composer").display
    asyncio.run(exercise())


def test_running_indicators_remain_static_across_ticks_and_cancellation(terminal_app, wait_for_ui, monkeypatch):
    async def exercise():
        app = terminal_app
        app.animation_level = "full"
        app.busy, app.current_run = True, Run("main")
        async with app.run_test(size=(80, 24)) as pilot:
            app.sink.emit(TurnStarted("main", "主任务"))
            app.sink.emit(ChildAgentStatus("main", "child", "running", "子任务"))
            await wait_for_ui(pilot, lambda: len(app.query(AgentTab)) == 2)
            for timestamp in (100.0, 100.1, 100.5):
                monkeypatch.setattr("praxis_agent.terminal.app.monotonic", lambda: timestamp)
                await app._tick()
                await pilot.pause()
                assert [tab.renderable.plain for tab in app.query(AgentTab)] == ["• 主 Agent", "• Agent 1"]
                assert app.query_one("#status").renderable.plain.startswith("• ")
            await pilot.press("escape")
            assert app.query_one("#status").renderable.plain.startswith("• 正在停止")
            assert [tab.renderable.plain for tab in app.query(AgentTab)] == ["• 主 Agent", "• Agent 1"]
    asyncio.run(exercise())


def test_overlapping_refreshes_do_not_duplicate_new_agent_tabs(terminal_app, wait_for_ui):
    async def exercise():
        app = terminal_app
        async with app.run_test(size=(80, 24)) as pilot:
            for index in range(12):
                app.store.get(f"child-{index}")
            # Periodic updates and switching pages can overlap while mounting new widgets.
            await asyncio.gather(app._update_agents(), app._update_agents())
            await pilot.pause()
            tabs = list(app.query(AgentTab))
            assert len(tabs) == 13
            assert len({tab.agent_id for tab in tabs}) == 13
            assert len(app._tab_agents) == 13
            await pilot.press("alt+right")
            await wait_for_ui(pilot, lambda: app.selected_agent_id == "child-0")
            assert len(app.query("AgentTab.selected")) == 1
    asyncio.run(exercise())


@pytest.mark.parametrize("size", [(100, 30), (38, 16)])
def test_welcome_at_top_and_conversation_controls_at_bottom(terminal_app, wait_for_ui, size):
    async def exercise():
        app = terminal_app
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            controls = app.query_one("#controls")
            assert app.query_one("#welcome").display
            assert controls.region.bottom < size[1]
            assert not app.query_one("#agent-bar").display
            assert not app.query_one("#status").display
            app.sink.emit(TurnStarted("root", "分析项目"))
            app.sink.emit(ChildAgentStatus("root", "child", "running", "测试"))
            await wait_for_ui(pilot, lambda: len(app.query(AgentTab)) == 2)
            assert not app.query_one("#welcome").display
            assert controls.region.bottom == size[1]
            await pilot.press("alt+right")
            await wait_for_ui(pilot, lambda: app.selected_agent_id == "child")
            assert controls.region.bottom == size[1]
            assert app.query_one("#transcript").size.height >= 1
    asyncio.run(exercise())


def test_many_agents_scroll_to_selection_and_keep_small_window_usable(terminal_app, wait_for_ui):
    async def exercise():
        app = terminal_app
        async with app.run_test(size=(48, 20)) as pilot:
            app.sink.emit(TurnStarted("root", "root"))
            for index in range(12):
                app.sink.emit(ChildAgentStatus("root", f"child-{index}", "running", "很长的任务描述" * 8))
            await wait_for_ui(pilot, lambda: len(app.query(AgentTab)) == 13)
            agents = app.query_one(AgentStrip)
            assert agents.size.height == 1
            assert app.query_one("#agent-bar").size.height == 1
            await pilot.press("alt+left")
            await wait_for_ui(pilot, lambda: app.selected_agent_id == "child-11")
            row = app.query_one("#agent-12")
            assert agents.content_region.contains_region(row.region)
            assert agents.scroll_x > 0 and agents.scroll_y == 0
            assert app.query_one("#hint").region.bottom == 20
            await pilot.press("alt+right", "f1")
            await wait_for_ui(pilot, lambda: app.selected_agent_id == "main")
            assert app.query_one("#composer").region.bottom <= 19
            assert app.query_one("#status").region.y > 0
            assert app.query_one("#shortcuts").size.height >= 1
            assert "Ctrl+G" not in str(app.query_one("#shortcuts Static").renderable)
    asyncio.run(exercise())


def test_agent_wheel_scrolls_horizontally_without_moving_history_or_selection(terminal_app, wait_for_ui):
    async def exercise():
        app = terminal_app
        app.store.get("main").committed_blocks.append(MessageBlock("\n\n".join(["history"] * 50)))
        for index in range(20):
            app.store.get(f"child-{index}")
        async with app.run_test(size=(48, 22)) as pilot:
            await wait_for_ui(pilot, lambda: len(app.query(AgentTab)) == 21)
            agents = app.query_one(AgentStrip)
            pane = app.query_one("#transcript", VerticalScroll)
            position = pane.scroll_y
            assert app.query_one("#agents-prev").styles.visibility == "hidden"
            assert app.query_one("#agents-next").styles.visibility == "visible"
            # Send real pointer events over a tab; they must bubble to the strip.
            await pilot._post_mouse_events([events.MouseScrollDown], "#agent-0", offset=(2, 0), button=0)
            await wait_for_ui(pilot, lambda: agents.scroll_x > 0)
            assert pane.scroll_y == position and app.selected_agent_id == "main"
            assert app.query_one("#agents-prev").styles.visibility == "visible"
            await pilot._post_mouse_events([events.MouseScrollUp], agents, offset=(2, 0), button=0)
            await wait_for_ui(pilot, lambda: agents.scroll_x == 0)
            await pilot.click("#agents-next")
            await wait_for_ui(pilot, lambda: agents.scroll_x > 0)
            await pilot.press("alt+left")
            await wait_for_ui(pilot, lambda: app.selected_agent_id == "child-19")
            selected = app.query_one("#agent-20")
            assert agents.content_region.contains_region(selected.region)
            await pilot.resize_terminal(32, 18)
            await pilot.pause()
            assert agents.size.height == 1 and agents.content_region.contains_region(selected.region)
            # Repainting status must not snap manual scrolling back to the selected tab.
            await pilot._post_mouse_events([events.MouseScrollUp], agents, offset=(2, 0), button=0)
            offset = agents.scroll_x
            await app._tick()
            await pilot.pause()
            assert agents.scroll_x == offset
            assert app.selected_agent_id == "child-19"
            await pilot.press("alt+right")
            await wait_for_ui(pilot, lambda: app.selected_agent_id == "main")
            assert agents.scroll_x == 0 and pane.is_vertical_scroll_end
    asyncio.run(exercise())


@pytest.mark.parametrize("ending,state,symbol,color", [
    (TurnCompleted("child", occurred_at=20), "ended", "✓", "green"),
    (TurnFailed("child", "Error", "failed", occurred_at=20), "failed", "×", "red"),
    (TurnCancelled("child", occurred_at=20), "cancelled", "■", "#a0a0a0"),
])
def test_agent_ending_is_visible_retained_and_not_overwritten_by_shutdown(
    terminal_app, wait_for_ui, ending, state, symbol, color,
):
    async def exercise():
        app = terminal_app
        app.busy = True
        app.current_run = Run("main")
        async with app.run_test(size=(80, 25)) as pilot:
            app.sink.emit(TurnStarted("main", "父任务"))
            app.sink.emit(ChildAgentStatus("main", "child", "queued", "子任务"))
            await wait_for_ui(pilot, lambda: len(app.query(AgentTab)) == 2)
            tab = app.query_one("#agent-1", AgentTab)
            assert tab.renderable.plain == "○ Agent 1"
            app.sink.emit(ChildAgentEvent("main", "child", TurnStarted("child", "子任务", occurred_at=10)))
            await wait_for_ui(pilot, lambda: tab.renderable.plain == "• Agent 1")
            await pilot.press("alt+right", "escape")
            assert tab.renderable.plain == "• Agent 1"  # Cancel request is not cleanup completion.
            assert "正在停止" in app.query_one("#status").renderable.plain
            app.sink.emit(ChildAgentEvent("main", "child", ending))
            app.sink.emit(ChildAgentStatus("main", "child", "closing", occurred_at=30))
            app.sink.emit(ChildAgentStatus("main", "child", "closed", occurred_at=40))
            await wait_for_ui(pilot, lambda: tab.renderable.plain == f"{symbol} Agent 1")
            view = app.store.get("child")
            assert view.state == state and view.elapsed(now=100) == 10
            assert "正在停止" not in app.query_one("#status").renderable.plain
            assert len(app.query(AgentTab)) == 2 and app.selected_agent_id == "child"
            assert tab.styles.background.a == 0 and tab.styles.text_style.bold
            segment = next(s for s in tab.render_line(0) if symbol in s.text)
            assert segment.style.color.name == color
            assert not segment.style.dim
            assert tab.tooltip is None
            await pilot.press("alt+left", "alt+right")
            assert tab.renderable.plain == f"{symbol} Agent 1"
    asyncio.run(exercise())


@pytest.mark.parametrize("ending", [TurnCompleted("child"), TurnFailed("child", "Error", "failed"),
                                     TurnCancelled("child")])
def test_new_child_turn_resets_prior_terminal_state(ending):
    view = AgentView("child", "Agent 1")
    view.consume(TurnStarted("child", "first"))
    view.consume(ending)
    view.set_status(ChildAgentStatus("main", "child", "queued"))
    assert view.state == "queued" and view.started_at is None and view.finished_at is None
    view.set_status(ChildAgentStatus("main", "child", "running", occurred_at=100))
    view.consume(TurnStarted("child2", "second", occurred_at=101))
    assert view.state == "running" and view.finished_at is None and view.elapsed(now=103) == 2
    view.set_status(ChildAgentStatus("main", "child", "closing"))
    assert view.state == "closing" and view.phase == "正在关闭"


def test_input_wrap_resize_and_explicit_newline_preserve_text(terminal_app, wait_for_ui):
    async def exercise():
        app = terminal_app
        async with app.run_test(size=(70, 24)) as pilot:
            editor = app.query_one(Composer)
            draft = "这是一段需要自动换行的中文输入 abc " * 5
            editor.load_draft(draft)
            await wait_for_ui(pilot, lambda: editor.size.height > 1)
            assert editor.text == draft and "\n" not in editor.text
            await pilot.resize_terminal(36, 18)
            await pilot.pause()
            assert editor.text == draft and editor.cursor_location == (0, len(draft))
            assert editor.region.bottom <= 17
            await pilot.press("ctrl+j")
            assert editor.text == draft
            await pilot.press("shift+enter", "x")
            assert editor.text == draft + "\nx"
            assert editor.size.height <= 6
    asyncio.run(exercise())


def test_queued_input_visible_editable_and_consumed_into_transcript(terminal_app, wait_for_ui):
    async def exercise():
        app = terminal_app
        app.agent = SimpleNamespace()
        app.busy = True
        async with app.run_test(size=(80, 28)) as pilot:
            editor = app.query_one(Composer)
            editor.load_draft("补充检查测试")
            await pilot.press("enter")
            pending = app.query_one("#pending", Static)
            await wait_for_ui(pilot, lambda: pending.display)
            assert pending.render_line(0).text.strip() == "› 补充检查测试"
            assert all(segment.style.color.name == "#a0a0a0"
                       for segment in pending.render_line(0) if segment.text.strip())
            assert app.user_input.qsize() == 1 and not editor.text
            await pilot.press("shift+left")
            assert editor.text == "补充检查测试" and app.user_input.empty()
            await pilot.press("enter")
            task = app.user_input.get_nowait()
            app.sink.emit(MessagesReceived("root", ({"role": "user", "content": task},), source="user"))
            await wait_for_ui(pilot, lambda: bool(app.query("#record-0")))
            block = app.query_one("#record-0").renderable
            assert isinstance(block, MessageBlock) and block.user and block.text == task
            assert not pending.display
            message = app.query_one("#record-0", Static)
            body = [segment for segment in message.render_line(0) if "补充检查测试" in segment.text]
            assert body and all(segment.style.color == message.rich_style.color
                                and not segment.style.dim for segment in body)
    asyncio.run(exercise())


@pytest.mark.parametrize("child_page", [False, True])
def test_escape_from_either_page_cancels_whole_tree_without_exiting(terminal_app, wait_for_ui, child_page):
    async def exercise():
        app = terminal_app
        parent, child = Run("main"), Run("child")
        parent.add_child(child)
        app.busy, app.current_run = True, parent
        async with app.run_test(size=(80, 25)) as pilot:
            app.sink.emit(ChildAgentStatus("root", "child", "running", "测试"))
            await wait_for_ui(pilot, lambda: len(app.query(AgentTab)) == 2)
            if child_page:
                await pilot.press("alt+right")
                await wait_for_ui(pilot, lambda: app.selected_agent_id == "child")
            await pilot.press("escape")
            assert parent.cancelled.is_set() and child.cancelled.is_set()
            assert app.stopping and app.is_running
            assert "正在停止" in app.query_one("#status").renderable.plain
    asyncio.run(exercise())
