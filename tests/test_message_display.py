"""Internal model labels stay in source data, never in the rendered conversation."""

import asyncio
from io import StringIO

import pytest
from rich.console import Console

from praxis_agent.events import (
    AssistantTextCompleted,
    AssistantTextDelta,
    ModelRequestCompleted,
    ModelRequestStarted,
    TurnCancelled,
    TurnCompleted,
    TurnFailed,
)
from praxis_agent.model import ModelResponse
from praxis_agent.terminal.app import DetailsScreen
from praxis_agent.terminal.cells import MessageBlock
from praxis_agent.terminal.selection import SelectableText
from praxis_agent.terminal.views import AgentView


def rendered(content):
    output = StringIO()
    Console(file=output, width=80, color_system=None).print(content)
    return output.getvalue()


@pytest.mark.parametrize("prefix", [
    "[message_id=m15]\n",
    "[message_id=m123, operation_id=op2]\r\n",
    "[message_id=m0123456789abcdef0123456789abcdef]\n",
    "[message_id=mabcdef0123456789abcdef0123456789, operation_id=op18]\n",
    "[message_id=m1]\n[message_id=m2, operation_id=op3]\n",
    "[message_id=m15] \t\n",
])
@pytest.mark.parametrize("chunk_size", [1, 3, 17])
def test_metadata_never_flashes_during_streaming_and_source_is_unchanged(prefix, chunk_size):
    view = AgentView("main", "主 Agent")
    view.consume(ModelRequestStarted("turn", 1))
    source = prefix + "已经找到源码。\n\n    keep_indentation()"
    for start in range(0, len(source), chunk_size):
        view.consume(AssistantTextDelta("turn", 1, source[start:start + chunk_size]))
        block, = view.committed_blocks
        assert block.text == source[:start + chunk_size]
        assert "message" not in block.display_text and "operation" not in block.display_text
        assert "[" not in rendered(block)
        if start + chunk_size <= len(prefix):
            assert block.display_text == ""
    view.consume(ModelRequestCompleted("turn", 1, ModelResponse((), source, ())))
    view.consume(AssistantTextCompleted("turn", source))
    assert view.committed_blocks == [block]
    assert block.text == source
    assert block.display_text == "已经找到源码。\n\n    keep_indentation()"
    assert not block.streaming
    assert "message_id" not in rendered(view.transcript_content())


@pytest.mark.parametrize("source", [
    "正文讨论 [message_id=m15] 的含义。",
    "[message_id=m15] 是内部标签的示例。",
    "`[message_id=m15]`\n这是一段代码。",
    "```text\n[message_id=m15]\n```",
    "> [message_id=m15]\n这是一段引用。",
    "    [message_id=m15]\n    indented code",
    "[message_id=hello]\n合法正文",
    "[message_id=mabc]\n不是生成的 ID",
    "[message_id=m15, operation_id=other]\n不是内部标签",
    "[message_id=m15, extra=value]\n保留未知格式",
    "[message_type=example]",
    "[普通链接](https://example.com)",
])
def test_prose_code_quotes_and_invalid_labels_survive_streaming(source):
    block = MessageBlock("", streaming=True)
    for char in source:
        block.update(block.text + char, streaming=True)
    assert block.display_text == source
    block.update(source)
    assert block.display_text == source and block.text == source


@pytest.mark.parametrize("source", ["[", "[message_", "[message_id=m15", "[message_id=m1, operation_id=o"])
@pytest.mark.parametrize("completion", [
    ModelRequestCompleted("turn", 1), TurnCompleted("turn"),
    TurnFailed("turn", "NetworkError", "connection lost"), TurnCancelled("turn"),
])
def test_incomplete_prefix_is_released_when_stream_ends(source, completion):
    view = AgentView("main", "主 Agent")
    view.consume(AssistantTextDelta("turn", 1, source))
    block, = view.committed_blocks
    assert block.display_text == ""
    revision = block.revision
    view.consume(completion)
    assert block.display_text == source
    assert block.revision > revision  # The mounted widget must be redrawn too.


def test_labels_in_user_input_and_later_in_assistant_text_are_preserved():
    source = "[message_id=m15]\n这是用户输入。"
    assert MessageBlock(source, user=True, streaming=True).display_text == source
    body = "正文\n[message_id=m16]\n这里只是一个示例。"
    assert MessageBlock("[message_id=m15]\n" + body).display_text == body


@pytest.mark.parametrize("restore", [False, True])
def test_live_history_details_and_copy_use_same_display_text(
    terminal_app, wait_for_ui, drag_selection, restore,
):
    async def exercise():
        app = terminal_app
        source = "[message_id=m15, operation_id=op2]\n已经找到源码。"
        view = app.store.get("main")
        if restore:
            view.restore_history([{"type": "message", "role": "assistant", "content": source}])
        async with app.run_test(size=(80, 26)) as pilot:
            if not restore:
                app.sink.emit(AssistantTextDelta("turn", 1, "[message_id=m"))
                await wait_for_ui(pilot, lambda: bool(app.query("#record-0")))
                assert "message_id" not in rendered(app.query_one("#record-0").renderable)
                app.sink.emit(AssistantTextCompleted("turn", source))
            await wait_for_ui(pilot, lambda: bool(app.query("#record-0")) and
                              app.query_one("#record-0").renderable.display_text == "已经找到源码。")
            widget = app.query_one("#record-0", SelectableText)
            assert widget.render_line(0).text.strip() == "• 已经找到源码。"
            await drag_selection(pilot, widget, (2, 0), (16, 0))
            await pilot.press("ctrl+c")
            assert app.clipboard == "已经找到源码。"
            await pilot.press("ctrl+t")
            assert isinstance(app.screen, DetailsScreen)
            details = app.screen.query_one("#details-content", SelectableText)
            assert "message_id" not in rendered(details.renderable)
            await drag_selection(pilot, details, (2, 0), (16, 0))
            await pilot.press("ctrl+c")
            assert app.clipboard == "已经找到源码。"
            assert view.committed_blocks[0].text == source
    asyncio.run(exercise())
