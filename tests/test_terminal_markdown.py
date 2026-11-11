"""Conversation Markdown remains readable without altering model source."""

import asyncio
from io import StringIO

import pytest
from rich.console import Console

from praxis_agent.terminal.cells import MessageBlock
from praxis_agent.events import AssistantTextCompleted
from praxis_agent.terminal.selection import SelectableText


def render(source, width=80, **kwargs):
    console = Console(file=StringIO(), width=width)
    return list(console.render(MessageBlock(source, **kwargs)))


def plain(segments):
    return "".join(segment.text for segment in segments)


def test_headings_use_codex_hierarchy_without_visible_markers():
    segments = render('# Title\n\n## Section\n\n```python\nprint("## literal")\n```')
    output = plain(segments)
    assert "# Title" not in output and "## Section" not in output
    assert 'print("## literal")' in output
    assert any("Title" in s.text and s.style.bold and s.style.underline for s in segments)
    assert any("Section" in s.text and s.style.bold for s in segments)
    assert "## User input" in plain(render("## User input", user=True))


def test_prose_emphasis_and_headings_keep_default_foreground():
    segments = render("# Title\n\n## Section\n\n**Strong** and *emphasis*.\n\n- Item")
    assert not any(s.style and s.style.color and not s.style.color.is_default for s in segments)
    assert any("Strong" in s.text and s.style.bold for s in segments)
    assert any("emphasis" in s.text and s.style.italic for s in segments)


def test_long_lists_have_uniform_spacing_and_aligned_continuations():
    source = "9. short\n10. " + "wrapped content " * 5 + "\n11. last"
    output = plain(render(source, width=32))
    lines = output.splitlines()
    first = next(i for i, line in enumerate(lines) if "9. short" in line)
    second = next(i for i, line in enumerate(lines) if "10. wrapped" in line)
    last = next(i for i, line in enumerate(lines) if "11. last" in line)
    assert not lines[first + 1].strip()
    assert not lines[last - 1].strip()
    assert lines[second + 1].startswith(" " * 6)
    compact = plain(render("1. short\n2. last"))
    compact_lines = compact.splitlines()
    first = next(i for i, line in enumerate(compact_lines) if "1. short" in line)
    assert "2. last" in compact_lines[first + 1]


def test_nested_lists_keep_all_items_and_literal_hashes():
    output = plain(render("1. **Parent**\n   - child `#literal`\n   - sibling\n2. End", width=30))
    for value in ("1. Parent", "child", "#literal", "sibling", "2. End"):
        assert value in output


@pytest.mark.parametrize("width", [26, 80])
def test_tables_keep_content_and_use_codex_mocha_colors(width):
    segments = render("| 文件 | 状态 |\n| --- | --- |\n| report.csv | 已完成 |\n| notes.txt | 待处理 |", width)
    output = plain(segments)
    for value in ("文件", "状态", "report.csv", "已完成", "notes.txt", "待处理"):
        assert value in output
    assert ("文件: " in output) == (width == 26)
    assert any("文件" in s.text and s.style.color.triplet.hex == "#f9e2af" for s in segments)
    assert any("文件" in s.text and s.style.bold for s in segments)


@pytest.mark.parametrize("closed", [False, True])
def test_streaming_code_wraps_without_losing_text(closed):
    code = 'result = "abcdefghijklmnopqrstuvwxyz0123456789"'
    source = "```python\n" + code + ("\n```" if closed else "")
    block = MessageBlock(source, streaming=not closed)
    console = Console(file=StringIO(), width=24)
    output = plain(console.render(block))
    assert "python" not in output  # Codex does not add a language label.
    assert "".join(code.split()) in "".join(output.split())
    assert block.text == source


def test_heading_copy_and_narrow_table_in_real_terminal(terminal_app, wait_for_ui, drag_selection):
    async def exercise():
        app = terminal_app
        source = "## Heading\n\n| File | Status |\n| --- | --- |\n| report.csv | Complete |"
        async with app.run_test(size=(80, 26)) as pilot:
            app.sink.emit(AssistantTextCompleted("turn", source))
            await wait_for_ui(pilot, lambda: bool(app.query("#record-0")))
            widget = app.query_one("#record-0", SelectableText)
            assert widget.render_line(0).text.strip() == "• Heading"
            await drag_selection(pilot, widget, (2, 0), (9, 0))
            await pilot.press("ctrl+c")
            assert app.clipboard == "Heading"
            await pilot.resize_terminal(28, 26)
            await pilot.pause()
            output = "\n".join(widget.render_line(y).text for y in range(widget.size.height))
            assert "File: report.csv" in output
            assert "Status: Complete" in output
            assert widget.renderable.text == source
    asyncio.run(exercise())
