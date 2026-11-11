"""Codex CLI-inspired Markdown, using its default dark Catppuccin Mocha palette.

Reference: openai/codex rust-v0.160.1, tui/src/markdown_render.rs and render/highlight.rs.
Pygments tokenization approximates Codex's syntect scopes; wrapping uses Rich.
"""

from typing import ClassVar

from pygments.token import Comment, Keyword, Name, Number, Operator, String, Token
from rich import box
from rich.markdown import BlockQuote, CodeBlock, Heading, ListElement, ListItem, Markdown, TableElement
from rich.segment import Segment
from rich.style import Style
from rich.syntax import ANSISyntaxTheme, Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme


CODE_THEME = ANSISyntaxTheme({
    Token: Style(color="#cdd6f4"),
    Keyword: Style(color="#cba6f7"),
    Comment: Style(color="#9399b2", italic=True),
    String: Style(color="#a6e3a1"),
    Number: Style(color="#fab387"),
    Operator: Style(color="#94e2d5"),
    Name.Function: Style(color="#89b4fa", italic=True),
    Name.Class: Style(color="#f9e2af", italic=True),
    Name.Builtin: Style(color="#fab387"),
    Name.Decorator: Style(color="#fab387", italic=True),
})
MARKDOWN_THEME = Theme({
    "markdown.h1": "bold underline",
    "markdown.h2": "bold",
    "markdown.h3": "bold italic",
    **{f"markdown.h{level}": "italic" for level in range(4, 7)},
    "markdown.strong": "bold",
    "markdown.em": "italic",
    "markdown.code": "#a6e3a1",
    "markdown.code_block": "none",
    "markdown.block_quote": "green",
    "markdown.hr": "dim",
    "markdown.link": "underline #63a8f8",
    "markdown.link_url": "underline #63a8f8",
    "markdown.item.number": "bright_blue",
    "markdown.item.bullet": "none",
})

# Codex uses a heavy header rule, light body rules, and no enclosing border.
TABLE_BOX = box.Box("    \n    \n ━ ━\n    \n ─ ─\n ─ ─\n    \n    \n")
TABLE_HEADER = "bold #f9e2af"
# Codex blends foreground 20% toward background; Praxis uses #f2f2f2 on #0c0c0c.
TABLE_SEPARATOR = "#3a3a3a"


class ConversationHeading(Heading):
    def __rich_console__(self, console, options):
        # Praxis intentionally omits Codex's visible heading markers.
        heading = self.text.copy()
        heading.justify = "left"
        yield heading


class ConversationCodeBlock(CodeBlock):
    def __rich_console__(self, console, options):
        syntax = Syntax(
            self.text.plain.rstrip("\n"), self.lexer_name,
            theme=self.theme, word_wrap=True, padding=0,
        )
        # Parent tables default to ellipsis; code must retain long tokens in full.
        yield from console.render(syntax, options.update(overflow="fold", no_wrap=False))


class ConversationQuote(BlockQuote):
    def __rich_console__(self, console, options):
        lines = console.render_lines(
            self.elements, options.update(width=max(1, options.max_width - 2)), style=self.style,
        )
        for line in lines:
            yield Segment("> ", self.style)
            yield from line
            yield Segment("\n")


class ConversationListItem(ListItem):
    def render_number(self, console, options, number, last_number):
        width = len(str(last_number)) + 2
        lines = console.render_lines(
            self.elements, options.update(width=max(1, options.max_width - width)),
            style=self.style,
        )
        style = console.get_style("markdown.item.number")
        for index, line in enumerate(lines):
            yield Segment(f"{number}. ".rjust(width) if index == 0 else " " * width, style)
            yield from line
            yield Segment("\n")


class ConversationList(ListElement):
    def __rich_console__(self, console, options):
        start = self.list_start if self.list_start is not None else 1
        rendered = []
        for index, item in enumerate(self.items):
            segments = (
                item.render_bullet(console, options)
                if self.list_type == "bullet_list_open"
                else item.render_number(console, options, start + index, start + len(self.items) - 1)
            )
            rendered.append(list(segments))
        # Keep short lists compact; separate every sibling when any item wraps.
        # This follows Codex's consolidated list layout at the current width.
        multiline = any(sum(s.text.count("\n") for s in item) > 1 for item in rendered)
        for index, item in enumerate(rendered):
            if index and multiline:
                yield Segment("\n")
            yield from item


class ConversationTable(TableElement):
    def __rich_console__(self, console, options):
        headers = (
            [cell.content for cell in self.header.row.cells]
            if self.header is not None and self.header.row is not None else []
        )
        rows = (
            [[cell.content for cell in row.cells] for row in self.body.rows]
            if self.body is not None else []
        )
        # Below twelve cells per column, labelled records are easier to read.
        if rows and options.max_width < len(headers) * 12 + max(0, len(headers) - 1) * 2:
            for index, row in enumerate(rows):
                if index:
                    yield Text("")
                for header, value in zip(headers, row):
                    label = header.copy()
                    label.stylize(TABLE_HEADER)
                    yield label + Text(": ") + value
            return
        table = Table(
            box=TABLE_BOX, pad_edge=True, padding=(0, 1), collapse_padding=False,
            show_edge=False, show_lines=True,
            border_style=TABLE_SEPARATOR, header_style=TABLE_HEADER,
        )
        for header in headers:
            table.add_column(header)
        for row in rows:
            table.add_row(*row)
        yield table


class ConversationMarkdown(Markdown):
    elements: ClassVar[dict] = {
        **Markdown.elements,
        "heading_open": ConversationHeading,
        "table_open": ConversationTable,
        "fence": ConversationCodeBlock,
        "code_block": ConversationCodeBlock,
        "blockquote_open": ConversationQuote,
        "bullet_list_open": ConversationList,
        "ordered_list_open": ConversationList,
        "list_item_open": ConversationListItem,
    }
