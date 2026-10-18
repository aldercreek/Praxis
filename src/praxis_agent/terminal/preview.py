"""Shared, screen-row-bounded previews for live and completed operations."""

from dataclasses import dataclass

from rich.text import Text


@dataclass
class OutputPreview:
    lines: list[Text]
    max_rows: int = 5
    prefix: str = "  └ "
    style: str = "dim"
    prefix_style: str = ""

    def __rich_console__(self, console, options):
        width = max(1, options.max_width - len(self.prefix))
        rows = [row for line in self.lines for row in line.wrap(console, width)]
        if not rows:
            rows = [Text("（无输出）")]
        if len(rows) > self.max_rows:
            head = max(1, (self.max_rows - 1) // 2)
            tail = self.max_rows - 1 - head
            omitted = len(rows) - head - tail
            notice = Text(f"… 省略 {omitted} 行 (Ctrl+T 展开)")
            notice.truncate(width, overflow="ellipsis")
            rows = rows[:head] + [notice] + (rows[-tail:] if tail else [])
        for index, row in enumerate(rows):
            prefix = self.prefix if index == 0 else " " * len(self.prefix)
            yield Text.assemble((prefix, self.prefix_style), row, style=self.style)


def output_lines(output: str | dict[str, object]) -> list[Text]:
    if isinstance(output, dict):
        output = "\n".join(output[field] for field in ("error", "stdout", "stderr") if output[field])
    return [Text.from_ansi(line) for line in output.splitlines()]
