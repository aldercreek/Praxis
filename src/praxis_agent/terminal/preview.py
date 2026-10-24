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
            rows = [Text("(no output)")]
        if len(rows) > self.max_rows:
            head = max(1, (self.max_rows - 1) // 2)
            tail = self.max_rows - 1 - head
            omitted = len(rows) - head - tail
            notice = Text(f"… {omitted} lines omitted (Ctrl+T to expand)")
            notice.truncate(width, overflow="ellipsis")
            rows = rows[:head] + [notice] + (rows[-tail:] if tail else [])
        for index, row in enumerate(rows):
            prefix = self.prefix if index == 0 else " " * len(self.prefix)
            yield Text.assemble((prefix, self.prefix_style), row, style=self.style)


def output_lines(output: str | dict[str, object]) -> list[Text]:
    if isinstance(output, dict):
        heading = (f"Running in background · {output['session_id']}\n"
                   if output.get("status") == "running" and output.get("session_id") else "")
        output = heading + "\n".join(output[field] for field in ("error", "stdout", "stderr") if output[field])
    return [Text.from_ansi(line) for line in output.splitlines()]
