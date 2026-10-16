"""Bounded command previews owned by one agent page on the UI thread."""

from dataclasses import dataclass, field
from time import monotonic

from rich.console import Group
from rich.text import Text

from praxis_agent.events import ToolOutputDelta


@dataclass
class _Output:
    streams: dict[str, str] = field(default_factory=dict)
    updated_at: float = 0


class CommandOutputPreview:
    """The ViewEventSink queue serializes updates before they reach this state."""

    def __init__(self):
        self._outputs: dict[str, _Output] = {}

    def update(self, event: ToolOutputDelta) -> None:
        key = event.call_id
        if event.finished:
            self._outputs.pop(key, None)
            return
        if not event.text:
            return
        output = self._outputs.setdefault(key, _Output())
        output.streams[event.stream] = (output.streams.get(event.stream, "") + event.text)[-8192:]
        output.updated_at = event.occurred_at
        # Keep a bounded number of previews even if a worker is interrupted.
        while len(self._outputs) > 8:
            self._outputs.pop(next(iter(self._outputs)))

    def clear(self) -> None:
        self._outputs.clear()

    def render(self) -> list[Group]:
        now = monotonic()
        blocks = []
        for output in self._outputs.values():
            title = "命令输出"
            age = max(0, int(now - output.updated_at))
            title += f" · 最近输出 {age}s 前"
            lines = [Text(title, style="dim")]
            nonempty = {name: text for name, text in output.streams.items() if text}
            budget = 3 if len(nonempty) > 1 else 6
            for name, text in nonempty.items():
                for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines()[-budget:]:
                    # Bound both line count and width, including no-newline logs.
                    visible = Text.from_ansi(line)
                    if len(visible) > 240:
                        visible = visible[:239] + Text("…")
                    if name == "stderr":
                        visible.stylize("red")
                    lines.append(Text("  ") + visible)
            blocks.append(Group(*lines))
        return blocks
