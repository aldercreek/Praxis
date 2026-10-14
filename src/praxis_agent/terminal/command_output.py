"""Bounded command previews shared with Rich's refresh thread."""

from dataclasses import dataclass, field
from threading import Lock
from time import monotonic

from rich.console import Group
from rich.text import Text

from praxis_agent.events import ToolOutputDelta


@dataclass
class _Output:
    command: str
    agent_id: str
    streams: dict[str, str] = field(default_factory=dict)
    updated_at: float = 0
    finished_at: float | None = None


class CommandOutputPreview:
    """Workers only update buffers; the UI refresh thread renders them."""

    def __init__(self):
        self._lock = Lock()
        self._outputs: dict[tuple[str, str], _Output] = {}

    def update(self, event: ToolOutputDelta) -> None:
        key = (event.agent_id, event.call_id)
        with self._lock:
            if event.finished:
                if not event.agent_id:
                    self._outputs.pop(key, None)
                elif key in self._outputs:
                    self._outputs[key].finished_at = event.occurred_at
                return
            if not event.text:
                return
            output = self._outputs.setdefault(key, _Output(event.command, event.agent_id))
            output.streams[event.stream] = (output.streams.get(event.stream, "") + event.text)[-8192:]
            output.updated_at = event.occurred_at
            # Keep a bounded number of previews even if a worker is interrupted.
            while len(self._outputs) > 8:
                self._outputs.pop(next(iter(self._outputs)))

    def clear(self) -> None:
        with self._lock:
            self._outputs.clear()

    def render(self) -> list[Group]:
        now = monotonic()
        blocks = []
        with self._lock:
            for key, output in list(self._outputs.items()):
                if output.finished_at is not None and now - output.finished_at > 5:
                    del self._outputs[key]
                    continue
                title = f"子 Agent {output.agent_id[:14]} · {output.command[:80]}" if output.agent_id else "命令输出"
                age = max(0, int(now - output.updated_at))
                title += " · 已结束" if output.finished_at is not None else f" · 最近输出 {age}s 前"
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
