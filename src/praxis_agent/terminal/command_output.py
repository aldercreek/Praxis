"""Bounded raw buffers; live and completed commands share the same preview renderer."""

from dataclasses import dataclass, field

from praxis_agent.events import ToolOutputDelta


@dataclass
class _Stream:
    text: str = ""
    truncated: bool = False

    def append(self, text: str) -> None:
        self.text += text
        if len(self.text) > 8192:
            self.text = self.text[:4096] + self.text[-4096:]
            self.truncated = True


@dataclass
class CommandOutput:
    streams: dict[str, _Stream] = field(default_factory=dict)

    def text(self) -> str:
        streams = (
            stream.text[:4096] + "\n… 输出缓冲已截断\n" + stream.text[4096:]
            if stream.truncated else stream.text
            for stream in self.streams.values()
        )
        return "\n".join(line for stream in streams for line in stream.splitlines())


class CommandOutputs:
    def __init__(self):
        self.outputs: dict[str, CommandOutput] = {}

    def update(self, event: ToolOutputDelta) -> None:
        if event.finished:
            self.outputs.pop(event.call_id, None)
        elif event.text:
            output = self.outputs.setdefault(event.call_id, CommandOutput())
            output.streams.setdefault(event.stream, _Stream()).append(event.text)
            while len(self.outputs) > 8:
                self.outputs.pop(next(iter(self.outputs)))

    def clear(self) -> None:
        self.outputs.clear()
