"""Bounded code-output previews with complete overflow logs on disk."""

from pathlib import Path
from typing import TextIO
from uuid import uuid4


_HEAD_CHARS = 6_000
_TAIL_CHARS = 24_000
_PREVIEW_CHARS = _HEAD_CHARS + _TAIL_CHARS


class _Stream:
    def __init__(self, path: Path):
        self.path = path
        self.head = ""
        self.tail = ""
        self.total_chars = 0
        self.error = ""
        self._file: TextIO | None = None
        self._created = False

    def append(self, text: str) -> None:
        if not text:
            return
        previous = self.head + self.tail if self.total_chars <= _PREVIEW_CHARS else ""
        self.total_chars += len(text)
        head_remaining = _HEAD_CHARS - len(self.head)
        self.head += text[:head_remaining]
        if len(text) - head_remaining >= _TAIL_CHARS:
            self.tail = text[-_TAIL_CHARS:]
        else:
            self.tail = (self.tail + text[head_remaining:])[-_TAIL_CHARS:]
        try:
            if self.error:
                raise OSError(self.error)
            if not self._created and self.total_chars > _PREVIEW_CHARS:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = self.path.open("x", encoding="utf-8", newline="")
                self._created = True
                self._file.write(previous)
            if self._file is not None:
                # Bound the temporary UTF-8 encoding buffer even for one huge print.
                for offset in range(0, len(text), 65_536):
                    self._file.write(text[offset:offset + 65_536])
        except (OSError, UnicodeError) as error:
            self.error = f"{type(error).__name__}: {error}"
            raise

    def close(self) -> None:
        stream, self._file = self._file, None
        if stream is not None:
            try:
                stream.close()
            except (OSError, UnicodeError) as error:
                self.error = f"{type(error).__name__}: {error}"

    def preview(self) -> str:
        omitted = self.total_chars - len(self.head) - len(self.tail)
        if omitted > 0:
            return self.head + f"\n... [{omitted} characters omitted] ...\n" + self.tail
        return self.head + self.tail

    def reference(self) -> dict | None:
        if not self._created:
            return None
        return {
            "path": str(self.path), "total_chars": self.total_chars,
            "omitted_chars": max(0, self.total_chars - _PREVIEW_CHARS),
            "complete": not self.error,
        }


class CodeOutput:
    def __init__(self, directory: Path):
        call_id = uuid4().hex
        self.streams = {name: _Stream(directory / f"{call_id}-{name}.log")
                        for name in ("stdout", "stderr")}

    def append(self, stream: str, text: str) -> None:
        self.streams[stream].append(text)

    def close(self) -> None:
        for stream in self.streams.values():
            stream.close()

    @property
    def errors(self) -> dict[str, str]:
        return {name: stream.error for name, stream in self.streams.items() if stream.error}

    def fields(self) -> dict:
        return {
            **{name: stream.preview() for name, stream in self.streams.items()},
            "output_files": {name: ref for name, stream in self.streams.items()
                             if (ref := stream.reference()) is not None},
            **({"output_errors": self.errors} if self.errors else {}),
        }
