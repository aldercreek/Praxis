"""Read text with its format, and reuse that format when saving edits or rewrites."""

import codecs
from dataclasses import dataclass
from pathlib import Path

from charset_normalizer import from_bytes


# UTF-32 LE starts with the UTF-16 LE BOM, so check the longer signatures first.
_BOM_ENCODINGS = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n")


@dataclass(frozen=True)
class TextFile:
    content: str = ""
    encoding: str = "utf-8"
    newline: str = "\n"
    bom: bytes = b""

    @classmethod
    def read(cls, path: Path) -> "TextFile":
        data = path.read_bytes()
        bom, encoding = next(
            ((bom, encoding) for bom, encoding in _BOM_ENCODINGS if data.startswith(bom)),
            (b"", "utf-8"),
        )
        if bom:
            content = data[len(bom):].decode(encoding)
        else:
            try:
                # ASCII is also UTF-8. Checking the full file avoids mistaking a
                # UTF-8 character split at the sample boundary for another encoding.
                content = data.decode("utf-8")
            except UnicodeDecodeError:
                detected = from_bytes(data[:4096]).best()
                encoding = detected.encoding if detected is not None else "utf-8"
                if encoding == "ascii":
                    encoding = "utf-8"
                content = data.decode(encoding)

        newline = "\r\n" if "\r\n" in content and "\n" not in content.replace("\r\n", "") else "\n"
        return cls(content, encoding, newline, bom)

    def write(self, path: Path, content: str) -> str:
        rendered = normalize_newlines(content).replace("\n", self.newline)
        # Finish encoding before opening the file, so encoding errors cannot
        # truncate the original. Explicit bytes also avoid platform newline conversion.
        data = self.bom + rendered.encode(self.encoding)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return rendered
