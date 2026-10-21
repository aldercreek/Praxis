"""Bounded ripgrep pages. Content offsets count matching lines, not context."""

import base64
import json
import shutil
import subprocess
import tempfile
from collections import deque
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Timer

from praxis_agent.execution import current_handle, current_run


_PREVIEW = " ... [preview; use read_file for full text]"


def _check(scopes):
    for scope in scopes:
        scope.check()


@contextmanager
def _search_output(directory, arguments, timeout):
    executable = shutil.which("rg")
    if executable is None:
        raise RuntimeError("ripgrep (rg) is required; install it on PATH")
    scopes = tuple(scope for scope in (current_run(), current_handle()) if scope is not None)
    _check(scopes)
    expired = Event()
    with tempfile.TemporaryFile() as errors, ExitStack() as subscriptions:
        process = subprocess.Popen(
            [executable, "--no-config", *arguments], cwd=directory,
            stdout=subprocess.PIPE, stderr=errors,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )

        def stop():
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

        def expire():
            expired.set()
            stop()

        timer = Timer(timeout, expire)
        timer.daemon = True
        try:
            for scope in scopes:
                subscriptions.callback(scope.on_cancel(stop))
            timer.start()
            yield process
        finally:
            timer.cancel()
            # Reaching a page boundary intentionally stops the remaining scan.
            stopped_early = process.poll() is None
            stop()
            process.wait()
            process.stdout.close()
            if timer.ident is not None:
                timer.join()
            # A killed writer may leave a partial JSON record. Report the
            # cancellation/timeout rather than the incidental parse error.
            _check(scopes)
            if expired.is_set():
                raise TimeoutError("Search timed out; narrow relative_path or file_pattern")
        # stderr is redirected to a file so a full error pipe cannot deadlock
        # stdout consumption. Diagnose errors even when taking an early page.
        errors.seek(0)
        error = errors.read(16_000).decode("utf-8", errors="replace").strip()
        if error or (not stopped_early and process.returncode not in (0, 1)):
            hint = ("Fix the regular expression or use regex=false." if "regex" in error.lower()
                    else "Check the path, permissions and search filters.")
            raise RuntimeError(f"Search failed: {error or process.returncode}. {hint}")


def _terminated(stream, separator):
    """Read one delimiter-terminated field (paths may contain newlines/colons)."""
    value = bytearray()
    while byte := stream.read(1):
        if byte == separator:
            return bytes(value)
        value.extend(byte)
    return bytes(value) if value else None


def _relative_path(raw, directory, workspace):
    return (directory / raw).relative_to(workspace).as_posix()


def _file_records(stream, mode, directory, workspace):
    while (raw := _terminated(stream, b"\0")) is not None:
        path = _relative_path(raw.decode("utf-8"), directory, workspace)
        if mode == "files":
            yield path
        else:
            yield {"path": path, "count": int(stream.readline())}


def _decode(value):
    return value["text"] if "text" in value else base64.b64decode(value["bytes"]).decode("utf-8", errors="replace")


@dataclass(frozen=True)
class _Line:
    path: str
    number: int
    text: str
    anchor: int = 0


@dataclass(frozen=True)
class _Snippet:
    match: _Line
    lines: tuple[_Line, ...]


def _content_records(stream, directory, workspace, context, character_limit):
    window, pending = deque(), deque()

    def snippet(match):
        return _Snippet(match, tuple(line for line in window
                                    if abs(line.number - match.number) <= context))

    for raw in stream:
        event = json.loads(raw)
        kind, data = event["type"], event.get("data")
        if kind == "end":
            while pending:
                yield snippet(pending.popleft())
            window.clear()
            continue
        if kind not in ("match", "context"):
            continue
        text = _decode(data["lines"]).removesuffix("\n").removesuffix("\r")
        anchor = 0
        if kind == "match" and data["submatches"]:
            anchor = len(text.encode("utf-8")[:data["submatches"][0]["start"]].decode("utf-8", errors="ignore"))
        if len(text) > character_limit:
            # Prioritize the actual match even when the budget is very small.
            prefix = "... " if anchor else ""
            text = prefix + text[anchor:anchor + character_limit] + _PREVIEW
            anchor = len(prefix)
        line = _Line(_relative_path(_decode(data["path"]), directory, workspace), data["line_number"], text, anchor)
        window.append(line)
        if kind == "match":
            pending.append(line)
        while pending and line.number >= pending[0].number + context:
            yield snippet(pending.popleft())
        # Retain context for pending matches and the next possible match only.
        cutoff = pending[0].number - context if pending else line.number - context + 1
        while window and window[0].number < cutoff:
            window.popleft()
    while pending:
        yield snippet(pending.popleft())


def _render_content(snippets, context):
    selected = {(item.match.path, item.match.number) for item in snippets}
    lines = {}
    for item in snippets:
        for line in item.lines:
            lines[line.path, line.number] = line
    results = []
    previous = None
    for key, line in lines.items():
        if context and previous is not None and (key[0] != previous[0] or key[1] != previous[1] + 1):
            results.append("--")
        separator = ":" if key in selected else "-"
        results.append(f"{line.path}{separator}{line.number}{separator}{line.text}")
        previous = key
    return results


def _page(records, *, mode, context, offset, limit, character_limit):
    items = []
    render = (lambda values: _render_content(values, context)) if mode == "content" else list

    def fits(values):
        results = render(values)
        # Reserve enough space for either the continuation or EOF metadata.
        return max(len(json.dumps({"results": results, "next_offset": next_offset}, ensure_ascii=False))
                   for next_offset in (None, offset + len(values))) <= character_limit

    if not fits([]):
        raise ValueError("Search budget cannot fit metadata; increase the output budget")
    for index, item in enumerate(records):
        if index < offset:
            continue
        if len(items) == limit or (items and not fits([*items, item])):
            return {"results": render(items), "next_offset": offset + len(items)}
        if not fits([item]):
            if mode != "content":
                raise ValueError("Search budget cannot fit a file record; increase the output budget")
            # A single oversized match must still advance the page. Drop its
            # context and clearly label the bounded matching-line preview.
            text = item.match.text[item.match.anchor:].removesuffix(_PREVIEW)
            prefix = "... " if item.match.anchor else ""

            def preview(length):
                line = replace(item.match, text=prefix + text[:length] + _PREVIEW)
                return _Snippet(line, (line,))

            if not fits([preview(1)]):
                raise ValueError("Search budget cannot fit a matching-line preview; increase the output budget")
            low, high = 1, len(text)
            while low < high:
                middle = (low + high + 1) // 2
                if fits([preview(middle)]):
                    low = middle
                else:
                    high = middle - 1
            item = preview(low)
        items.append(item)
    return {"results": render(items), "next_offset": None}


def search_page(workspace: Path, directory: Path, target: str, query: str, globs: list[str], *,
                mode: str, regex: bool, case_sensitive: bool, context_lines: int,
                limit: int, offset: int, character_limit: int, timeout: float):
    arguments = ["--sort", "path", "--with-filename", "--hidden", "--no-ignore", "--text", "--encoding", "utf-8",
                 "--path-separator", "/", "--case-sensitive" if case_sensitive else "--ignore-case"]
    if not regex:
        arguments.append("--fixed-strings")
    if mode == "content":
        arguments.extend(["--json", "--context", str(context_lines)])
    else:
        arguments.extend(["--files-with-matches" if mode == "files" else "--count", "--null"])
    arguments.extend([*globs, "--", query, target])
    with _search_output(directory, arguments, timeout) as process:
        stream = process.stdout
        records = (_content_records(stream, directory, workspace, context_lines, character_limit)
                   if mode == "content" else _file_records(stream, mode, directory, workspace))
        page = _page(records, mode=mode, context=context_lines, offset=offset,
                     limit=limit, character_limit=character_limit)
        if page["next_offset"] is None:
            # EOF is not itself proof of a successful process exit.
            process.wait()
        return page
