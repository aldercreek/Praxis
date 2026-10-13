import os
import json
import base64
from bisect import bisect_right
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from praxis_agent.context.budgets import DEFAULT_BUDGETS


_SEARCH_TIMEOUT_SECONDS = 30


def _resolve_workspace(workspace: Path) -> Path:
    """返回经过校验的工作区绝对路径。"""
    workspace_root = workspace.resolve()
    if not workspace_root.is_dir():
        raise NotADirectoryError(f"workspace is not a directory: {workspace}")
    return workspace_root


def _resolve_workspace_path(workspace_root: Path, relative_path: str) -> Path:
    """解析用户传入的路径，并确保它不会越出工作区。"""
    requested_path = Path(relative_path)
    if requested_path.is_absolute():
        raise ValueError("path must be relative to the workspace")

    # 先解析再检查边界，防止通过 `..` 或符号链接越出工作区。
    file_path = (workspace_root / requested_path).resolve()
    if not file_path.is_relative_to(workspace_root):
        raise ValueError("path must stay inside the workspace")
    return file_path


def _require_workspace_file(workspace: Path, relative_path: str) -> Path:
    """返回工作区内一个已经存在的普通文件。"""
    workspace_root = _resolve_workspace(workspace)
    file_path = _resolve_workspace_path(workspace_root, relative_path)

    if not file_path.exists():
        raise FileNotFoundError(f"file does not exist: {relative_path}")
    if not file_path.is_file():
        raise IsADirectoryError(f"path is not a file: {relative_path}")

    return file_path


def _glob_arguments(file_pattern: str, ignore_patterns: list[str] | None) -> list[str]:
    if not isinstance(file_pattern, str) or not file_pattern or file_pattern.startswith("!"):
        raise ValueError("file_pattern must be a non-empty positive glob")
    if ignore_patterns is None:
        ignore_patterns = []
    if not isinstance(ignore_patterns, list) or any(
        not isinstance(pattern, str) or not pattern or pattern.startswith("!")
        for pattern in ignore_patterns
    ):
        raise ValueError("ignore_patterns must be a list of non-empty globs without a leading !")
    arguments = ["--glob", file_pattern]
    for pattern in ignore_patterns:
        arguments.extend(("--glob", "!" + (pattern + "**" if pattern.endswith("/") else pattern)))
    return arguments


def _run_ripgrep(workspace: Path, arguments: list[str]) -> bytes:
    executable = shutil.which("rg")
    if executable is None:
        raise RuntimeError("ripgrep (rg) is required; install it on PATH")
    try:
        result = subprocess.run([executable, "--no-config", *arguments], cwd=workspace,
                                capture_output=True, timeout=_SEARCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("Search timed out; narrow relative_path or file_pattern") from error
    if result.returncode not in (0, 1):
        raise RuntimeError("Search failed: " + result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def read_file(
    workspace: Path,
    relative_path: str,
    start_line: int = 1,
    start_offset: int = 0,
    end_line: int | None = None,
    end_offset: int | None = None,
    *,
    character_limit: int | None = DEFAULT_BUDGETS["max_file_read_chars"],
) -> dict[str, object]:
    """Read a half-open text range; offsets count characters including line endings.

    A None limit is reserved for programmatic calls. A finite limit budgets the
    complete JSON result, so the model preview cannot cut off continuation data.
    """
    file_path = _require_workspace_file(workspace, relative_path)
    with file_path.open(encoding="utf-8", newline="") as source:
        text = source.read()
    lines = text.splitlines(keepends=True) or [""]
    if lines[-1].endswith(("\n", "\r")):
        lines.append("")
    starts = [0]
    for line in lines[:-1]:
        starts.append(starts[-1] + len(line))

    def position(line, offset):
        if isinstance(line, bool) or not isinstance(line, int) or not 1 <= line <= len(lines):
            raise ValueError(f"line must be between 1 and {len(lines)}")
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= len(lines[line - 1]):
            raise ValueError(f"offset for line {line} must be between 0 and {len(lines[line - 1])}")
        return starts[line - 1] + offset

    start = position(start_line, start_offset)
    if end_line is None and end_offset is None:
        end = len(text)
    elif end_line is None:
        raise ValueError("end_line is required when end_offset is supplied")
    elif end_offset is None:
        end = position(end_line, 0) + len(lines[end_line - 1])
    else:
        end = position(end_line, end_offset)
    if end < start:
        raise ValueError("end position must not precede start position")

    def render(stop):
        truncated = stop < end
        index = bisect_right(starts, stop) - 1
        return {
            "text": text[start:stop],
            "truncated": truncated,
            "next_position": {"line": index + 1, "offset": stop - starts[index]} if truncated else None,
            "note": ("Only part of the requested range is shown. Continue from next_position "
                     "with the same end_line/end_offset; omitted end means end of file.") if truncated else "",
        }

    if character_limit is None:
        return render(end)
    if character_limit < 1:
        raise ValueError("character_limit must be positive")
    candidate = render(min(end, start + character_limit))
    if len(json.dumps(candidate, ensure_ascii=False)) <= character_limit:
        return candidate
    low, high = start, min(end - 1, start + character_limit)
    while low < high:
        middle = (low + high + 1) // 2
        if len(json.dumps(render(middle), ensure_ascii=False)) <= character_limit:
            low = middle
        else:
            high = middle - 1
    result = render(low)
    if (low == start and start < end) or len(json.dumps(result, ensure_ascii=False)) > character_limit:
        raise ValueError("Read budget cannot fit metadata and text; increase the output budget")
    return result


def edit_file(
    workspace: Path,
    relative_path: str,
    edits: list[Mapping[str, object]],
) -> dict[str, object]:
    """按顺序应用多个精确替换，并在全部成功后写入文件。"""
    if not isinstance(edits, list) or not edits:
        raise ValueError("edits must be a non-empty list")

    file_path = _require_workspace_file(workspace, relative_path)
    original_content = file_path.read_text(encoding="utf-8")
    updated_content = original_content

    # 先在内存中完成全部校验，避免后续替换失败时留下部分修改。
    for index, edit in enumerate(edits, start=1):
        if not isinstance(edit, Mapping):
            raise TypeError(f"edit {index} must be an object")
        if set(edit) != {"old_text", "new_text"}:
            raise ValueError(
                f"edit {index} must contain only old_text and new_text"
            )

        old_text = edit["old_text"]
        new_text = edit["new_text"]
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise TypeError(
                f"edit {index} old_text and new_text must be strings"
            )
        if not old_text:
            raise ValueError(f"edit {index} old_text must not be empty")
        if old_text == new_text:
            raise ValueError(
                f"edit {index} new_text must differ from old_text"
            )

        match_count = updated_content.count(old_text)
        if match_count == 0:
            raise ValueError(f"edit {index} old_text was not found")
        if match_count > 1:
            raise ValueError(
                f"edit {index} old_text must match exactly once; "
                f"found {match_count} matches"
            )
        updated_content = updated_content.replace(old_text, new_text, 1)

    if updated_content == original_content:
        raise ValueError("edits must produce a file change")

    file_path.write_text(updated_content, encoding="utf-8")
    return {"path": Path(relative_path).as_posix()}


def write_file(workspace: Path, relative_path: str, content: str) -> dict[str, object]:
    """创建或完整重写 UTF-8 文本文件，并返回文件路径。"""
    workspace_root = _resolve_workspace(workspace)
    file_path = _resolve_workspace_path(workspace_root, relative_path)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as new_file:
        new_file.write(content)

    return {"path": Path(relative_path).as_posix()}


def list_files(workspace: Path, relative_path: str = ".") -> dict[str, object]:
    """List all immediate files and directories, including hidden/ignored names."""
    workspace_root = _resolve_workspace(workspace)
    target_path = _resolve_workspace_path(workspace_root, relative_path)
    if not target_path.exists():
        raise FileNotFoundError(f"path does not exist: {relative_path}")
    if not target_path.is_dir():
        raise NotADirectoryError(f"path is not a directory: {relative_path}")

    entries = []
    with os.scandir(target_path) as scanner:
        for entry in scanner:
            if entry.is_dir():
                kind = "directory"
            elif entry.is_file():
                kind = "file"
            else:
                continue
            entries.append({
                "path": (target_path / entry.name).relative_to(workspace_root).as_posix(),
                "type": kind,
            })
    entries.sort(key=lambda entry: (entry["type"] != "directory", entry["path"]))
    return {"entries": entries}


def find_files(
    workspace: Path,
    file_pattern: str,
    relative_path: str = ".",
    ignore_patterns: list[str] | None = None,
) -> dict[str, object]:
    """Find every matching file; only caller-supplied globs exclude paths."""
    globs = _glob_arguments(file_pattern, ignore_patterns)
    workspace_root = _resolve_workspace(workspace)
    target_path = _resolve_workspace_path(workspace_root, relative_path)
    if not target_path.exists():
        raise FileNotFoundError(f"path does not exist: {relative_path}")
    if not target_path.is_dir():
        raise NotADirectoryError(f"path is not a directory: {relative_path}")
    output = _run_ripgrep(target_path, ["--files", "--hidden", "--no-ignore",
                                      "--null", "--path-separator", "/", *globs, "--", "."])
    prefix = target_path.relative_to(workspace_root)
    files = sorted(
        (prefix / name.removeprefix("./")).as_posix()
        for name in output.decode("utf-8").split("\0") if name
    )
    return {"files": files}


def search_text(
    workspace: Path,
    query: str,
    relative_path: str = ".",
    file_pattern: str = "*",
    ignore_patterns: list[str] | None = None,
) -> dict[str, object]:
    """Return every non-overlapping literal match as path/line/character offset."""
    if not isinstance(query, str) or not query or "\n" in query or "\r" in query:
        raise ValueError("query must be a non-empty single-line string")
    globs = _glob_arguments(file_pattern, ignore_patterns)
    workspace_root = _resolve_workspace(workspace)
    target = _resolve_workspace_path(workspace_root, relative_path)
    if not target.exists():
        raise FileNotFoundError(f"path does not exist: {relative_path}")
    if not target.is_dir() and not target.is_file():
        raise ValueError(f"path is not a file or directory: {relative_path}")
    directory = target if target.is_dir() else target.parent
    target_argument = "." if target.is_dir() else target.name
    output = _run_ripgrep(directory, ["--json", "--fixed-strings", "--case-sensitive",
        "--hidden", "--no-ignore", "--text", "--encoding", "utf-8", "--path-separator", "/",
        *globs, "--", query, target_argument])

    def decode(value):
        return value["text"] if "text" in value else base64.b64decode(value["bytes"]).decode("utf-8")

    matches = []
    for record in output.splitlines():
        event = json.loads(record)
        if event["type"] != "match":
            continue
        data = event["data"]
        path = (directory / decode(data["path"])).relative_to(workspace_root).as_posix()
        line = decode(data["lines"]).encode("utf-8")
        previous_byte = 0
        character_offset = 0
        for match in data["submatches"]:
            byte_offset = match["start"]
            character_offset += len(line[previous_byte:byte_offset].decode("utf-8"))
            matches.append({"path": path, "line": data["line_number"], "offset": character_offset})
            previous_byte = byte_offset
    matches.sort(key=lambda match: (match["path"], match["line"], match["offset"]))
    return {"matches": matches}
