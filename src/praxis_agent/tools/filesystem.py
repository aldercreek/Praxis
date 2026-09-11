import shutil
import subprocess
from collections.abc import Iterator, Mapping
from difflib import unified_diff
from pathlib import Path
from threading import Event, Thread, Timer


# 递归扫描时跳过依赖目录、缓存目录和构建产物目录。
_IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".idea",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
    }
)
_IGNORED_DIRECTORY_SUFFIXES = (".egg-info",)
_MAX_SEARCH_FILE_SIZE = 1_000_000
_SEARCH_TIMEOUT_SECONDS = 30
_MAX_READ_LINES = 400
_MAX_READ_CHARACTERS = 20_000


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


def _is_ignored_directory(name: str) -> bool:
    """判断目录是否属于依赖、缓存或构建产物。"""
    return name in _IGNORED_DIRECTORY_NAMES or name.endswith(
        _IGNORED_DIRECTORY_SUFFIXES
    )


def _iter_workspace_entries(
    workspace_root: Path,
    target_path: Path,
    max_depth: int,
) -> Iterator[tuple[Path, bool]]:
    """按路径顺序遍历有限深度内的文件和目录。"""
    if target_path.is_file():
        yield target_path, False
        return

    def visit(directory: Path, depth: int) -> Iterator[tuple[Path, bool]]:
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError:
            return

        for child in children:
            try:
                resolved_child = child.resolve()
                is_directory = child.is_dir()
                is_file = child.is_file()
            except OSError:
                continue
            if not resolved_child.is_relative_to(workspace_root):
                continue
            if is_directory and _is_ignored_directory(child.name):
                continue
            if not is_directory and not is_file:
                continue

            yield child, is_directory
            if (
                is_directory
                and depth < max_depth
                and not child.is_symlink()
            ):
                yield from visit(child, depth + 1)

    yield from visit(target_path, 1)


def _normalize_search_match(match: str) -> str:
    """统一 ripgrep 返回的路径分隔符和相对路径前缀。"""
    parts = match.split(":", 2)
    if len(parts) != 3:
        return match
    display_path, line_number, content = parts
    display_path = display_path.replace("\\", "/").removeprefix("./")
    return f"{display_path}:{line_number}:{content}"


def _ripgrep_glob_arguments(file_pattern: str) -> list[str]:
    """构造文件模式和忽略目录对应的 ripgrep 参数。"""
    arguments: list[str] = []
    if file_pattern != "*":
        arguments.extend(("--glob", file_pattern))
    for directory_name in sorted(_IGNORED_DIRECTORY_NAMES):
        arguments.extend(("--glob", f"!**/{directory_name}/**"))
    for directory_suffix in _IGNORED_DIRECTORY_SUFFIXES:
        arguments.extend(("--glob", f"!**/*{directory_suffix}/**"))
    return arguments


def _run_ripgrep(
    workspace_root: Path,
    arguments: list[str],
    max_results: int,
    operation: str,
) -> tuple[list[str], bool]:
    """运行必需的 ripgrep，并限制执行时间和输出行数。"""
    executable = shutil.which("rg")
    if executable is None:
        raise RuntimeError(
            "ripgrep (rg) is required for find_files and search_text; "
            "install it and ensure rg is available on PATH"
        )

    try:
        process = subprocess.Popen(
            [executable, *arguments],
            cwd=workspace_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        raise RuntimeError(f"failed to start ripgrep: {error}") from error

    output_lines: list[str] = []
    truncated = False
    timed_out = Event()
    stderr_parts: list[str] = []
    assert process.stdout is not None
    assert process.stderr is not None

    def read_stderr() -> None:
        stderr_parts.append(process.stderr.read())

    def kill_after_timeout() -> None:
        if process.poll() is not None:
            return
        timed_out.set()
        try:
            process.kill()
        except OSError:
            pass

    stderr_reader = Thread(target=read_stderr, daemon=True)
    timeout_timer = Timer(_SEARCH_TIMEOUT_SECONDS, kill_after_timeout)
    timeout_timer.daemon = True
    stderr_reader.start()
    timeout_timer.start()

    # 多读取一行，用来区分“刚好达到上限”和“确实被截断”。
    try:
        for output_line in process.stdout:
            output_lines.append(
                output_line.removesuffix("\n").removesuffix("\r")
            )
            if len(output_lines) > max_results:
                truncated = True
                break
    finally:
        timeout_timer.cancel()
        timeout_timer.join(timeout=1)
        if truncated and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        stderr_reader.join(timeout=2)
        process.stdout.close()
        process.stderr.close()

    if timed_out.is_set():
        raise TimeoutError(
            f"ripgrep {operation} timed out after "
            f"{_SEARCH_TIMEOUT_SECONDS} seconds; narrow relative_path "
            "or file_pattern"
        )

    if not truncated and process.returncode not in (0, 1):
        stderr = "".join(stderr_parts)
        detail = stderr.strip() or f"exit code {process.returncode}"
        raise RuntimeError(f"ripgrep {operation} failed: {detail}")
    return output_lines[:max_results], truncated


def _search_with_ripgrep(
    workspace_root: Path,
    target_path: Path,
    query: str,
    file_pattern: str,
    max_results: int,
) -> tuple[list[str], bool]:
    """使用 ripgrep 搜索文件内容。"""
    target_argument = target_path.relative_to(workspace_root).as_posix() or "."
    arguments = [
        "--fixed-strings",
        "--line-number",
        "--no-heading",
        "--with-filename",
        "--hidden",
        "--color",
        "never",
        "--sort",
        "path",
        "--path-separator",
        "/",
        "--max-filesize",
        str(_MAX_SEARCH_FILE_SIZE),
        "--no-messages",
    ]
    arguments.extend(_ripgrep_glob_arguments(file_pattern))
    arguments.extend(("--", query, target_argument))

    result = _run_ripgrep(
        workspace_root,
        arguments,
        max_results,
        "search",
    )
    matches, truncated = result
    return [_normalize_search_match(match) for match in matches], truncated


def _find_files_with_ripgrep(
    workspace_root: Path,
    target_path: Path,
    file_pattern: str,
    max_results: int,
) -> tuple[list[str], bool]:
    """使用 ripgrep 按 glob 模式查找文件。"""
    target_argument = target_path.relative_to(workspace_root).as_posix() or "."
    arguments = [
        "--files",
        "--hidden",
        "--sort",
        "path",
        "--path-separator",
        "/",
        "--no-messages",
    ]
    arguments.extend(_ripgrep_glob_arguments(file_pattern))
    arguments.extend(("--", target_argument))

    result = _run_ripgrep(
        workspace_root,
        arguments,
        max_results,
        "file discovery",
    )
    matches, truncated = result
    normalized_matches = [
        match.replace("\\", "/").removeprefix("./")
        for match in matches
    ]
    normalized_matches.sort()
    return normalized_matches, truncated


def _build_unified_diff(
    relative_path: str,
    original_content: str,
    updated_content: str,
    *,
    created: bool = False,
) -> str:
    """为修改或新建的文件生成 Git 风格的差异文本。"""
    display_path = Path(relative_path).as_posix()
    before_path = "/dev/null" if created else f"a/{display_path}"
    diff = "".join(
        unified_diff(
            original_content.splitlines(keepends=True),
            updated_content.splitlines(keepends=True),
            fromfile=before_path,
            tofile=f"b/{display_path}",
        )
    )
    # 新建空文件时 difflib 不会生成变更块，因此至少保留文件头信息。
    return diff or f"--- {before_path}\n+++ b/{display_path}\n"


def read_file(
    workspace: Path,
    relative_path: str,
    start_line: int = 1,
    max_lines: int = 200,
) -> str:
    """按行读取工作区内的 UTF-8 文本文件。"""
    if start_line < 1:
        raise ValueError("start_line must be at least 1")
    if max_lines < 1:
        raise ValueError("max_lines must be at least 1")
    if max_lines > _MAX_READ_LINES:
        raise ValueError(f"max_lines must not exceed {_MAX_READ_LINES}")

    file_path = _require_workspace_file(workspace, relative_path)
    lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    total_lines = len(lines)
    if total_lines == 0:
        return "[lines 0-0 of 0]"
    if start_line > total_lines:
        raise ValueError(
            f"start_line {start_line} is past the end of the file "
            f"({total_lines} lines)"
        )

    selected_lines = lines[start_line - 1 : start_line - 1 + max_lines]
    visible_lines: list[str] = []
    visible_characters = 0
    last_line_truncated = False

    # 保证一次工具调用不会把过多文件内容送回模型上下文。
    for line in selected_lines:
        remaining_characters = _MAX_READ_CHARACTERS - visible_characters
        if len(line) <= remaining_characters:
            visible_lines.append(line)
            visible_characters += len(line)
            continue
        if not visible_lines:
            visible_lines.append(line[:remaining_characters])
            last_line_truncated = True
        break

    end_line = start_line + len(visible_lines) - 1
    header_parts = [f"lines {start_line}-{end_line} of {total_lines}"]
    if end_line < total_lines:
        header_parts.append(f"next_start_line={end_line + 1}")
    if last_line_truncated:
        header_parts.append(
            f"line {end_line} truncated at {_MAX_READ_CHARACTERS} characters"
        )

    header = f"[{'; '.join(header_parts)}]"
    return f"{header}\n{''.join(visible_lines)}"


def edit_file(
    workspace: Path,
    relative_path: str,
    edits: list[Mapping[str, object]],
) -> str:
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

    diff = _build_unified_diff(relative_path, original_content, updated_content)
    file_path.write_text(updated_content, encoding="utf-8")
    return diff


def create_file(workspace: Path, relative_path: str, content: str) -> str:
    """创建新的 UTF-8 文本文件，并返回统一差异格式。"""
    workspace_root = _resolve_workspace(workspace)
    file_path = _resolve_workspace_path(workspace_root, relative_path)
    if file_path.exists():
        raise FileExistsError(f"file already exists: {relative_path}")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    # 独占创建模式可以防止在存在性检查后误覆盖其他进程刚创建的文件。
    with file_path.open("x", encoding="utf-8") as new_file:
        new_file.write(content)

    return _build_unified_diff(relative_path, "", content, created=True)


def list_files(
    workspace: Path,
    relative_path: str = ".",
    max_entries: int = 200,
    max_depth: int = 2,
) -> str:
    """列出工作区内有限深度的文件和目录。"""
    if max_entries < 1:
        raise ValueError("max_entries must be at least 1")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")

    workspace_root = _resolve_workspace(workspace)
    target_path = _resolve_workspace_path(workspace_root, relative_path)
    if not target_path.exists():
        raise FileNotFoundError(f"path does not exist: {relative_path}")
    if not target_path.is_file() and not target_path.is_dir():
        raise ValueError(f"path is not a file or directory: {relative_path}")

    entries: list[str] = []
    for entry_path, is_directory in _iter_workspace_entries(
        workspace_root,
        target_path,
        max_depth,
    ):
        if len(entries) == max_entries:
            entries.append(f"... truncated after {max_entries} entries")
            break
        display_path = entry_path.relative_to(workspace_root).as_posix()
        entries.append(f"{display_path}/" if is_directory else display_path)

    return "\n".join(entries) if entries else "No entries found."


def find_files(
    workspace: Path,
    file_pattern: str,
    relative_path: str = ".",
    max_results: int = 100,
) -> str:
    """按 glob 模式查找工作区内的文件路径。"""
    if not file_pattern:
        raise ValueError("file_pattern must not be empty")
    if max_results < 1:
        raise ValueError("max_results must be at least 1")

    workspace_root = _resolve_workspace(workspace)
    target_path = _resolve_workspace_path(workspace_root, relative_path)
    if not target_path.exists():
        raise FileNotFoundError(f"path does not exist: {relative_path}")
    if not target_path.is_file() and not target_path.is_dir():
        raise ValueError(f"path is not a file or directory: {relative_path}")

    matches, truncated = _find_files_with_ripgrep(
        workspace_root,
        target_path,
        file_pattern,
        max_results,
    )

    if not matches:
        return f"No files found matching {file_pattern!r}."
    if truncated:
        matches.append(f"... truncated after {max_results} files")
    return "\n".join(matches)


def search_text(
    workspace: Path,
    query: str,
    relative_path: str = ".",
    max_results: int = 50,
    file_pattern: str = "*",
) -> str:
    """按文件模式区分大小写地搜索固定字符串。"""
    if not query:
        raise ValueError("query must not be empty")
    if max_results < 1:
        raise ValueError("max_results must be at least 1")
    if not file_pattern:
        raise ValueError("file_pattern must not be empty")

    workspace_root = _resolve_workspace(workspace)
    target_path = _resolve_workspace_path(workspace_root, relative_path)
    if not target_path.exists():
        raise FileNotFoundError(f"path does not exist: {relative_path}")
    if not target_path.is_file() and not target_path.is_dir():
        raise ValueError(f"path is not a file or directory: {relative_path}")

    matches, truncated = _search_with_ripgrep(
        workspace_root,
        target_path,
        query,
        file_pattern,
        max_results,
    )

    if not matches:
        return f"No matches found for {query!r}."
    if truncated:
        matches.append(f"... truncated after {max_results} matches")
    return "\n".join(matches)
