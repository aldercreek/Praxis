import os
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from difflib import unified_diff
from pathlib import Path, PurePosixPath


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


def _iter_workspace_files(
    workspace_root: Path,
    target_path: Path,
) -> Iterator[Path]:
    """遍历工作区文件，同时跳过生成目录。"""
    if target_path.is_file():
        yield target_path
        return

    for current_directory, directory_names, file_names in os.walk(target_path):
        # os.walk 会使用这个列表决定接下来进入哪些目录，因此需要原地修改。
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not _is_ignored_directory(name)
        )
        for file_name in sorted(file_names):
            candidate = Path(current_directory) / file_name
            try:
                resolved_candidate = candidate.resolve()
            except OSError:
                continue
            # 目录树中的符号链接文件仍可能指向工作区外部。
            if (
                resolved_candidate.is_relative_to(workspace_root)
                and candidate.is_file()
            ):
                yield candidate


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


def _matches_file_pattern(
    file_path: Path,
    workspace_root: Path,
    pattern: str,
) -> bool:
    """判断文件的工作区相对路径是否符合 glob 模式。"""
    display_path = file_path.relative_to(workspace_root).as_posix()
    return pattern == "*" or PurePosixPath(display_path).match(pattern)


def _normalize_search_match(match: str) -> str:
    """统一 ripgrep 返回的路径分隔符和相对路径前缀。"""
    parts = match.split(":", 2)
    if len(parts) != 3:
        return match
    display_path, line_number, content = parts
    display_path = display_path.replace("\\", "/").removeprefix("./")
    return f"{display_path}:{line_number}:{content}"


def _search_with_ripgrep(
    workspace_root: Path,
    target_path: Path,
    query: str,
    file_pattern: str,
    max_results: int,
) -> tuple[list[str], bool] | None:
    """使用 ripgrep 搜索；程序不可用时返回 None。"""
    executable = shutil.which("rg")
    if executable is None:
        return None

    target_argument = target_path.relative_to(workspace_root).as_posix() or "."
    command = [
        executable,
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
    if file_pattern != "*":
        command.extend(("--glob", file_pattern))
    for directory_name in sorted(_IGNORED_DIRECTORY_NAMES):
        command.extend(("--glob", f"!**/{directory_name}/**"))
    for directory_suffix in _IGNORED_DIRECTORY_SUFFIXES:
        command.extend(("--glob", f"!**/*{directory_suffix}/**"))
    command.extend(("--", query, target_argument))

    try:
        process = subprocess.Popen(
            command,
            cwd=workspace_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None

    matches: list[str] = []
    truncated = False
    assert process.stdout is not None

    # 只读取 max_results 之外的一个结果，用来判断输出是否真的被截断。
    for output_line in process.stdout:
        normalized_match = _normalize_search_match(
            output_line.removesuffix("\n").removesuffix("\r")
        )
        matches.append(normalized_match)
        if len(matches) > max_results:
            truncated = True
            break

    if truncated and process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        _, stderr = process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        _, stderr = process.communicate()

    if not truncated and process.returncode not in (0, 1):
        detail = stderr.strip() or f"exit code {process.returncode}"
        raise RuntimeError(f"ripgrep search failed: {detail}")
    return matches[:max_results], truncated


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

    ripgrep_result = _search_with_ripgrep(
        workspace_root,
        target_path,
        query,
        file_pattern,
        max_results,
    )
    if ripgrep_result is not None:
        matches, truncated = ripgrep_result
        if not matches:
            return f"No matches found for {query!r}."
        if truncated:
            matches.append(f"... truncated after {max_results} matches")
        return "\n".join(matches)

    # 没有安装 ripgrep 时，退回到 Python 文件扫描。
    matches: list[str] = []
    for file_path in _iter_workspace_files(workspace_root, target_path):
        if not _matches_file_pattern(
            file_path,
            workspace_root,
            file_pattern,
        ):
            continue
        try:
            # 限制单个文件大小，并且只搜索能够按 UTF-8 解码的文本文件。
            if file_path.stat().st_size > _MAX_SEARCH_FILE_SIZE:
                continue
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        display_path = file_path.relative_to(workspace_root).as_posix()
        for line_number, line in enumerate(content.splitlines(), start=1):
            if query not in line:
                continue
            matches.append(f"{display_path}:{line_number}:{line}")
            # 多读取一个匹配项，确保只有确实被截断时才显示截断提示。
            if len(matches) > max_results:
                visible_matches = matches[:max_results]
                visible_matches.append(
                    f"... truncated after {max_results} matches"
                )
                return "\n".join(visible_matches)

    if not matches:
        return f"No matches found for {query!r}."
    return "\n".join(matches)
