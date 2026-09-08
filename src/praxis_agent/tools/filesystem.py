import os
from collections.abc import Iterator, Mapping
from difflib import unified_diff
from pathlib import Path


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
_MAX_SEARCH_FILE_SIZE = 1_000_000


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
            if name not in _IGNORED_DIRECTORY_NAMES
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


def read_file(workspace: Path, relative_path: str) -> str:
    """读取工作区内的 UTF-8 文本文件。"""
    file_path = _require_workspace_file(workspace, relative_path)
    return file_path.read_text(encoding="utf-8")


def replace_text(
    workspace: Path,
    relative_path: str,
    old_text: str,
    new_text: str,
) -> str:
    """精确替换唯一一处文本，并返回统一差异格式。"""
    if not old_text:
        raise ValueError("old_text must not be empty")
    if old_text == new_text:
        raise ValueError("new_text must differ from old_text")

    file_path = _require_workspace_file(workspace, relative_path)
    original_content = file_path.read_text(encoding="utf-8")
    match_count = original_content.count(old_text)

    # 只允许唯一匹配，避免模型生成的修改产生歧义或误改多处。
    if match_count == 0:
        raise ValueError("old_text was not found in the file")
    if match_count > 1:
        raise ValueError(
            f"old_text must match exactly once; found {match_count} matches"
        )

    updated_content = original_content.replace(old_text, new_text, 1)
    diff = _build_unified_diff(relative_path, original_content, updated_content)

    file_path.write_text(updated_content, encoding="utf-8")
    return diff


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
) -> str:
    """递归列出工作区文件，并使用相对于工作区的路径。"""
    if max_entries < 1:
        raise ValueError("max_entries must be at least 1")

    workspace_root = _resolve_workspace(workspace)
    target_path = _resolve_workspace_path(workspace_root, relative_path)
    if not target_path.exists():
        raise FileNotFoundError(f"path does not exist: {relative_path}")
    if not target_path.is_file() and not target_path.is_dir():
        raise ValueError(f"path is not a file or directory: {relative_path}")

    entries: list[str] = []
    for file_path in _iter_workspace_files(workspace_root, target_path):
        if len(entries) == max_entries:
            entries.append(f"... truncated after {max_entries} files")
            break
        entries.append(file_path.relative_to(workspace_root).as_posix())

    return "\n".join(entries) if entries else "No files found."


def search_text(
    workspace: Path,
    query: str,
    relative_path: str = ".",
    max_results: int = 50,
) -> str:
    """在 UTF-8 文本文件中区分大小写地搜索固定字符串。"""
    if not query:
        raise ValueError("query must not be empty")
    if max_results < 1:
        raise ValueError("max_results must be at least 1")

    workspace_root = _resolve_workspace(workspace)
    target_path = _resolve_workspace_path(workspace_root, relative_path)
    if not target_path.exists():
        raise FileNotFoundError(f"path does not exist: {relative_path}")
    if not target_path.is_file() and not target_path.is_dir():
        raise ValueError(f"path is not a file or directory: {relative_path}")

    matches: list[str] = []
    for file_path in _iter_workspace_files(workspace_root, target_path):
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
            matches.append(f"{display_path}:{line_number}:{line.rstrip()}")
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
