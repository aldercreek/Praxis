from __future__ import annotations

import json

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import partial, wraps
from inspect import Parameter, signature
from difflib import get_close_matches
from pathlib import Path
from typing import TYPE_CHECKING

from praxis_agent.tools.command import run_command
from praxis_agent.tools.filesystem import (
    create_file,
    edit_file,
    find_files,
    list_files,
    read_file,
    search_text,
)

if TYPE_CHECKING:
    from praxis_agent.context.capacity import ContextPolicy
    from praxis_agent.context.manager import ContextManager
    from praxis_agent.context.models import ConversationHistory
    from praxis_agent.planning.state import RunState


ToolHandler = Callable[..., "str | ToolResult"]


# parameters 使用 JSON Schema 约束模型生成的工具参数。
_READ_FILE_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "relative_path": {"type": "string", "description": "Relative workspace path or absolute file path."},
        "start_line": {"type": "integer", "minimum": 1, "description": "One-based starting line, inclusive; defaults to 1."},
        "start_offset": {"type": "integer", "minimum": 0, "description": "Zero-based character offset within the starting line, including newline characters; defaults to 0."},
        "end_line": {"type": ["integer", "null"], "minimum": 1, "description": "One-based ending line. Null with end_offset=null reads to EOF."},
        "end_offset": {"type": ["integer", "null"], "minimum": 0, "description": "Exclusive character offset in the ending line. Supply both end fields or neither."},
    },
    "required": ["relative_path"],
    "additionalProperties": False,
}

_EDIT_FILE_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "relative_path": {
            "type": "string",
            "description": "Path to edit, relative to the workspace root.",
        },
        "edits": {
            "type": "array",
            "minItems": 1,
            "description": (
                "Exact text replacements applied in array order. "
                "Each old_text must occur exactly once when its edit runs."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "old_text": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Exact text that must occur once.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["old_text", "new_text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["relative_path", "edits"],
    "additionalProperties": False,
}

_CREATE_FILE_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "relative_path": {
            "type": "string",
            "description": "New file path, relative to the workspace root.",
        },
        "content": {
            "type": "string",
            "description": "Complete UTF-8 text content for the new file.",
        },
    },
    "required": ["relative_path", "content"],
    "additionalProperties": False,
}

_LIST_FILES_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "relative_path": {
            "type": "string",
            "description": "Directory to list, relative to the workspace root or absolute. Use '.' for the workspace root.",
        },
    },
    "required": ["relative_path"],
    "additionalProperties": False,
}

_FIND_FILES_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "file_pattern": {"type": "string", "minLength": 1,
            "description": "Positive ripgrep glob relative to the search directory. '*.py' matches names at any depth; 'src/*.py' matches directly under src; '**/*.py' matches recursively. Use '/' separators."},
        "relative_path": {"type": "string",
            "description": "Search directory, relative to the workspace root or absolute. Defaults to '.'."},
        "ignore_patterns": {"type": "array", "items": {"type": "string", "minLength": 1},
            "description": "Exclude matching paths using ripgrep globs relative to the search directory, e.g. ['**/node_modules/**', '**/.git/**', '*.log']. A trailing '/' excludes directory contents. No leading '!'. Defaults to []; hidden files and gitignored files are otherwise included."},
    },
    "required": ["file_pattern"],
    "additionalProperties": False,
}

_SEARCH_TEXT_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1, "description": "Case-sensitive literal single-line text to locate; no regex."},
        "relative_path": {"type": "string", "description": "Directory or file, relative to the workspace or absolute. Defaults to '.'."},
        "file_pattern": {"type": "string", "minLength": 1, "description": "Positive ripgrep glob relative to the search directory; defaults to '*'."},
        "ignore_patterns": deepcopy(_FIND_FILES_PARAMETERS["properties"]["ignore_patterns"]),
    },
    "required": ["query"],
    "additionalProperties": False,
}

_RUN_COMMAND_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "argv": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
            "description": (
                "Command and arguments as separate strings. "
                "Do not use shell syntax such as pipes or redirects."
            ),
        },
        "relative_path": {
            "type": "string",
            "description": (
                "Working directory relative to the workspace root. "
                "Use '.' for the workspace root."
            ),
        },
        "timeout_seconds": {
            "type": "integer",
            "minimum": 1,
            "maximum": 120,
            "description": "Execution timeout in seconds, default 60, maximum 120; independent of model context size.",
        },
    },
    "required": ["argv", "relative_path", "timeout_seconds"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class ToolContext:
    """保存由 Agent 注入、模型不可见的单次任务运行时依赖。"""

    workspace: Path
    run_state: RunState
    context_manager: ContextManager | None = None
    history: ConversationHistory | None = None
    invoke_tool: Callable[[str, dict], ToolResult] | None = None
    check_execution: Callable[[], None] | None = None
    invoke_programmatic_tool: Callable[[str, dict], ToolResult] | None = None
    programmatic_call: bool = False


@dataclass(frozen=True, slots=True)
class Tool:
    """描述一个可以由 Agent 运行时执行的具名函数。"""

    name: str
    description: str
    parameters: dict[str, object]
    handler: ToolHandler
    requires_approval: bool = False
    # 仅注册时设置，不进入模型的工具定义或参数。
    parallel_safe: bool = False
    strict: bool = True
    programmatic: bool = False

    def to_definition(self) -> dict[str, object]:
        """返回可以直接传给模型 API 的工具定义。"""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            # 返回副本，避免调用方修改注册表内部保存的 Schema。
            "parameters": deepcopy(self.parameters),
            "strict": self.strict,
        }


@dataclass(frozen=True, slots=True)
class ToolResult:
    """表示工具成功输出或经过控制的失败结果。"""

    success: bool
    output: str
    structured_output: dict[str, object] | None = None


class ToolRegistry:
    """保存工具定义，并通过统一入口执行指定工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    @property
    def names(self) -> tuple[str, ...]:
        """按照注册顺序返回工具名称。"""
        return tuple(self._tools)

    @property
    def tools(self) -> tuple[Tool, ...]:
        """按照注册顺序返回工具元数据。"""
        return tuple(self._tools.values())

    def definitions(self) -> list[dict[str, object]]:
        """按照注册顺序返回全部模型工具定义。"""
        return [tool.to_definition() for tool in self._tools.values()]

    def register(self, tool: Tool) -> None:
        """注册一个工具，并拒绝重复的工具名称。"""
        if tool.name in self._tools:
            raise ValueError(f"tool is already registered: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def snapshot(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry._tools = self._tools.copy()
        return registry

    def requires_approval(self, name: str) -> bool:
        """返回指定工具是否必须获得用户确认。"""
        tool = self._tools.get(name)
        return tool.requires_approval if tool is not None else False

    def allows_programmatic(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool.programmatic if tool is not None else False

    def is_parallel_safe(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool.parallel_safe if tool is not None else False

    def argument_error(self, name: str, detail: str) -> ToolResult:
        """参数解析或执行前校验失败时，只返回本次需要修正的差异。"""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, f"unknown tool: {name}")
        return ToolResult(
            False,
            f"{detail}\n{name} 未执行，请修正以上参数后重试。",
        )

    def execute(
        self,
        name: str,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        """注入运行时上下文并将工具输出或异常转换成统一结果。"""
        prepared = self.prepare(name, arguments, context=context)
        if isinstance(prepared, ToolResult):
            return prepared
        try:
            result = prepared()
            return result if isinstance(result, ToolResult) else ToolResult(True, result)
        except Exception as error:
            return ToolResult(False, f"{type(error).__name__}: {error}")

    def prepare(
        self, name: str, arguments: Mapping[str, object], *, context: ToolContext,
    ) -> Callable[[], str | ToolResult] | ToolResult:
        """在主线程校验参数并绑定 handler；返回失败结果或待执行函数。"""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, f"unknown tool: {name}")
        if not isinstance(arguments, Mapping):
            return self.argument_error(name, "参数必须是 JSON 对象。")

        keyword_arguments = dict(arguments)
        issues = _argument_issues(tool, keyword_arguments)
        if issues:
            return self.argument_error(name, "TypeError: " + "\n".join(issues))
        try:
            # bind 可以在不执行工具的情况下发现缺失或多余的参数。
            signature(tool.handler).bind(context, **keyword_arguments)
        except TypeError as error:
            return self.argument_error(name, f"TypeError: {error}")
        return partial(tool.handler, context, **keyword_arguments)


def _argument_issues(tool: Tool, arguments: Mapping[str, object]) -> list[str]:
    """检查工具顶层参数；嵌套结构和业务约束继续由各工具校验。"""
    if not tool.strict:
        from jsonschema.validators import validator_for
        validator = validator_for(tool.parameters)(tool.parameters)
        return [error.message for error in validator.iter_errors(dict(arguments))]
    properties = tool.parameters.get("properties", {})
    issues: list[str] = []
    for name in arguments:
        if name not in properties:
            hint = ""
            if name == "max_output_tokens" and "max_output_characters" in properties:
                hint = "返回长度请使用 max_output_characters，单位为字符，不是 token。"
            else:
                matches = get_close_matches(name, properties, n=1, cutoff=0.75)
                if matches:
                    hint = f"是否要使用 {matches[0]}？"
            issues.append(f"未知参数 {name}：请删除该字段。{hint}")

    # Python 默认参数仍然可以省略，不把模型协议的 required 当成新的运行时限制。
    parameters = list(signature(tool.handler).parameters.values())[1:]
    for parameter in parameters:
        if (
            parameter.default is Parameter.empty
            and parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
            and parameter.name not in arguments
        ):
            description = properties.get(parameter.name, {}).get("description", "")
            issues.append(f"缺少必需参数 {parameter.name}。{description}")

    types = {"string": str, "integer": int, "number": (int, float),
             "boolean": bool, "array": list, "object": Mapping}
    for name, value in arguments.items():
        rule = properties.get(name)
        if rule is None:
            continue
        kind = rule.get("type")
        expected = types.get(kind)
        if expected is not None and (
            not isinstance(value, expected)
            or kind in ("integer", "number") and isinstance(value, bool)
        ):
            issues.append(f"{name} 应为 {kind}，实际类型为 {type(value).__name__}。")
            continue
        if "enum" in rule and value not in rule["enum"]:
            issues.append(f"{name} 必须是以下值之一：{rule['enum']}。")
        if kind in ("integer", "number"):
            if "minimum" in rule and value < rule["minimum"]:
                issues.append(f"{name}={value} 小于下限 {rule['minimum']}。")
            if "maximum" in rule and value > rule["maximum"]:
                issues.append(f"{name}={value} 超过上限 {rule['maximum']}。")
    return issues


def _read_location(context: ToolContext, name: str) -> tuple[Path, str]:
    if Path(name).is_absolute():
        # Local file references are readable across model-managed workspaces.
        # Write tools remain relative to the assigned directory; this is not a sandbox.
        path = Path(name).resolve()
        return (path, ".") if path.is_dir() else (path.parent, path.name)
    return context.workspace, name


def _inject_workspace(handler: ToolHandler, *, external_reads: bool = False) -> ToolHandler:
    """把使用 Path 的底层函数适配为接收 ToolContext 的工具 handler。"""

    @wraps(handler)
    def execute(context: ToolContext, **arguments: object) -> str | ToolResult:
        workspace = context.workspace
        if external_reads:
            workspace, arguments["relative_path"] = _read_location(context, arguments.get("relative_path", "."))
        data = handler(workspace, **arguments)
        if isinstance(data, dict):
            return ToolResult(True, json.dumps(data, ensure_ascii=False), structured_output=data)
        return data

    return execute


def create_default_registry(policy: ContextPolicy | None = None, *,
                            check_execution: Callable[[], None] | None = None) -> ToolRegistry:
    """创建包含全部内置工具的注册表。"""
    # planning.tool 依赖这里的 Tool 类型，因此等 Registry 初始化完成再导入。
    from praxis_agent.planning.tool import create_update_plan_tool
    from praxis_agent.context.capacity import ContextPolicy

    policy = policy or ContextPolicy()
    def read_with_limits(
        context: ToolContext, relative_path: str, start_line: int = 1,
        start_offset: int = 0, end_line: int | None = None, end_offset: int | None = None,
    ) -> ToolResult:
        workspace, relative_path = _read_location(context, relative_path)
        data = read_file(
            workspace, relative_path, start_line, start_offset, end_line, end_offset,
            character_limit=None if context.programmatic_call else min(
                policy.max_file_read_chars, policy.max_tool_output_chars),
        )
        return ToolResult(True, json.dumps(data, ensure_ascii=False), structured_output=data)

    def list_directory(context: ToolContext, relative_path: str = ".") -> ToolResult:
        workspace, relative_path = _read_location(context, relative_path)
        data = list_files(workspace, relative_path)
        return ToolResult(True, json.dumps(data, ensure_ascii=False), structured_output=data)

    def find_paths(context: ToolContext, file_pattern: str, relative_path: str = ".",
                   ignore_patterns: list[str] | None = None) -> ToolResult:
        workspace, relative_path = _read_location(context, relative_path)
        data = find_files(workspace, file_pattern, relative_path, ignore_patterns)
        return ToolResult(True, json.dumps(data, ensure_ascii=False), structured_output=data)

    registry = ToolRegistry()
    def command_with_control(context: ToolContext, argv: list[str], relative_path: str = ".",
                             timeout_seconds: int = 60) -> str:
        return run_command(context.workspace, argv, relative_path, timeout_seconds,
                           check_execution=check_execution)
    registry.register(
        Tool(
            name="read_file",
            programmatic=True,
            description=("Read a UTF-8 file range as structured text/truncated/next_position/note. "
                         "Positions use one-based lines and zero-based character offsets (including line endings); "
                         "start inclusive, end exclusive. Omit both end fields or pass null for EOF. "
                         "Model reads are bounded: continue from next_position with the same endpoint. "
                         "Programmatic reads return the entire requested range as an object. "
                         "truncated=false means the requested range is complete, not necessarily the whole file."),
            parameters=_READ_FILE_PARAMETERS,
            handler=read_with_limits,
            strict=False,
            parallel_safe=True,
        )
    )
    registry.register(
        Tool(
            name="edit_file",
            programmatic=True,
            description=(
                "Apply multiple exact text replacements to one existing "
                "UTF-8 workspace file in a single all-or-nothing edit. Returns a path/diff object."
            ),
            parameters=_EDIT_FILE_PARAMETERS,
            handler=_inject_workspace(edit_file),
        )
    )
    registry.register(
        Tool(
            name="create_file",
            programmatic=True,
            description="Create a new UTF-8 workspace file; returns a path/diff object. Existing files are rejected.",
            parameters=_CREATE_FILE_PARAMETERS,
            handler=_inject_workspace(create_file),
        )
    )
    registry.register(
        Tool(
            name="list_files",
            programmatic=True,
            description=(
                "List all immediate files and subdirectories, including hidden and ignored names. "
                "Returns an entries object with path and type, directories first. "
                "No recursion or entry limit; list a subdirectory explicitly to explore it."
            ),
            parameters=_LIST_FILES_PARAMETERS,
            handler=list_directory,
            parallel_safe=True,
        )
    )
    registry.register(
        Tool(
            name="find_files",
            programmatic=True,
            description=(
                "Find all files by case-sensitive glob, sorted by path, as a files object. "
                "No result limit, no automatic ignore rules; hidden files are included. "
                "Use ignore_patterns for exclusions and relative_path to scope the search."
            ),
            parameters=_FIND_FILES_PARAMETERS,
            handler=find_paths,
            strict=False,
            parallel_safe=True,
        )
    )
    registry.register(
        Tool(
            name="search_text",
            programmatic=True,
            description=(
                "Locate all non-overlapping literal matches in UTF-8 files. Returns a matches object "
                "with path, one-based line and zero-based character offset for each occurrence; "
                "no source text. Use read_file to inspect. No size/result caps or automatic ignores; "
                "only ignore_patterns excludes paths. Same-line occurrences are separate records."
            ),
            parameters=_SEARCH_TEXT_PARAMETERS,
            handler=_inject_workspace(search_text, external_reads=True),
            strict=False,
            parallel_safe=True,
        )
    )
    registry.register(
        Tool(
            name="run_command",
            description=(
                "Run a non-interactive command inside the workspace and "
                "return its exit code, stdout, and stderr."
            ),
            parameters=_RUN_COMMAND_PARAMETERS,
            handler=command_with_control,
            requires_approval=True,
        )
    )
    registry.register(create_update_plan_tool())
    from praxis_agent.context.tools import create_context_tools

    for tool in create_context_tools(max_history_read_chars=min(policy.max_history_read_chars, policy.max_tool_output_chars)):
        registry.register(tool)
    return registry
