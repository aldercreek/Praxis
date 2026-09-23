from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
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
    from praxis_agent.context.manager import ContextManager
    from praxis_agent.context.models import ConversationHistory
    from praxis_agent.planning.state import RunState


ToolHandler = Callable[..., str]


# parameters 使用 JSON Schema 约束模型生成的工具参数。
_READ_FILE_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "relative_path": {
            "type": "string",
            "description": "Path to read, relative to the workspace root.",
        },
        "start_line": {
            "type": "integer",
            "minimum": 1,
            "description": "First one-based line to return.",
        },
        "max_lines": {
            "type": "integer",
            "minimum": 1,
            "maximum": 400,
            "description": "Maximum number of lines to return.",
        },
    },
    "required": ["relative_path", "start_line", "max_lines"],
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
            "description": "Directory or file path relative to the workspace root. Use '.' for the whole workspace.",
        },
        "max_entries": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "Maximum number of file and directory entries to return."
            ),
        },
        "max_depth": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "Maximum traversal depth. Direct children have depth 1. "
                "Use 1 to locate candidate project directories before exploring them."
            ),
        },
        "entry_type": {
            "type": "string",
            "enum": ["all", "files", "directories"],
            "description": "Use directories to locate projects; all for a project overview; files to omit directory entries.",
        },
    },
    "required": ["relative_path", "max_entries", "max_depth", "entry_type"],
    "additionalProperties": False,
}

_FIND_FILES_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "file_pattern": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Glob pattern for file paths, for example '*.py' or "
                "'src/*.py'."
            ),
        },
        "relative_path": {
            "type": "string",
            "description": (
                "Directory or file path relative to the workspace root. "
                "Use '.' for the whole workspace."
            ),
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum number of matching file paths to return.",
        },
        "include_hidden": {
            "type": "boolean",
            "description": "Use false normally. Set true when searching hidden files or directories such as .github. Ignore rules still apply.",
        },
    },
    "required": ["file_pattern", "relative_path", "max_results", "include_hidden"],
    "additionalProperties": False,
}

_SEARCH_TEXT_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "description": "Case-sensitive fixed text to find.",
        },
        "relative_path": {
            "type": "string",
            "description": "Directory or file path relative to the workspace root. Use '.' for the whole workspace.",
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum number of matching lines to return.",
        },
        "max_line_characters": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum characters of text per matching line. Use 500 normally. Longer lines return a snippet around the first match with column positions.",
        },
        "max_output_characters": {
            "type": "integer",
            "minimum": 512,
            "description": "Maximum total returned characters including paths and truncation notice. Use 20000 normally; increase only for a narrowly scoped search.",
        },
        "file_pattern": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Glob pattern for files, for example '*.py'. "
                "Use '*' to search all files."
            ),
        },
    },
    "required": [
        "query",
        "relative_path",
        "max_results",
        "file_pattern",
        "max_line_characters",
        "max_output_characters",
    ],
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
            "description": "Maximum number of seconds to wait.",
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


@dataclass(frozen=True, slots=True)
class Tool:
    """描述一个可以由 Agent 运行时执行的具名函数。"""

    name: str
    description: str
    parameters: dict[str, object]
    handler: ToolHandler
    requires_approval: bool = False

    def to_definition(self) -> dict[str, object]:
        """返回可以直接传给模型 API 的工具定义。"""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            # 返回副本，避免调用方修改注册表内部保存的 Schema。
            "parameters": deepcopy(self.parameters),
            "strict": True,
        }


@dataclass(frozen=True, slots=True)
class ToolResult:
    """表示工具成功输出或经过控制的失败结果。"""

    success: bool
    output: str


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

    def requires_approval(self, name: str) -> bool:
        """返回指定工具是否必须获得用户确认。"""
        tool = self._tools.get(name)
        return tool.requires_approval if tool is not None else False

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
        try:
            output = tool.handler(context, **keyword_arguments)
        except Exception as error:
            # 将工具异常转换为结果，保证未来的 Agent 循环可以继续运行。
            return ToolResult(False, f"{type(error).__name__}: {error}")

        return ToolResult(True, output)


def _argument_issues(tool: Tool, arguments: Mapping[str, object]) -> list[str]:
    """检查工具顶层参数；嵌套结构和业务约束继续由各工具校验。"""
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
                hint = "请分段读取。" if tool.name == "read_file" and name == "max_lines" else ""
                issues.append(f"{name}={value} 超过上限 {rule['maximum']}。{hint}")
    return issues


def _inject_workspace(handler: ToolHandler) -> ToolHandler:
    """把使用 Path 的底层函数适配为接收 ToolContext 的工具 handler。"""

    @wraps(handler)
    def execute(context: ToolContext, **arguments: object) -> str:
        return handler(context.workspace, **arguments)

    return execute


def create_default_registry() -> ToolRegistry:
    """创建包含全部内置工具的注册表。"""
    # planning.tool 依赖这里的 Tool 类型，因此等 Registry 初始化完成再导入。
    from praxis_agent.planning.tool import create_update_plan_tool

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="read_file",
            description="Read a bounded line range from a UTF-8 workspace file.",
            parameters=_READ_FILE_PARAMETERS,
            handler=_inject_workspace(read_file),
        )
    )
    registry.register(
        Tool(
            name="edit_file",
            description=(
                "Apply multiple exact text replacements to one existing "
                "UTF-8 workspace file in a single all-or-nothing edit."
            ),
            parameters=_EDIT_FILE_PARAMETERS,
            handler=_inject_workspace(edit_file),
        )
    )
    registry.register(
        Tool(
            name="create_file",
            description="Create a new UTF-8 workspace file.",
            parameters=_CREATE_FILE_PARAMETERS,
            handler=_inject_workspace(create_file),
        )
    )
    registry.register(
        Tool(
            name="list_files",
            description=(
                "List workspace entries by bounded depth, shallow levels first. "
                "To locate a project, use entry_type=directories and max_depth=1, "
                "then inspect likely parent directories."
            ),
            parameters=_LIST_FILES_PARAMETERS,
            handler=_inject_workspace(list_files),
        )
    )
    registry.register(
        Tool(
            name="find_files",
            description=(
                "Find files, not directory names, by case-sensitive glob pattern. "
                "Locate the project with list_files first, then scope relative_path "
                "to that project. Limited results are sorted for display but the "
                "selected subset may vary between searches."
            ),
            parameters=_FIND_FILES_PARAMETERS,
            handler=_inject_workspace(find_files),
        )
    )
    registry.register(
        Tool(
            name="search_text",
            description=(
                "Search fixed text in workspace files using a glob pattern. "
                "Returns path:line:text, bounded by per-line and total character limits. "
                "Long lines show the first match's neighborhood; increase limits or narrow "
                "the file and query to inspect more."
            ),
            parameters=_SEARCH_TEXT_PARAMETERS,
            handler=_inject_workspace(search_text),
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
            handler=_inject_workspace(run_command),
            requires_approval=True,
        )
    )
    registry.register(create_update_plan_tool())
    from praxis_agent.context.tools import create_context_tools

    for tool in create_context_tools():
        registry.register(tool)
    return registry
