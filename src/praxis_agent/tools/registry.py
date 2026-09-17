from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from inspect import signature
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
                "Maximum traversal depth. Direct children have depth 1."
            ),
        },
    },
    "required": ["relative_path", "max_entries", "max_depth"],
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
    },
    "required": ["file_pattern", "relative_path", "max_results"],
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
            return ToolResult(False, "tool arguments must be a mapping")

        keyword_arguments = dict(arguments)
        try:
            # bind 可以在不执行工具的情况下发现缺失或多余的参数。
            signature(tool.handler).bind(context, **keyword_arguments)
            output = tool.handler(context, **keyword_arguments)
        except Exception as error:
            # 将工具异常转换为结果，保证未来的 Agent 循环可以继续运行。
            return ToolResult(False, f"{type(error).__name__}: {error}")

        return ToolResult(True, output)


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
            description="List files and directories under a workspace path.",
            parameters=_LIST_FILES_PARAMETERS,
            handler=_inject_workspace(list_files),
        )
    )
    registry.register(
        Tool(
            name="find_files",
            description="Find workspace files by name or glob pattern.",
            parameters=_FIND_FILES_PARAMETERS,
            handler=_inject_workspace(find_files),
        )
    )
    registry.register(
        Tool(
            name="search_text",
            description="Search fixed text in workspace files using a glob pattern.",
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
