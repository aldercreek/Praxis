from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from inspect import signature
from pathlib import Path

from praxis_agent.tools.command import run_command
from praxis_agent.tools.filesystem import (
    create_file,
    list_files,
    read_file,
    replace_text,
    search_text,
)


ToolHandler = Callable[..., str]


# parameters 使用 JSON Schema 约束模型生成的工具参数。
_READ_FILE_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "relative_path": {
            "type": "string",
            "description": "Path to read, relative to the workspace root.",
        }
    },
    "required": ["relative_path"],
    "additionalProperties": False,
}

_REPLACE_TEXT_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "relative_path": {
            "type": "string",
            "description": "Path to edit, relative to the workspace root.",
        },
        "old_text": {
            "type": "string",
            "description": "Exact text that must occur once in the file.",
        },
        "new_text": {
            "type": "string",
            "description": "Replacement text.",
        },
    },
    "required": ["relative_path", "old_text", "new_text"],
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
            "description": "Maximum number of file paths to return.",
        },
    },
    "required": ["relative_path", "max_entries"],
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
    },
    "required": ["query", "relative_path", "max_results"],
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
    """保存绑定到工作区的工具，并通过统一入口执行。"""

    def __init__(self, workspace: Path) -> None:
        workspace_root = workspace.resolve()
        if not workspace_root.is_dir():
            raise NotADirectoryError(f"workspace is not a directory: {workspace}")

        self.workspace = workspace_root
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
    ) -> ToolResult:
        """执行一个工具，并将输出或错误转换成统一结果。"""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, f"unknown tool: {name}")
        if not isinstance(arguments, Mapping):
            return ToolResult(False, "tool arguments must be a mapping")

        keyword_arguments = dict(arguments)
        try:
            # bind 可以在不执行工具的情况下发现缺失或多余的参数。
            signature(tool.handler).bind(self.workspace, **keyword_arguments)
            output = tool.handler(self.workspace, **keyword_arguments)
        except Exception as error:
            # 将工具异常转换为结果，保证未来的 Agent 循环可以继续运行。
            return ToolResult(False, f"{type(error).__name__}: {error}")

        return ToolResult(True, output)


def create_default_registry(workspace: Path) -> ToolRegistry:
    """创建包含全部内置文件系统工具的注册表。"""
    registry = ToolRegistry(workspace)
    registry.register(
        Tool(
            name="read_file",
            description="Read a UTF-8 workspace file.",
            parameters=_READ_FILE_PARAMETERS,
            handler=read_file,
        )
    )
    registry.register(
        Tool(
            name="replace_text",
            description="Replace one exact text match in an existing workspace file.",
            parameters=_REPLACE_TEXT_PARAMETERS,
            handler=replace_text,
        )
    )
    registry.register(
        Tool(
            name="create_file",
            description="Create a new UTF-8 workspace file.",
            parameters=_CREATE_FILE_PARAMETERS,
            handler=create_file,
        )
    )
    registry.register(
        Tool(
            name="list_files",
            description="List files under a workspace path.",
            parameters=_LIST_FILES_PARAMETERS,
            handler=list_files,
        )
    )
    registry.register(
        Tool(
            name="search_text",
            description="Search workspace text files.",
            parameters=_SEARCH_TEXT_PARAMETERS,
            handler=search_text,
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
            handler=run_command,
            requires_approval=True,
        )
    )
    return registry
