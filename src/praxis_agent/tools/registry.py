from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import signature
from pathlib import Path

from praxis_agent.tools.filesystem import (
    create_file,
    list_files,
    read_file,
    replace_text,
    search_text,
)


ToolHandler = Callable[..., str]


@dataclass(frozen=True, slots=True)
class Tool:
    """描述一个可以由 Agent 运行时执行的具名函数。"""

    name: str
    description: str
    handler: ToolHandler


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

    def register(self, tool: Tool) -> None:
        """注册一个工具，并拒绝重复的工具名称。"""
        if tool.name in self._tools:
            raise ValueError(f"tool is already registered: {tool.name}")
        self._tools[tool.name] = tool

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
    registry.register(Tool("read_file", "Read a UTF-8 workspace file.", read_file))
    registry.register(
        Tool(
            "replace_text",
            "Replace one exact text match in an existing workspace file.",
            replace_text,
        )
    )
    registry.register(
        Tool("create_file", "Create a new UTF-8 workspace file.", create_file)
    )
    registry.register(
        Tool("list_files", "List files under a workspace path.", list_files)
    )
    registry.register(
        Tool("search_text", "Search workspace text files.", search_text)
    )
    return registry
