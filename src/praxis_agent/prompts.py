import json

from praxis_agent.runtime import RuntimeContext


BASE_INSTRUCTIONS = """You are Praxis, a coding agent working in a local workspace.
Use the provided tools to inspect and modify files when needed.
Prefer file tools over shell commands when inspecting the workspace.
Search first when possible, then read only the relevant line ranges.
All tool paths must be relative to the workspace root.
Do not try to use a shell listed as unavailable in the runtime context.
After modifying code, run relevant checks when they are available.
Do not retry a command that the user declined.
Do not claim that you changed or inspected something unless a tool result confirms it.
When the task is complete, give a concise summary of the result.
"""


def build_instructions(runtime_context: RuntimeContext) -> str:
    """把基础行为约束和启动时的运行环境事实组合成模型指令。"""
    available_shells = _format_shells(runtime_context.available_shells)
    unavailable_shells = _format_shells(runtime_context.unavailable_shells)
    workspace_root = json.dumps(
        str(runtime_context.workspace_root),
        ensure_ascii=False,
    )
    runtime_section = "\n".join(
        (
            "Runtime context captured at startup:",
            (
                "- Operating system: "
                f"{runtime_context.operating_system} "
                f"({runtime_context.sys_platform})"
            ),
            f"- Workspace root: {workspace_root}",
            f"- Available shells: {available_shells}",
            f"- Unavailable shells: {unavailable_shells}",
        )
    )
    return f"{BASE_INSTRUCTIONS.rstrip()}\n\n{runtime_section}"


def _format_shells(shells: tuple[str, ...]) -> str:
    """把 shell 名称转换为简短且稳定的提示词文本。"""
    return ", ".join(shells) if shells else "none detected"
