import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from praxis_agent.agent import AgentError, create_default_agent
from praxis_agent.model import ModelClientError
from praxis_agent.tools import ToolResult


_MAX_ARGUMENT_DISPLAY_CHARS = 500
_MAX_RESULT_DISPLAY_CHARS = 4_000


def _truncate_for_display(text: str, max_chars: int) -> str:
    """限制终端展示长度，同时保留输出的开头和结尾。"""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + "\n... terminal output truncated ...\n"
        + text[-half:]
    )


def _show_tool_call(
    name: str,
    arguments: Mapping[str, object],
) -> None:
    """在终端展示模型准备调用的工具和参数。"""
    encoded_arguments = json.dumps(dict(arguments), ensure_ascii=False)
    visible_arguments = _truncate_for_display(
        encoded_arguments,
        _MAX_ARGUMENT_DISPLAY_CHARS,
    )
    print(f"[tool] {name} {visible_arguments}")


def _show_tool_result(name: str, result: ToolResult) -> None:
    """展示工具状态；命令工具还会展示受限长度的运行输出。"""
    status = "ok" if result.success else "error"
    print(f"[tool:{status}] {name}")
    if name == "run_command":
        print(
            _truncate_for_display(
                result.output,
                _MAX_RESULT_DISPLAY_CHARS,
            )
        )


def _approve_tool_call(
    name: str,
    arguments: Mapping[str, object],
) -> bool:
    """询问用户是否允许执行需要审批的工具。"""
    del arguments
    try:
        answer = input(f"Allow {name}? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="praxis",
        description="Run the Praxis coding agent.",
    )
    parser.add_argument("task", nargs="*", help="Task for the agent")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task = " ".join(args.task).strip()

    if not task:
        task = input("praxis> ").strip()

    try:
        agent = create_default_agent(
            Path.cwd(),
            on_tool_call=_show_tool_call,
            on_tool_result=_show_tool_result,
            approve_tool_call=_approve_tool_call,
        )
        print(agent.run(task))
    except (AgentError, ModelClientError, ValueError) as error:
        print(f"error: {error}")
        return 1

    return 0
