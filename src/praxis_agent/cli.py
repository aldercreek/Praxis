import argparse
from collections.abc import Sequence
from pathlib import Path

from praxis_agent.agent import Agent, AgentError, create_default_agent
from praxis_agent.model import ModelClientError
from praxis_agent.terminal import (
    approve_tool_call,
    show_tool_call,
    show_tool_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="praxis",
        description="Run the Praxis coding agent.",
    )
    parser.add_argument("task", nargs="*", help="Task for the agent")
    return parser


def _run_task(agent: Agent, task: str) -> int:
    """执行一轮任务，并将预期内的错误转换为终端输出。"""
    try:
        print(agent.run(task))
    except (AgentError, ModelClientError, ValueError) as error:
        print(f"error: {error}")
        return 1
    return 0


def _run_interactive(agent: Agent) -> int:
    """持续读取任务，直到用户退出当前会话。"""
    while True:
        try:
            task = input("praxis> ").strip()
            if not task:
                continue

            command = task.lower()
            if command == "/exit":
                return 0
            if command == "/clear":
                agent.clear_history()
                print("Conversation cleared.")
                continue

            # 单轮失败不会结束交互模式，用户可以修正任务后继续。
            _run_task(agent, task)
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 130


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task = " ".join(args.task).strip()

    try:
        agent = create_default_agent(
            Path.cwd(),
            on_tool_call=show_tool_call,
            on_tool_result=show_tool_result,
            approve_tool_call=approve_tool_call,
        )
    except ValueError as error:
        print(f"error: {error}")
        return 1

    if task:
        return _run_task(agent, task)
    return _run_interactive(agent)
