from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from praxis_agent.agent import Agent, AgentError, create_default_agent
from praxis_agent.config import LangfuseConfig
from praxis_agent.events import CompositeEventSink, EventSink
from praxis_agent.model import ModelClientError
from praxis_agent.sessions import SQLiteSessionStore, SessionStoreError
from praxis_agent.terminal import (
    ConsoleRenderer,
    approve_tool_call,
    print_session_list,
    print_session_resumed,
)

if TYPE_CHECKING:
    from praxis_agent.observability import LangfuseEventSink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="praxis",
        description="Run the Praxis coding agent.",
    )
    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument(
        "-c",
        "--continue",
        dest="resume_latest",
        action="store_true",
        help="Continue the latest session for the current workspace",
    )
    session_group.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="Resume a specific session",
    )
    session_group.add_argument(
        "--sessions",
        dest="list_sessions",
        action="store_true",
        help="List saved sessions for the current workspace",
    )
    parser.add_argument(
        "--bypass",
        action="store_true",
        help="Run tools without approval prompts for this process",
    )
    parser.add_argument("task", nargs="*", help="Task for the agent")
    return parser


def _run_task(agent: Agent, task: str) -> int:
    """执行一轮任务，并将预期内的错误转换为终端输出。"""
    try:
        agent.run(task)
    except (
        AgentError,
        ModelClientError,
        SessionStoreError,
        ValueError,
    ) as error:
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
                try:
                    agent.clear_history()
                except (SessionStoreError, ValueError) as error:
                    print(f"error: {error}")
                    continue
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


def _list_sessions(workspace: Path) -> int:
    """列出当前工作区会话，不初始化模型或可观测性组件。"""
    try:
        sessions = SQLiteSessionStore().list_sessions(workspace)
    except (SessionStoreError, ValueError) as error:
        print(f"error: {error}")
        return 1
    print_session_list(sessions, workspace.resolve())
    return 0


def _build_event_sink(
    workspace: Path,
) -> tuple[EventSink, LangfuseEventSink | None]:
    """创建终端消费者，并在密钥齐全时附加 Langfuse 消费者。"""
    renderer = ConsoleRenderer()
    try:
        config = LangfuseConfig.from_env()
        if config is None:
            return renderer, None
        # SDK 依赖较重，只在真正启用观测时加载。
        from praxis_agent.observability import LangfuseEventSink

        langfuse_sink = LangfuseEventSink(config, workspace)
    except Exception as error:
        # 可观测性属于辅助能力，配置或初始化失败不能阻止 Agent 工作。
        print(f"warning: Langfuse is disabled: {error}", file=sys.stderr)
        return renderer, None

    return CompositeEventSink((renderer, langfuse_sink)), langfuse_sink


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_sessions and args.task:
        parser.error("--sessions cannot be used with a task")

    task = " ".join(args.task).strip()
    workspace = Path.cwd()
    if args.list_sessions:
        return _list_sessions(workspace)

    event_sink, langfuse_sink = _build_event_sink(workspace)

    try:
        agent = create_default_agent(
            workspace,
            event_sink=event_sink,
            approve_tool_call=approve_tool_call,
            bypass_approvals=args.bypass,
            resume_latest=args.resume_latest,
            resume_session_id=args.resume,
        )
    except (SessionStoreError, ValueError) as error:
        print(f"error: {error}")
        if langfuse_sink is not None:
            langfuse_sink.close()
        return 1

    if args.resume_latest or args.resume is not None:
        print_session_resumed(
            agent.session_id or "",
            agent.history_item_count,
        )

    try:
        if task:
            return _run_task(agent, task)
        return _run_interactive(agent)
    finally:
        if langfuse_sink is not None:
            langfuse_sink.close()
