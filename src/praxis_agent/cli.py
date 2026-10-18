from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from praxis_agent.agent import create_default_agent
from praxis_agent.config import LangfuseConfig
from praxis_agent.events import CompositeEventSink, EventSink
from praxis_agent.sessions import SessionStoreError, SQLiteSessionStore
from praxis_agent.terminal.sessions import print_session_list

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
    renderer: EventSink,
) -> tuple[EventSink, LangfuseEventSink | None]:
    """创建终端消费者，并在密钥齐全时附加 Langfuse 消费者。"""
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

    from praxis_agent.terminal.app import PraxisApp
    from praxis_agent.terminal.views import ViewEventSink

    display_sink = ViewEventSink()
    app = PraxisApp(display_sink, initial_task=task)
    event_sink, langfuse_sink = _build_event_sink(workspace, display_sink)

    try:
        agent = create_default_agent(
            workspace,
            event_sink=event_sink,
            approve_tool_call=app.approve_tool_call,
            bypass_approvals=args.bypass,
            resume_latest=args.resume_latest,
            resume_session_id=args.resume,
        )
    except (SessionStoreError, ValueError) as error:
        print(f"error: {error}")
        if langfuse_sink is not None:
            langfuse_sink.close()
        return 1

    app.agent = agent
    try:
        if args.resume_latest or args.resume is not None:
            app.store.views["main"].restore_history(agent.session_store.load_conversation_items(agent.session_id))
        return app.run() or 0
    finally:
        try:
            agent.close()
        finally:
            if langfuse_sink is not None:
                langfuse_sink.close()
