from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from praxis_agent.agent import create_default_agent
from praxis_agent.sessions import SessionStoreError, SQLiteSessionStore
from praxis_agent.terminal.sessions import print_session_list


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
    """列出当前工作区会话，不初始化模型。"""
    try:
        sessions = SQLiteSessionStore().list_sessions(workspace)
    except (SessionStoreError, ValueError) as error:
        print(f"error: {error}")
        return 1
    print_session_list(sessions, workspace.resolve())
    return 0


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

    try:
        agent = create_default_agent(
            workspace,
            event_sink=display_sink,
            approve_tool_call=app.approve_tool_call,
            bypass_approvals=args.bypass,
            resume_latest=args.resume_latest,
            resume_session_id=args.resume,
        )
    except (SessionStoreError, ValueError) as error:
        print(f"error: {error}")
        return 1

    app.agent = agent
    try:
        if args.resume_latest or args.resume is not None:
            app.store.views["main"].restore_history(agent.session_store.load_conversation_items(agent.session_id))
        return app.run() or 0
    finally:
        agent.close()
