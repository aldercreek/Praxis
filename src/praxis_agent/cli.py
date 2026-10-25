from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import httpx

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
        help="Enable approval bypass on the shared local service (requires no active turns)",
    )
    parser.add_argument("task", nargs="*", help="Task for the agent")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_sessions and args.task:
        parser.error("--sessions cannot be used with a task")
    from datetime import datetime
    from urllib.parse import urlencode

    from praxis_agent.client import ServiceClient
    from praxis_agent.planning.state import RunStatus
    from praxis_agent.sessions.models import SessionSummary
    from praxis_agent.terminal.app import PraxisApp
    from praxis_agent.terminal.connection import TerminalSession
    from praxis_agent.terminal.views import ViewEventSink

    client = None
    try:
        client = ServiceClient()
        workspace = Path.cwd().resolve()
        sessions = client.request("GET", "/api/sessions?" + urlencode({"workspace": str(workspace)}))["sessions"]
        if args.list_sessions:
            print_session_list([SessionSummary(item["id"], item["title"], datetime.fromisoformat(item["updated_at"]),
                                RunStatus(item["latest_turn"]["status"]) if item.get("latest_turn") else None)
                                for item in sessions], workspace)
            return 0
        if args.bypass:
            client.request("PUT", "/api/settings", {"bypass_approvals": True})
        if args.resume:
            matches = [item for item in sessions if item["id"].startswith(args.resume)]
            if len(matches) != 1:
                raise ValueError("Session ID does not identify one session in this workspace")
            session = matches[0]
        elif args.resume_latest:
            if not sessions:
                raise ValueError("No session found for this workspace")
            session = sessions[0]
        else:
            session = client.create(workspace)
        connection = TerminalSession(client, session)
        app = PraxisApp(ViewEventSink(), initial_task=" ".join(args.task).strip(), connection=connection)
        return app.run() or 0
    except (RuntimeError, ValueError, OSError, httpx.HTTPError) as error:
        print(f"error: {error}")
        return 1
    finally:
        if client:
            client.close()
