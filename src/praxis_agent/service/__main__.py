from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import socket
from pathlib import Path
from uuid import uuid4

import uvicorn

from praxis_agent.sessions import SQLiteSessionStore
from praxis_agent.sessions.database import data_directory

from .discovery import ServiceLock, atomic_write, ensure_service
from .http import PROTOCOL_VERSION, create_app
from .manager import SessionManager
from .store import ServiceStore


async def serve(directory):
    lock = ServiceLock(directory)
    if not lock.acquire():
        return
    manager = None
    listener = None
    instance = uuid4().hex
    try:
        # No database opening or recovery before winning process ownership.
        store = ServiceStore(SQLiteSessionStore(directory / "state.sqlite3"))
        store.recover()
        manager = SessionManager(store)
        token = secrets.token_urlsafe(32)
        atomic_write(directory / "service.token", token)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        listener.setblocking(False)
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(manager, token=token, instance_id=instance),
                log_level="warning",
                access_log=False,
                timeout_graceful_shutdown=3,
            )
        )
        task = asyncio.create_task(server.serve(sockets=[listener]))
        while not server.started and not task.done():
            await asyncio.sleep(0.01)
        if server.started:
            atomic_write(
                directory / "service.json",
                json.dumps(
                    dict(
                        pid=os.getpid(),
                        instance_id=instance,
                        port=port,
                        protocol_version=PROTOCOL_VERSION,
                    )
                ),
            )
        await task
    finally:
        if manager:
            await asyncio.to_thread(manager.close)
        registration = directory / "service.json"
        try:
            if json.loads(registration.read_text())["instance_id"] == instance:
                registration.unlink()
        except (OSError, ValueError, KeyError):
            pass
        if listener:
            listener.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description="Praxis shared local service")
    parser.add_argument("--data-dir", type=Path, default=data_directory())
    parser.add_argument(
        "--ensure",
        action="store_true",
        help="Discover or launch a detached service, print its registration, then exit",
    )
    args = parser.parse_args()
    directory = args.data_dir.expanduser().resolve()
    logging.basicConfig(level=logging.INFO)
    if args.ensure:
        record = ensure_service(directory)
        print(
            json.dumps({key: value for key, value in record.items() if key != "token"})
        )
    else:
        asyncio.run(serve(directory))


if __name__ == "__main__":
    main()
