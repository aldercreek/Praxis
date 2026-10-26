"""OS process ownership, private registration and bounded client discovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx

from praxis_agent.sessions.database import data_directory

from .http import PROTOCOL_VERSION


class ServiceUnavailable(RuntimeError):
    pass


class ServiceLock:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.file = None

    def acquire(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.file = (self.directory / "service.lock").open("a+b")
        self.file.seek(0, 2)
        if not self.file.tell():
            self.file.write(b"\0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            self.file = None
            return False
        return True

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


def atomic_write(path: Path, text: str):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def discover(directory=None):
    directory = Path(directory or data_directory())
    try:
        record = json.loads((directory / "service.json").read_text(encoding="utf-8"))
        token = (directory / "service.token").read_text(encoding="utf-8").strip()
        port = record["port"]
        if type(port) is not int or not 0 < port < 65536:
            return None
        endpoint = f"http://127.0.0.1:{port}"
        response = httpx.get(
            endpoint + "/api/health",
            headers={"Authorization": f"Bearer {token}"},
            timeout=1,
            trust_env=False,
        )
        if response.status_code != 200:
            return None
        health = response.json()
        if (
            health.get("service") != "praxis"
            or health.get("instance_id") != record["instance_id"]
        ):
            return None
        if health.get("protocol_version") != PROTOCOL_VERSION:
            raise ServiceUnavailable(
                "Incompatible Praxis protocol; update clients and service together"
            )
        if not health.get("ready"):
            return None
        return {**record, "url": endpoint, "token": token}
    except (OSError, ValueError, KeyError, httpx.HTTPError):
        return None


def ensure_service(directory=None, *, timeout=15):
    directory = Path(directory or data_directory()).resolve()
    existing = discover(directory)
    if existing:
        return existing
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    args = [sys.executable, "-m", "praxis_agent.service", "--data-dir", str(directory)]
    options = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, close_fds=True)
    if os.name == "nt":
        # DETACHED_PROCESS lets the virtualenv launcher create a visible console
        # for the real interpreter. Start without a console window instead.
        options["creationflags"] = (
            subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        options["start_new_session"] = True
    with (directory / "service.log").open("ab") as log:
        subprocess.Popen(args, stderr=log, **options)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        existing = discover(directory)
        if existing:
            return existing
        time.sleep(0.2)
    raise ServiceUnavailable(
        "Praxis did not become ready. Check service.log; an unresponsive lock owner is never replaced automatically."
    )
