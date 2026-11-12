"""Own a loopback-only Phoenix child; never attach to an unknown occupied port."""
from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path
import socket
import subprocess
import sys
from threading import Event, RLock
from time import monotonic

import httpx


class LocalPhoenix:
    def __init__(self, directory: Path):
        self.directory = directory
        self.process = None
        self.lock = RLock()

    @property
    def alive(self):
        return self.process is not None and self.process.poll() is None

    def start(self, stopping: Event):
        if find_spec("phoenix") is None:
            raise RuntimeError(
                f'Phoenix is not installed for {sys.executable}. Install it with: '
                f'"{sys.executable}" -m pip install "arize-phoenix>=21,<22"'
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        log_path = self.directory / "server.log"
        # CLI does not preserve --port 0, so retry a lost bind race explicitly.
        for attempt in range(3):
            if stopping.is_set():
                raise RuntimeError("Phoenix startup cancelled")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            url = f"http://127.0.0.1:{port}"
            env = {k: v for k, v in os.environ.items() if not k.startswith("PHOENIX_")}
            env.update(PHOENIX_HOST="127.0.0.1", PHOENIX_WORKING_DIR=str(self.directory),
                       PHOENIX_TELEMETRY_ENABLED="false", PHOENIX_ALLOW_EXTERNAL_RESOURCES="false",
                       PHOENIX_DISABLE_AGENT_ASSISTANT="true", PHOENIX_ENABLE_MCP_SERVER="false",
                       PHOENIX_DEFAULT_RETENTION_POLICY_DAYS="30", PYTHONUTF8="1")
            with self.lock, log_path.open("w", encoding="utf-8") as log:
                if stopping.is_set():
                    raise RuntimeError("Phoenix startup cancelled")
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "praxis_agent.phoenix_service", str(port)],
                    stdin=subprocess.PIPE, stdout=log, stderr=log, env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            deadline = monotonic() + 60
            with httpx.Client(trust_env=False, timeout=1) as client:
                while monotonic() < deadline and not stopping.is_set():
                    if not self.alive:
                        break
                    # Require our child's successful bind log before probing the URL.
                    output = log_path.read_text(encoding="utf-8", errors="replace")
                    if f"Uvicorn running on {url}" in output:
                        try:
                            if client.get(url + "/healthz").is_success and self.alive:
                                return url
                        except httpx.HTTPError:
                            pass
                    stopping.wait(0.1)
            self.close()
            output = log_path.read_text(encoding="utf-8", errors="replace")
            collision = any(word in output.lower() for word in ("address already in use", "10048", "errno 98"))
            if not collision or attempt == 2:
                raise RuntimeError(f"Local Phoenix failed to start; see {log_path}")
        raise RuntimeError("Could not allocate a Phoenix port")

    def close(self):
        with self.lock:
            process, self.process = self.process, None
            if process is None:
                return
            if process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)


def main():
    import _thread
    from threading import Thread

    def watch_owner():
        sys.stdin.buffer.read()
        _thread.interrupt_main()

    port = sys.argv[1]
    sys.argv = ["phoenix", "serve", "--port", port, "--grpc-port", "0"]
    from phoenix.server.main import main as serve
    # Import numerical libraries before blocking on stdin in another thread:
    # Windows native module initialization may acquire the standard-stream locks.
    Thread(target=watch_owner, daemon=True).start()
    serve()


if __name__ == "__main__":
    main()
