"""Native Jupyter transport and process ownership; no execution sandbox."""
from __future__ import annotations

import asyncio
import contextvars
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from queue import Empty

from praxis_agent.execution import Handle, current_run
from praxis_agent.tools.processes import CommandProcess

MIME = "application/vnd.praxis.result+json"
MAX_MESSAGE = 16 * 1024 * 1024


class PythonError(Exception):
    def __init__(self, name, detail):
        self.name = name
        super().__init__(detail)


class PythonKernel:
    def __init__(self):
        self.process = None
        self.client = None
        self.server = None
        self.directory = None
        self.connections = set()
        self.active = None
        self.token = secrets.token_urlsafe(32)
        self.diagnostics = ""
        self.diagnostic_thread = None

    def _read_diagnostics(self):
        stream = self.process.process.stderr
        try:
            while data := stream.read1(4096):
                self.diagnostics = (self.diagnostics + data.decode("utf-8", errors="replace"))[-8000:]
        finally:
            stream.close()

    @property
    def worker_pid(self):
        return self.process.process.pid if self.process else None

    async def open(self, workspace, python=sys.executable):
        from jupyter_client import AsyncKernelClient
        from jupyter_client.connect import write_connection_file

        process_owner = self

        class NativeKernelClient(AsyncKernelClient):
            async def _async_is_alive(self):
                # We own the real process. An unconnected startup heartbeat is
                # not evidence of death (especially during concurrent startups).
                return process_owner.process.process.poll() is None

        try:
            self.directory = tempfile.TemporaryDirectory(prefix="praxis-kernel-")
            self.server = await asyncio.start_server(self._callback, "127.0.0.1", 0, limit=MAX_MESSAGE)
            address = self.server.sockets[0].getsockname()[:2]
            connection = str(Path(self.directory.name) / "connection.json")
            write_connection_file(connection, ip="127.0.0.1", key=secrets.token_hex(32).encode())
            env = {**os.environ, "PRAXIS_KERNEL_CALLBACK": json.dumps({"address": address, "token": self.token}),
                   "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
            # Bootstrap via absolute path: no editable-install/PYTHONPATH assumptions.
            self.process = CommandProcess(
                [str(python), "-I", "-c", "import runpy,sys; runpy.run_path(sys.argv.pop(1), run_name='__main__')",
                 str(Path(__file__).with_name("kernel_bootstrap.py")), "-f", connection,
                 "--HistoryManager.hist_file=:memory:", "--IPKernelApp.quiet=True"],
                cwd=str(workspace), env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            self.diagnostic_thread = threading.Thread(target=self._read_diagnostics, daemon=True,
                                                       name="praxis-kernel-diagnostics")
            self.diagnostic_thread.start()
            self.client = NativeKernelClient(connection_file=connection)
            self.client.load_connection_file()
            self.client.start_channels(hb=False, stdin=False, control=False)
            await self.client.wait_for_ready(timeout=30)
        except BaseException as error:
            await self.close()
            if isinstance(error, Exception) and self.diagnostics:
                raise RuntimeError(f"Kernel startup failed: {error}\n{self.diagnostics}") from error
            raise

    async def _callback(self, reader, writer):
        self.connections.add(writer)
        try:
            while line := await reader.readline():
                request = json.loads(line)
                active = self.active
                request_id = request.get("request_id")
                try:
                    if (request.get("token") != self.token or active is None
                            or request.get("execution_id") != active["id"]):
                        raise RuntimeError("Execution is no longer active; tool was not dispatched")
                    method = request.get("method")
                    if method not in active["callbacks"]:
                        raise ValueError("Unknown callback method")
                    # Register ownership BEFORE yielding, so cancellation/settle cannot
                    # miss an accepted callback. Propagate the originating Run context.
                    args = request["arguments"]
                    if method == "call_parallel_tools":
                        args = [[{"name": item["name"], "arguments": {"invalid": object()}}
                                 if item.get("encoding_error") else item for item in args[0]]]
                    callback = active["callbacks"][method]
                    owner = active["scope"].add(Handle("python callback"))
                    result = await asyncio.to_thread(
                        active["context"].copy().run, owner.execute, lambda handle: callback(*args),
                    )
                    if self.active is not active:
                        raise RuntimeError("Execution has ended; late tool result discarded")
                    reply = {"request_id": request_id, "success": True, "result": result}
                except Exception as error:
                    reply = {"request_id": request_id, "success": False, "error": str(error)}
                try:
                    data = json.dumps(reply, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
                    if len(data) > MAX_MESSAGE:
                        raise ValueError("Tool response exceeds 16 MiB; request smaller data")
                except (TypeError, ValueError):
                    data = json.dumps({"request_id": request_id, "success": False,
                                       "error": "Tool response cannot be encoded within the 16 MiB limit"}).encode() + b"\n"
                writer.write(data)
                await writer.drain()
                # A persistent connection must not retain the previous Run,
                # output collector, tool result or thousands of finished handles.
                active = callback = args = result = owner = reply = request = None
        except (ConnectionError, ValueError, asyncio.CancelledError):
            pass
        finally:
            self.connections.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass

    async def execute(self, code, callbacks, capture, seconds):
        client = self.client
        execution_id = client.execute(code, store_history=False, allow_stdin=False, stop_on_error=True)
        self.active = {"id": execution_id, "callbacks": callbacks, "scope": current_run(),
                       "context": contextvars.copy_context()}
        result = None

        async def receive():
            nonlocal result
            while True:
                if self.process.process.poll() is not None:
                    raise ChildProcessError("Python kernel exited unexpectedly")
                try:
                    message = await client.get_iopub_msg(timeout=0.1)
                except Empty:
                    continue
                if message.get("parent_header", {}).get("msg_id") != execution_id:
                    continue
                kind, content = message["msg_type"], message["content"]
                if kind == "stream":
                    # Keep draining even if saving an overflow log fails.
                    try:
                        capture(content["name"], content["text"])
                    except (OSError, UnicodeError):
                        pass
                elif kind in ("execute_result", "display_data"):
                    data = content["data"]
                    result = data[MIME]["value"] if MIME in data else data.get("text/plain")
                elif kind == "status" and content["execution_state"] == "idle":
                    break
            # Both channels must complete; idle alone isn't the execution reply.
            while True:
                if self.process.process.poll() is not None:
                    raise ChildProcessError("Python kernel exited before returning its execution reply")
                try:
                    reply = await client.get_shell_msg(timeout=0.1)
                except Empty:
                    continue
                if reply.get("parent_header", {}).get("msg_id") == execution_id:
                    content = reply["content"]
                    if content["status"] == "error":
                        detail = "\n".join(content.get("traceback", [])) or f"{content['ename']}: {content['evalue']}"
                        detail = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", detail)
                        raise PythonError(content["ename"], detail[-30_000:])
                    if content["status"] != "ok":
                        raise ChildProcessError(f"Python execution {content['status']}")
                    return result

        try:
            async with asyncio.timeout(seconds):
                return await receive()
        except TimeoutError as error:
            raise TimeoutError(f"Python execution exceeded {seconds:g} seconds including tool waits") from error
        finally:
            self.active = None

    async def close(self):
        self.active = None
        if self.server is not None:
            self.server.close()
        # Python 3.13+ Server.wait_closed also waits for accepted transports.
        # Close them before awaiting it, including a callback awaiting a host tool.
        for writer in tuple(self.connections):
            writer.close()
        if self.server is not None:
            await self.server.wait_closed()
            self.server = None
        if self.process is not None:
            # Teardown must not queue behind a saturated host-callback executor:
            # a timed-out tool may itself be waiting for cancellation after this.
            self.process.request_stop()
            async with asyncio.timeout(10):
                while self.process.process.poll() is None:
                    await asyncio.sleep(0.01)
                while self.diagnostic_thread is not None and self.diagnostic_thread.is_alive():
                    await asyncio.sleep(0.01)
            self.process.close()
            self.process = None
        if self.client is not None:
            self.client.stop_channels()
            self.client = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None
