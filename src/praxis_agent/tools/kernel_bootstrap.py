"""Executed inside CPython; tools use a separate, authenticated callback channel."""
from __future__ import annotations

import json
import os
import socket
import threading
from contextvars import ContextVar
from uuid import uuid4

MIME = "application/vnd.praxis.result+json"
MAX_MESSAGE = 16 * 1024 * 1024


def install(shell, address, token):
    connections = threading.local()
    execution_context = ContextVar("praxis_execution", default=None)

    def begin_execution(info):
        execution_context.set(shell.kernel.get_parent()["header"].get("msg_id"))

    def end_execution(result):
        execution_context.set(None)

    shell.events.register("pre_run_cell", begin_execution)
    shell.events.register("post_run_cell", end_execution)

    def rpc(method, arguments):
        # Never use ipykernel's fallback to the most recent shell request: a
        # background thread from an old cell could otherwise impersonate a new one.
        execution_id = execution_context.get()
        if execution_id is None:
            raise RuntimeError("Execution is no longer active in this Python context; use call_parallel_tools for concurrent tools")
        request_id = uuid4().hex
        message = json.dumps({"token": token, "execution_id": execution_id,
                              "request_id": request_id, "method": method,
                              "arguments": arguments}, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        if len(message) > MAX_MESSAGE:
            raise ValueError("Tool request exceeds 16 MiB; pass file paths or smaller data")
        # Flush streams before a callback so cancellation preserves preceding output.
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        if not hasattr(connections, "stream"):
            sock = socket.create_connection(tuple(address), timeout=10)
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connections.socket = sock
            connections.stream = sock.makefile("rwb")
        stream = connections.stream
        try:
            stream.write(message)
            stream.flush()
            line = stream.readline(MAX_MESSAGE + 1)
            if not line or len(line) > MAX_MESSAGE:
                raise ConnectionError("Praxis callback connection closed or response too large")
            reply = json.loads(line)
            if reply.get("request_id") != request_id:
                raise ConnectionError("Mismatched tool reply")
        except BaseException:
            stream.close()
            connections.socket.close()
            del connections.stream
            raise  # Never replay a tool: it may already have changed external state.
        if not reply["success"]:
            raise RuntimeError(reply["error"])
        return reply["result"]

    def call_tool(name, arguments):
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise TypeError("call_tool expects a tool name and an arguments object")
        return rpc("call_tool", [name, arguments])

    def call_parallel_tools(requests):
        if not isinstance(requests, list) or any(
            not isinstance(item, dict) or set(item) != {"name", "arguments"}
            or not isinstance(item["name"], str) or not item["name"].strip()
            or not isinstance(item["arguments"], dict) for item in requests
        ):
            raise TypeError("call_parallel_tools expects a list of {name: str, arguments: dict}")
        encoded = []
        for item in requests:
            try:
                json.dumps(item["arguments"], allow_nan=False)
                encoded.append(item)
            except (TypeError, ValueError, OverflowError):
                # Keep invalid items in order; the host records their failures too.
                encoded.append({"name": item["name"], "encoding_error": True})
        return rpc("call_parallel_tools", [encoded])

    def format_result(value, include=None, exclude=None):
        try:
            serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
            if len(serialized) <= 256_000:
                return {MIME: {"value": json.loads(serialized)}}, {}
        except (TypeError, ValueError, OverflowError, RecursionError):
            pass
        try:
            preview = repr(value)
        except Exception:
            preview = f"<{type(value).__name__}: representation unavailable>"
        if len(preview) > 30_000:
            preview = preview[:6_000] + "\n... [result truncated; save large data to a file] ...\n" + preview[-24_000:]
        return {"text/plain": preview}, {}

    shell.display_formatter.format = format_result
    shell.user_ns.update(call_tool=call_tool, call_parallel_tools=call_parallel_tools)


if __name__ == "__main__":
    import sys
    from ipykernel.kernelapp import IPKernelApp

    endpoint = json.loads(os.environ.pop("PRAXIS_KERNEL_CALLBACK"))
    app = IPKernelApp.instance()
    app.initialize()
    # Bootstrap without project modules shadowing stdlib/kernel dependencies,
    # then expose the workspace for ordinary user imports.
    sys.path.insert(0, os.getcwd())
    install(app.shell, endpoint["address"], endpoint["token"])
    app.start()
