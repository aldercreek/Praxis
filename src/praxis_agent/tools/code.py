"""Restricted Python composition of the current agent's tools using Monty."""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass

from praxis_agent.tools.registry import Tool, ToolContext, ToolResult


@dataclass(frozen=True)
class CodeConfig:
    compute_seconds: float = 5.0
    memory_bytes: int = 128 * 1024 * 1024
    max_processes: int = 8
    queue_seconds: float = 30.0

    def __post_init__(self):
        for value in (self.compute_seconds, self.memory_bytes, self.max_processes, self.queue_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Monty limits must be finite positive numbers")

    @classmethod
    def from_env(cls):
        return cls(
            compute_seconds=float(os.getenv("PRAXIS_CODE_COMPUTE_SECONDS", "5")),
            memory_bytes=int(os.getenv("PRAXIS_CODE_MEMORY_BYTES", str(128 * 1024 * 1024))),
            max_processes=int(os.getenv("PRAXIS_CODE_MAX_PROCESSES", "8")),
            queue_seconds=float(os.getenv("PRAXIS_CODE_QUEUE_SECONDS", "30")),
        )


class CodeRuntime:
    """One lazy pool shared by a parent and its workers; every call gets fresh state."""

    def __init__(self, config: CodeConfig | None = None):
        self.config = config or CodeConfig.from_env()
        self._slots = threading.BoundedSemaphore(self.config.max_processes)
        self._lock = threading.Lock()
        self._pool = None
        self._closed = threading.Event()

    def _get_pool(self):
        from pydantic_monty import Monty
        with self._lock:
            if self._closed.is_set():
                raise RuntimeError("Code runtime is closed")
            if self._pool is None:
                pool = Monty(
                    max_processes=self.config.max_processes,
                    checkout_timeout=self.config.queue_seconds,
                    request_timeout=self.config.compute_seconds + 1,
                )
                self._pool = pool.__enter__()
            return self._pool

    def close(self):
        self._closed.set()
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.__exit__(None, None, None)

    def tool(self) -> Tool:
        return Tool(
            name="run_code",
            description=(
                "Execute a Python subset in Monty with fresh variables. Use call_tool(name, arguments) "
                "for permitted current-agent file tools, active MCP tools and search_history/read_history. "
                "Original string outputs remain inside the script; failed tools raise exceptions. "
                "Return the final expression; print output is bounded to 65536 characters. "
                "Respect tool schemas and pagination. No host files, network, commands, packages, "
                "management tools or recursion into run_code. Cancellation stops execution. "
                "Earlier effects survive errors: no automatic rollback or replay."
            ),
            parameters={"type": "object", "properties": {
                "code": {"type": "string", "minLength": 1}},
                "required": ["code"], "additionalProperties": False},
            handler=self.run,
        )

    def run(self, context: ToolContext, code: str) -> ToolResult:
        from pydantic_monty import FunctionSnapshot, MontyComplete, NameLookupSnapshot, MontyRuntimeError, MontySyntaxError

        calls = []
        printed = []
        printed_length = 0
        omitted = 0
        acquired = False

        def check():
            if self._closed.is_set():
                raise RuntimeError("Code runtime is closed")
            if context.check_execution:
                context.check_execution()

        def capture(stream, text):
            nonlocal printed_length, omitted
            remaining = max(0, 65536 - printed_length)
            if remaining:
                printed.append(text[:remaining])
            printed_length += min(len(text), remaining)
            omitted += max(0, len(text) - remaining)

        def call_tool(name, arguments):
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise TypeError("call_tool expects a tool name and an arguments object")
            if context.invoke_programmatic_tool is None:
                raise RuntimeError("Programmatic tool executor is unavailable")
            # Only plain JSON arguments cross the existing validation boundary.
            json.dumps(arguments, allow_nan=False)
            result = context.invoke_programmatic_tool(name, arguments)
            calls.append({"tool": name, "success": result.success})
            if not result.success:
                raise RuntimeError(result.output)
            return result.output

        def response(success, **values):
            return ToolResult(success, json.dumps({
                **values, "stdout": "".join(printed), "omitted_print_characters": omitted,
                "calls": calls,
                "effects": "Completed operations are not rolled back; do not replay blindly.",
            }, ensure_ascii=False, default=str))

        try:
            deadline = time.monotonic() + self.config.queue_seconds
            while not acquired:
                check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a Monty worker; script not executed")
                acquired = self._slots.acquire(timeout=min(0.1, remaining))
            check()
            pool = self._get_pool()
            with pool.checkout(script_name="praxis_run_code.py", limits={
                "max_feed_duration_secs": self.config.compute_seconds,
                "max_memory": self.config.memory_bytes,
            }) as session:
                check()
                snapshot = session.feed_start(code, external_lookup={"call_tool": call_tool},
                                              print_callback=capture)
                while not isinstance(snapshot, MontyComplete):
                    check()
                    if isinstance(snapshot, FunctionSnapshot):
                        try:
                            if snapshot.is_os_function or snapshot.function_name != "call_tool":
                                raise PermissionError("Host operation not exposed; use permitted call_tool tools")
                            value = call_tool(*snapshot.args, **snapshot.kwargs)
                            answer = {"return_value": value}
                        except Exception as error:
                            answer = {"exception": RuntimeError(str(error))}
                        # Cancellation is checked outside the catchable sandbox exception path.
                        check()
                        snapshot = snapshot.resume(answer)
                    elif isinstance(snapshot, NameLookupSnapshot):
                        snapshot = snapshot.resume_auto()
                    else:
                        raise RuntimeError("Asynchronous script calls are not supported")
                check()
                return response(True, result=snapshot.output)
        except Exception as error:
            detail = error.display() if isinstance(error, (MontyRuntimeError, MontySyntaxError)) else f"{type(error).__name__}: {error}"
            return response(False, error=detail)
        finally:
            if acquired:
                self._slots.release()
