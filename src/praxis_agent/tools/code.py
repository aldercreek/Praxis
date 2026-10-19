"""Python composition of the current agent's tools using Monty."""
from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from dataclasses import dataclass

from praxis_agent.execution import ExecutionStopped, Run, await_cancellable, current_handle, current_run
from praxis_agent.tools.registry import Tool, ToolContext, ToolResult


_TOOL_STUBS = """from typing import Any
def call_tool(name: str, arguments: dict[str, Any]) -> Any: ...
"""


@dataclass(frozen=True)
class CodeConfig:
    # Per-feed interpreter compute time; host/tool waiting is excluded.
    compute_seconds: float = 5.0
    memory_bytes: int = 512 * 1024 * 1024
    max_processes: int = 8
    queue_seconds: float = 30.0
    max_suspensions: int = 5000

    def __post_init__(self):
        for value in (self.compute_seconds, self.memory_bytes, self.max_processes, self.queue_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Monty limits must be finite positive numbers")
        if type(self.max_suspensions) is not int or not 0 < self.max_suspensions < 2**64:
            raise ValueError("max_suspensions must be a positive 64-bit integer")

    @classmethod
    def from_env(cls):
        defaults = cls()

        def value(name, default):
            return os.getenv(name, "").strip() or str(default)

        return cls(
            compute_seconds=float(value("PRAXIS_CODE_COMPUTE_SECONDS", defaults.compute_seconds)),
            memory_bytes=int(value("PRAXIS_CODE_MEMORY_BYTES", defaults.memory_bytes)),
            max_processes=int(value("PRAXIS_CODE_MAX_PROCESSES", defaults.max_processes)),
            queue_seconds=float(value("PRAXIS_CODE_QUEUE_SECONDS", defaults.queue_seconds)),
            max_suspensions=int(value("PRAXIS_CODE_MAX_SUSPENSIONS", defaults.max_suspensions)),
        )


class CodeRuntime:
    """Shared Monty pool with private, named, serialized sessions for each agent."""

    def __init__(self, config: CodeConfig | None = None):
        self.config = config or CodeConfig.from_env()
        self._slots = threading.BoundedSemaphore(self.config.max_processes)
        self._lock = threading.Lock()
        self._pool = None
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._stopping = None
        self._startup_error = None
        self._closed = threading.Event()
        self._close_done = threading.Event()
        self._active: dict[Run, threading.Event] = {}
        self._sessions: dict[tuple[str, str], bytes] = {}
        self._agent_locks: dict[str, threading.Lock] = {}

    def _ensure_loop(self):
        with self._lock:
            if self._closed.is_set():
                raise ExecutionStopped("Code runtime is closed")
            if self._thread is None:
                self._thread = threading.Thread(target=self._serve, name="praxis-code", daemon=True)
                self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError("Could not start code runtime") from self._startup_error
        return self._loop

    def _serve(self):
        async def serve():
            from pydantic_monty import AsyncMonty

            async with AsyncMonty(
                max_processes=self.config.max_processes,
                checkout_timeout=self.config.queue_seconds,
                request_timeout=self.config.compute_seconds + 1,
            ) as pool:
                self._pool = pool
                self._loop = asyncio.get_running_loop()
                self._stopping = asyncio.Event()
                self._ready.set()
                await self._stopping.wait()

        try:
            asyncio.run(serve())
        except BaseException as error:
            self._startup_error = error
        finally:
            self._ready.set()

    def close(self):
        with self._lock:
            closing = self._closed.is_set()
            self._closed.set()
            active = tuple(self._active.items())
        if closing:
            self._close_done.wait()
            return
        try:
            for scope, _ in active:
                scope.cancel()
            for _, done in active:
                done.wait()
            if self._thread is not None:
                self._ready.wait()
                if self._loop is not None and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(self._stopping.set)
                self._thread.join()
            with self._lock:
                self._pool = None
                self._sessions.clear()
                self._agent_locks.clear()
        finally:
            self._close_done.set()

    def release_agent(self, agent_id: str) -> None:
        with self._lock:
            for key in [key for key in self._sessions if key[0] == agent_id]:
                del self._sessions[key]
            self._agent_locks.pop(agent_id, None)

    def context_item(self, agent_id: str) -> dict[str, str]:
        with self._lock:
            names = sorted(name for owner, name in self._sessions if owner == agent_id)
        return {"role": "developer", "content": (
            "当前 agent 可用的 run_code 会话：\n"
            + json.dumps(names, ensure_ascii=False)
        )}

    def tool(self, agent_id: str = "parent") -> Tool:
        def execute(context: ToolContext, session_id: str, code: str) -> ToolResult:
            return self.run(context, code, session_id=session_id, agent_id=agent_id)

        return Tool(
            name="run_code",
            description=(
                "Compose allowed tools and process returned data in a private official Monty 1.2.0 Python-subset session. "
                "Built-in json, csv and tomllib can parse text returned by tools. "
                "For project imports/tests, installed libraries or subprocesses use run_command if available; "
                "otherwise report the capability gap. Monty cannot execute host project modules. "
                "Use call_tool for file tools, active MCP tools and history within the current role; "
                "commands, management tools and nested run_code are unavailable. "
                "Tool calls are synchronous; no host filesystem mounts or external modules are configured. "
                "Successful calls save session state. Failure preserves the last saved state but does not "
                "undo external effects. Returns the final expression, stdout, tool-call records and session status; "
                "keep outputs concise. See the monty skill for compatibility and recovery details."
            ),
            parameters={"type": "object", "properties": {
                "session_id": {"type": "string", "minLength": 1, "description": "Meaningful private session name, not a file path. Reuse restores saved variables/functions; a new name starts empty. Saved state survives idle/resume, not agent closure/restart."},
                "code": {"type": "string", "minLength": 1, "description": "Monty-compatible Python, type-checked before execution. Call tools via call_tool(name, arguments). The final expression is returned; print only needed output."}},
                "required": ["session_id", "code"], "additionalProperties": False},
            handler=execute,
        )

    def run(self, context: ToolContext, code: str, *, session_id: str, agent_id: str = "parent") -> ToolResult:
        from pydantic_monty import (
            MontyCrashedError, MontyError, MontyRuntimeError, MontySyntaxError, MontyTypingError,
        )

        if not isinstance(session_id, str) or not session_id.strip():
            return ToolResult(False, json.dumps({
                "error_kind": "input", "error_type": "ValueError", "phase": "input",
                "error": "session_id must be a non-empty meaningful name", "state_saved": False,
            }))
        key = (agent_id, session_id)
        with self._lock:
            agent_lock = self._agent_locks.setdefault(agent_id, threading.Lock())
        agent_acquired = False
        session_created = False
        state_saved = False
        calls = []
        printed = []
        acquired = False
        phase = "queue"
        scope = Run("run_code")
        done = threading.Event()
        unsubscribe = lambda: None
        pending = None

        def check():
            scope.check()
            if self._closed.is_set():
                raise ExecutionStopped("Code runtime is closed")
            if context.check_execution:
                context.check_execution()

        def capture(stream, text):
            printed.append(text)

        def call_tool(name, arguments):
            check()
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise TypeError("call_tool expects a tool name and an arguments object")
            if context.invoke_programmatic_tool is None:
                raise RuntimeError("Programmatic tool executor is unavailable")
            result = context.invoke_programmatic_tool(name, arguments)
            call = {"tool": name, "success": result.success}
            calls.append(call)
            check()
            if not result.success:
                call["error"] = f"Tool {name!r} failed: {result.output}"
                raise RuntimeError(call["error"])
            return result.structured_output if result.structured_output is not None else result.output

        def response(success, **values):
            return ToolResult(success, json.dumps({
                **values, "session_id": session_id, "session_created": session_created,
                "state_saved": state_saved, "stdout": "".join(printed),
                "calls": calls,
                "effects": "Completed operations are not rolled back; do not replay blindly.",
            }, ensure_ascii=False, default=str))

        def execute():
            nonlocal agent_acquired, acquired, session_created, pending
            deadline = time.monotonic() + self.config.queue_seconds
            while not agent_acquired:
                check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for this agent code execution; script not executed")
                agent_acquired = agent_lock.acquire(timeout=min(0.1, remaining))
            with self._lock:
                saved = self._sessions.get(key)
            session_created = saved is None
            while not acquired:
                check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a Monty worker; script not executed")
                acquired = self._slots.acquire(timeout=min(0.1, remaining))
            check()
            loop = self._ensure_loop()

            async def execute_session():
                nonlocal phase
                check()
                async with self._pool.checkout(script_name="praxis_run_code.py", limits={
                    "max_feed_duration_secs": self.config.compute_seconds,
                    "max_memory": self.config.memory_bytes,
                    "max_suspensions": self.config.max_suspensions,
                }, type_check=True, type_check_stubs=_TOOL_STUBS,
                        type_check_format="concise") as session:
                    check()
                    phase = "restore"
                    if saved is not None:
                        await session.load_session(saved)
                    check()
                    phase = "execute"
                    result = await session.feed_run(code, external_lookup={"call_tool": call_tool},
                                                    print_callback=capture)
                    check()
                    phase = "save"
                    state = await session.dump()
                    check()
                    return result, state

            async def owned_call():
                return await await_cancellable(execute_session(), scope)

            # Cancel the owned asyncio task, never this proxy: its completion
            # must acknowledge the session's asynchronous cleanup.
            pending = asyncio.run_coroutine_threadsafe(owned_call(), loop).result()

        try:
            parent = current_run()
            if parent is not None:
                parent.add_child(scope)
            handle = current_handle()
            if handle is not None:
                unsubscribe = handle.on_cancel(scope.cancel)
            with self._lock:
                if self._closed.is_set():
                    scope.cancel()
                self._active[scope] = done
            # Run.settle drains tool callbacks and freezes cancellation before
            # success is chosen. Only that successful outcome may be committed.
            scope.execute(execute)
            result, state = pending
            with self._lock:
                if self._closed.is_set():
                    raise ExecutionStopped("Code runtime is closed")
                self._sessions[key] = state
            state_saved = True
            return response(True, result=result)
        except Exception as error:
            inner = error.exception() if isinstance(error, MontyError) else error
            detail = (error.display() if isinstance(error, (MontyRuntimeError, MontySyntaxError, MontyTypingError))
                      else f"{type(error).__name__}: {error}")
            kind = "runtime" if phase == "execute" else phase
            if isinstance(error, ExecutionStopped):
                kind = "cancelled"
            elif isinstance(error, MontyTypingError):
                kind = "syntax" if "[invalid-syntax]" in detail else "type_check"
                phase = "preflight"
            elif isinstance(error, MontySyntaxError):
                kind, phase = "syntax", "preflight"
            elif isinstance(error, MontyCrashedError):
                kind = "timeout" if error.timed_out else "worker_crash"
            elif isinstance(error, MontyRuntimeError):
                message = str(inner)
                if any(call.get("error") == message for call in calls):
                    kind = "tool"
                elif (isinstance(inner, (MemoryError, TimeoutError))
                      or message.startswith("suspension limit ")):
                    kind = "resource"
            return response(False, error=detail, error_kind=kind,
                            error_type=type(inner).__name__, phase=phase)
        finally:
            unsubscribe()
            if acquired:
                self._slots.release()
            if agent_acquired:
                agent_lock.release()
            with self._lock:
                self._active.pop(scope, None)
                done.set()
