"""Python composition of the current agent's tools using Monty."""
from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from praxis_agent.execution import ExecutionStopped, Run, await_cancellable, current_handle, current_run, run_operation
from praxis_agent.tools.code_output import CodeOutput
from praxis_agent.tools.registry import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from pydantic_monty import AsyncMontySession


_TOOL_STUBS = """from typing import Any
def call_tool(name: str, arguments: dict[str, Any]) -> Any: ...
def call_parallel_tools(requests: list[dict[str, Any]]) -> list[dict[str, Any]]: ...
"""


@dataclass(frozen=True)
class CodeConfig:
    # Per-feed interpreter compute time; host/tool waiting is excluded.
    compute_seconds: float = 5.0
    queue_seconds: float = 30.0
    max_parallel_tools: int = 16

    def __post_init__(self):
        for value in (self.compute_seconds, self.queue_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Monty limits must be finite positive numbers")
        if type(self.max_parallel_tools) is not int or self.max_parallel_tools < 1:
            raise ValueError("max_parallel_tools must be a positive integer")

    @classmethod
    def from_env(cls):
        defaults = cls()

        def value(name, default):
            return os.getenv(name, "").strip() or str(default)

        return cls(
            compute_seconds=float(value("PRAXIS_CODE_COMPUTE_SECONDS", defaults.compute_seconds)),
            queue_seconds=float(value("PRAXIS_CODE_QUEUE_SECONDS", defaults.queue_seconds)),
            max_parallel_tools=int(value("PRAXIS_CODE_MAX_PARALLEL_TOOLS", defaults.max_parallel_tools)),
        )


@dataclass(eq=False)
class _AgentWorker:
    """One agent's execution lock and lazily opened Monty subprocess."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    released: threading.Event = field(default_factory=threading.Event)
    closed: threading.Event = field(default_factory=threading.Event)
    resources: AsyncExitStack | None = None
    session: AsyncMontySession | None = None
    process_handle: int | None = None

    async def open(self, config: CodeConfig) -> None:
        from pydantic_monty import AsyncMonty

        resources = AsyncExitStack()
        try:
            pool = await resources.enter_async_context(AsyncMonty(
                min_processes=0, max_processes=1,
                request_timeout=config.compute_seconds + 1,
            ))
            session = await resources.enter_async_context(pool.checkout(
                script_name="praxis_run_code.py",
                limits={
                    "max_feed_duration_secs": config.compute_seconds,
                    # Omission/None restores Monty's 1000 suspension default.
                    "max_suspensions": 2**64 - 1,
                },
                type_check=True, type_check_stubs=_TOOL_STUBS,
                type_check_format="concise",
            ))
            if os.name == "nt":
                import _winapi

                # Keep the exact process identity even if the SDK drops its handle.
                self.process_handle = _winapi.OpenProcess(0x100000, False, session.worker_pid)
        except BaseException:
            await resources.aclose()
            raise
        self.resources, self.session = resources, session

    async def close(self) -> None:
        resources, self.resources = self.resources, None
        process_handle, self.process_handle = self.process_handle, None
        self.session = None
        try:
            if resources is not None:
                await resources.aclose()
            if process_handle is not None:
                import _winapi

                # Monty's pool exit may precede Windows' process-exit signal.
                # Wait off the shared event loop so other agents can continue.
                status = await asyncio.to_thread(_winapi.WaitForSingleObject, process_handle, 10000)
                if status == _winapi.WAIT_TIMEOUT:
                    raise TimeoutError("Monty worker did not exit after pool cleanup")
        finally:
            if process_handle is not None:
                import _winapi

                _winapi.CloseHandle(process_handle)


class CodeRuntime:
    """One resident worker per agent, sharing only the host event loop."""

    def __init__(self, config: CodeConfig | None = None, *, output_dir: Path | None = None):
        self.config = config or CodeConfig.from_env()
        self._output_dir = (output_dir if output_dir is not None else
                            Path.home() / ".praxis" / "outputs" / "code").resolve()
        self._lock = threading.Lock()
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._stopping = None
        self._startup_error = None
        self._closed = threading.Event()
        self._close_done = threading.Event()
        self._active: dict[Run, tuple[_AgentWorker | None, threading.Event]] = {}
        self._workers: dict[str, _AgentWorker] = {}

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
            for _, (_, done) in active:
                done.wait()
            with self._lock:
                agents = tuple(self._workers)
            for agent_id in agents:
                self.release_agent(agent_id)
            if self._thread is not None:
                self._ready.wait()
                if self._loop is not None and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(self._stopping.set)
                self._thread.join()
        finally:
            self._close_done.set()

    def release_agent(self, agent_id: str) -> None:
        with self._lock:
            worker = self._workers.get(agent_id)
            if worker is None:
                return
            releasing = worker.released.is_set()
            worker.released.set()
            active = [(scope, done) for scope, (owner, done) in self._active.items()
                      if owner is worker]
        if releasing:
            worker.closed.wait()
            return
        try:
            for scope, _ in active:
                scope.cancel()
            for _, done in active:
                done.wait()
            if worker.resources is not None:
                asyncio.run_coroutine_threadsafe(worker.close(), self._loop).result()
        finally:
            with self._lock:
                if self._workers.get(agent_id) is worker:
                    del self._workers[agent_id]
            worker.closed.set()

    def tool(self, agent_id: str = "parent") -> Tool:
        def execute(context: ToolContext, code: str) -> ToolResult:
            return self.run(context, code, agent_id=agent_id)

        return Tool(
            name="run_code",
            description=(
                "Compose allowed tools and process returned data in a private official Monty 1.2.0 Python-subset session. "
                "Built-in json, csv and tomllib can parse text returned by tools. "
                "For project imports/tests, installed libraries or subprocesses use run_command if available; "
                "otherwise report the capability gap. Monty cannot execute host project modules. "
                "Use call_tool for file tools, active MCP tools and history within the current role; "
                "commands, management tools and nested run_code are unavailable. "
                "Use call_parallel_tools([{name, arguments}, ...]) for caller-selected independent calls: "
                "the host runs them concurrently without checking parallel_safe and returns ordered "
                "{success, output} or {success, error} entries; individual failures do not stop the batch. "
                "No host filesystem mounts or external modules are configured. "
                "This agent owns one resident worker: variables/functions persist across calls, including "
                "changes made before ordinary errors. There are no snapshots or rollback. Cancellation during "
                "execution, resource exhaustion or worker failure discards this agent's environment; "
                "state_lost=true means its variables must be rebuilt. External effects are never undone. "
                "Idle agents retain their workers; closing an agent, clearing the conversation or restarting "
                "discards its variables. Returns the final expression, stdout/stderr previews and tool-call records. "
                "Each stream retains 30000 characters (first 6000, last 24000); "
                "overflow logs are saved in full at output_files paths, readable with read_file. "
                "See the monty skill for usage and recovery details."
            ),
            parameters={"type": "object", "properties": {
                "code": {"type": "string", "minLength": 1, "description": "Python for Monty 1.2.0, type-checked before execution. Use call_tool(name, arguments) or call_parallel_tools([{name, arguments}, ...]). The final expression is returned; print only needed output."}},
                "required": ["code"], "additionalProperties": False},
            handler=execute,
        )

    def run(self, context: ToolContext, code: str, *, agent_id: str = "parent") -> ToolResult:
        from pydantic_monty import (
            MontyCrashedError, MontyError, MontyRuntimeError, MontySyntaxError, MontyTypingError,
        )

        worker = None
        agent_acquired = False
        worker_created = False
        state_lost = False
        calls = []
        output = CodeOutput(self._output_dir)
        phase = "queue"
        scope = Run("run_code")
        done = threading.Event()
        unsubscribe = lambda: None
        pending = None

        def check():
            scope.check()
            if self._closed.is_set():
                raise ExecutionStopped("Code runtime is closed")
            if worker is not None and worker.released.is_set():
                raise ExecutionStopped("Agent code worker is closed")
            if context.check_execution:
                context.check_execution()

        def capture(stream, text):
            output.append(stream, text)

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
            output.close()
            return ToolResult(success, json.dumps({
                **values, "worker_created": worker_created, "state_lost": state_lost,
                "calls": calls,
                "effects": "Completed operations are not rolled back; do not replay blindly.",
                **output.fields(),
            }, ensure_ascii=False))

        def execute_parallel_tools(requests):
            check()
            if not isinstance(requests, list) or any(
                not isinstance(item, dict) or set(item) != {"name", "arguments"}
                or not isinstance(item["name"], str) or not item["name"].strip()
                or not isinstance(item["arguments"], dict) for item in requests
            ):
                raise TypeError("call_parallel_tools expects a list of {name: str, arguments: dict}")
            if not requests:
                return []
            if context.invoke_parallel_tools is None:
                raise RuntimeError("Parallel tool executor is unavailable")
            results = context.invoke_parallel_tools(requests, self.config.max_parallel_tools)
            entries = []
            for item, result in zip(requests, results):
                record = {"tool": item["name"], "success": result.success}
                entry = {"success": result.success}
                if result.success:
                    entry["output"] = (result.structured_output if result.structured_output is not None
                                       else result.output)
                else:
                    record["error"] = entry["error"] = result.output
                calls.append(record)
                entries.append(entry)
            check()
            return entries

        def call_parallel_tools(requests):
            # Own the scheduler as well as its tools, so cancellation drains
            # result/event collection before run_code returns.
            return run_operation("parallel tools", lambda handle: execute_parallel_tools(requests))

        def execute():
            nonlocal agent_acquired, pending, phase
            deadline = time.monotonic() + self.config.queue_seconds
            while not agent_acquired:
                check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for this agent code execution; script not executed")
                agent_acquired = worker.lock.acquire(timeout=min(0.1, remaining))
            check()
            loop = self._ensure_loop()

            async def execute_session():
                nonlocal phase, worker_created, state_lost
                check()
                if worker.session is None:
                    phase = "startup"
                    await worker.open(self.config)
                    worker_created = True
                try:
                    check()
                    phase = "execute"
                    return await worker.session.feed_run(code, external_lookup={
                        "call_tool": call_tool, "call_parallel_tools": call_parallel_tools},
                                                         print_callback=capture)
                except (MontySyntaxError, MontyTypingError):
                    raise
                except BaseException as error:
                    inner = error.exception() if isinstance(error, MontyError) else error
                    recoverable = (isinstance(error, MontyRuntimeError)
                                   and not isinstance(inner, (MemoryError, TimeoutError))
                                   and not str(inner).startswith("suspension limit "))
                    if not recoverable:
                        state_lost = True
                        await worker.close()
                    raise

            async def owned_call():
                return await await_cancellable(execute_session(), scope)

            # Cancel the owned asyncio task, never this proxy: its completion
            # must acknowledge the session's asynchronous cleanup.
            result = asyncio.run_coroutine_threadsafe(owned_call(), loop).result()
            check()
            output.close()
            if output.errors:
                raise OSError("Could not save complete code output: " + str(output.errors))
            phase = "serialize"
            pending = response(True, result=result)
            check()

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
                else:
                    worker = self._workers.setdefault(agent_id, _AgentWorker())
                    if worker.released.is_set():
                        scope.cancel()
                self._active[scope] = worker, done
            # Drain host callbacks before releasing this agent's execution lock.
            scope.execute(execute)
            return pending
        except Exception as error:
            output.close()
            inner = error.exception() if isinstance(error, MontyError) else error
            detail = (error.display() if isinstance(error, (MontyRuntimeError, MontySyntaxError, MontyTypingError))
                      else f"{type(error).__name__}: {error}")
            kind = "runtime" if phase == "execute" else phase
            if isinstance(error, ExecutionStopped):
                kind = "cancelled"
            elif output.errors:
                kind = "output"
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
            output.close()
            unsubscribe()
            if agent_acquired:
                worker.lock.release()
            with self._lock:
                self._active.pop(scope, None)
                done.set()
