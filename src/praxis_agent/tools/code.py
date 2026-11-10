"""Persistent native CPython execution and composition of the current agent's tools."""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from praxis_agent.execution import ExecutionStopped, Run, await_cancellable, current_handle, current_run, run_operation
from praxis_agent.tools.code_output import CodeOutput
from praxis_agent.tools.python_kernel import PythonError, PythonKernel
from praxis_agent.tools.registry import Tool, ToolContext, ToolResult


@dataclass(frozen=True)
class CodeConfig:
    # Wall time per execution, including tool waits; kernel startup is separate.
    execution_seconds: float = 120.0
    idle_seconds: float = 900.0
    queue_seconds: float = 30.0
    max_parallel_tools: int = 16

    def __post_init__(self):
        for value in (self.execution_seconds, self.queue_seconds, self.idle_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Python limits must be finite positive numbers")
        if type(self.max_parallel_tools) is not int or self.max_parallel_tools < 1:
            raise ValueError("max_parallel_tools must be a positive integer")

    @classmethod
    def from_env(cls):
        defaults = cls()

        def value(name, default):
            return os.getenv(name, "").strip() or str(default)

        return cls(
            execution_seconds=float(value("PRAXIS_CODE_EXECUTION_SECONDS", defaults.execution_seconds)),
            idle_seconds=float(value("PRAXIS_CODE_IDLE_SECONDS", defaults.idle_seconds)),
            queue_seconds=float(value("PRAXIS_CODE_QUEUE_SECONDS", defaults.queue_seconds)),
            max_parallel_tools=int(value("PRAXIS_CODE_MAX_PARALLEL_TOOLS", defaults.max_parallel_tools)),
        )


@dataclass(eq=False)
class _AgentWorker:
    """One agent's execution lock and lazily opened CPython kernel."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    released: threading.Event = field(default_factory=threading.Event)
    closed: threading.Event = field(default_factory=threading.Event)
    session: PythonKernel | None = None
    last_used: float = field(default_factory=time.monotonic)

    async def open(self, workspace: Path, python: str) -> None:
        self.session = PythonKernel()
        try:
            await self.session.open(workspace, python)
        except BaseException:
            self.session = None
            raise

    async def close(self) -> None:
        session, self.session = self.session, None
        if session is not None:
            await session.close()


class CodeRuntime:
    """One resident worker per agent, sharing only the host event loop."""

    def __init__(self, config: CodeConfig | None = None, *, output_dir: Path | None = None,
                 environment_dir: Path | Callable[[], Path] | None = None):
        self.config = config or CodeConfig.from_env()
        self._environment_dir = environment_dir
        self._environment_lock = threading.Lock()
        self._python = None
        self._busy_agents: dict[str, int] = {}
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

    def _get_python(self):
        with self._environment_lock:
            if self._python is None:
                if self._environment_dir is None:
                    self._python = sys.executable
                else:
                    import venv
                    import sysconfig
                    import importlib.util

                    # Reuse the shipped kernel libraries; installs through this
                    # interpreter go to the session venv, not Praxis's environment.
                    root = Path(self._environment_dir() if callable(self._environment_dir)
                                else self._environment_dir)
                    # Reuse the runtime's pip too, avoiding ensurepip's costly
                    # per-session bootstrap. pip still targets this venv's prefix.
                    venv.EnvBuilder(with_pip=importlib.util.find_spec("pip") is None,
                                    system_site_packages=False).create(root)
                    packages = root / ("Lib/site-packages" if os.name == "nt" else
                                       f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
                    # venv-of-venv inherits the base installation, not the launching
                    # venv. Explicitly expose Praxis's shipped runtime dependencies.
                    (packages / "praxis-runtime.pth").write_text(
                        f"import site; site.addsitedir({sysconfig.get_path('purelib')!r})\n", encoding="utf-8")
                    self._python = str(root / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
            return self._python

    @contextmanager
    def agent_turn(self, agent_id):
        # Keep variables alive while the model thinks or other tools run.
        with self._lock:
            self._busy_agents[agent_id] = self._busy_agents.get(agent_id, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                self._busy_agents[agent_id] -= 1
                if self._busy_agents[agent_id] == 0:
                    del self._busy_agents[agent_id]
                if worker := self._workers.get(agent_id):
                    worker.last_used = time.monotonic()

    def reap_idle(self):
        # Remove candidates atomically so a new call never attaches to an
        # expiring worker. New work can immediately create a fresh kernel.
        with self._lock:
            if self._closed.is_set():
                return
            active = {worker for worker, _ in self._active.values()}
            expired = [(key, worker) for key, worker in self._workers.items()
                       if key not in self._busy_agents and worker not in active
                       and not worker.released.is_set()
                       and time.monotonic() - worker.last_used >= self.config.idle_seconds]
            for key, worker in expired:
                worker.released.set()
                del self._workers[key]
        for _, worker in expired:
            try:
                asyncio.run_coroutine_threadsafe(worker.close(), self._loop).result()
            finally:
                worker.closed.set()

    def reset_session(self):
        with self._lock:
            agents = tuple(self._workers)
        for agent_id in agents:
            self.release_agent(agent_id)
        with self._environment_lock:
            self._python = None

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
            while not self._stopping.is_set():
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=min(30, self.config.idle_seconds))
                except TimeoutError:
                    await asyncio.to_thread(self.reap_idle)

        try:
            with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if os.name == "nt" else None) as runner:
                runner.run(serve())
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
            if worker.session is not None:
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
                "Execute full Python in this agent's persistent native CPython kernel. "
                "Import project modules and installed packages, and read/write native filesystem paths directly. "
                "Use call_tool(name, arguments) for supported file/history tools and activated MCP tools. "
                "It returns the tool's structured value or text directly; read_file returns a dict with text and paging fields. "
                "Commands, management tools and nested run_code are unavailable through call_tool. "
                "call_parallel_tools([{name, arguments}, ...]) returns a LIST of ordered {success, output/error} entries; "
                "iterate that list directly. Only the outer run_code response wraps its final expression in result. "
                "Each agent has separate variables; agents in one session share the Python dependency environment. "
                "Calls within one agent run sequentially. Ordinary errors preserve completed changes. "
                "Cancellation, execution timeout or process failure kills this kernel and its descendants; "
                "state_lost=true means variables must be rebuilt. External effects are never rolled back. "
                "Idle reclamation and agent close also discard variables; worker_created=true signals fresh state. "
                "Returns the final expression as JSON where possible, otherwise a text preview, plus stdout/stderr. "
                "Stream previews retain first 6000 and last 24000 characters; capped overflow logs are in output_files. "
                "This is native execution without sandbox isolation. See the python skill for usage."
            ),
            parameters={"type": "object", "properties": {
                "code": {"type": "string", "minLength": 1,
                         "description": "Python source. Variables persist; final expression is returned. call_tool and call_parallel_tools are predefined."}},
                "required": ["code"], "additionalProperties": False},
            handler=execute,
        )

    def run(self, context: ToolContext, code: str, *, agent_id: str = "parent") -> ToolResult:
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
                    await worker.open(context.workspace, await asyncio.to_thread(self._get_python))
                    worker_created = True
                try:
                    check()
                    phase = "execute"
                    return await worker.session.execute(code, {
                        "call_tool": call_tool, "call_parallel_tools": call_parallel_tools},
                        capture, self.config.execution_seconds)
                except PythonError:
                    raise
                except BaseException:
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
            detail = str(error) if isinstance(error, PythonError) else f"{type(error).__name__}: {error}"
            kind = "runtime" if phase == "execute" else phase
            if isinstance(error, ExecutionStopped):
                kind = "cancelled"
            elif output.errors:
                kind = "output"
            elif isinstance(error, PythonError):
                if error.name in ("SyntaxError", "IndentationError", "TabError"):
                    kind, phase = "syntax", "preflight"
                elif any(call.get("error") and call["error"] in detail for call in calls):
                    kind = "tool"
            elif isinstance(error, TimeoutError):
                kind = "timeout"
            elif isinstance(error, ChildProcessError):
                kind = "worker_crash"
            return response(False, error=detail, error_kind=kind,
                            error_type=error.name if isinstance(error, PythonError) else type(error).__name__, phase=phase)
        finally:
            output.close()
            unsubscribe()
            if agent_acquired:
                worker.last_used = time.monotonic()
                worker.lock.release()
            with self._lock:
                self._active.pop(scope, None)
                done.set()
