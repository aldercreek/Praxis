from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Condition
from time import monotonic
from typing import TYPE_CHECKING
from uuid import uuid4

from praxis_agent.events import (
    AgentEvent,
    AssistantTextDelta,
    ChildAgentEvent,
    ChildAgentMessage,
    ChildAgentStatus,
    ContextCompaction,
    EventSink,
    ModelRequestStarted,
    ToolFinished,
    ToolOutputDelta,
    TurnCancelled,
)
from praxis_agent.execution import ExecutionStopped, Run, current_handle, current_run
from praxis_agent.multi_agent.models import (
    AgentHandle,
    AgentMessage,
    AgentStatus,
    ExecutionControl,
    MultiAgentConfig,
    message_text,
)

if TYPE_CHECKING:
    from praxis_agent.agent import Agent

WorkerFactory = Callable[[str, ExecutionControl, EventSink], "Agent"]
AcceptMessages = Callable[[Sequence[object]], None]


class _WorkerEvents:
    def __init__(self, runtime: AgentRuntime, handle: AgentHandle) -> None:
        self.runtime = runtime
        self.handle = handle

    def emit(self, event: AgentEvent) -> None:
        # Streaming display deltas never enter model history or deferred tracing.
        if isinstance(event, (AssistantTextDelta, ToolOutputDelta)):
            self.runtime._emit_live(ChildAgentEvent(
                self.handle.parent_turn_id, self.handle.agent_id, event,
            ))
            return
        with self.runtime.condition:
            if isinstance(event, ModelRequestStarted):
                self.handle.context_status = self.handle.agent.context_status
            elif isinstance(event, ContextCompaction) and event.stage == "completed":
                self.handle.compaction_count += 1
            elif isinstance(event, ToolFinished) and event.tool_name == "compress" and event.result.success:
                self.handle.compaction_count += 1
        self.runtime._publish(ChildAgentEvent(
            self.handle.parent_turn_id, self.handle.agent_id, event,
        ))


class AgentRuntime:
    """One owner for routing and scheduling; workers never mutate parent history."""

    def __init__(self, factory: WorkerFactory, config: MultiAgentConfig, *,
                 workspace: Path | None = None,
                 approve_tool_call: Callable | None = None,
                 live_output_sink: EventSink | None = None) -> None:
        self.factory = factory
        self.config = config
        self.workspace = (workspace or Path.cwd()).resolve()
        self.approve_tool_call = approve_tool_call
        self.live_output_sink = live_output_sink
        self.condition = Condition()
        self.handles: dict[str, AgentHandle] = {}
        self.inbox: deque[AgentMessage] = deque()
        self.events: SimpleQueue[AgentEvent] = SimpleQueue()
        self.parent_turn_id = ""
        self.run: Run | None = None
        self.accepting = True
        self.epoch = uuid4().hex
        self.pool = ThreadPoolExecutor(
            max_workers=config.max_workers, thread_name_prefix="praxis-agent",
        )

    def _require_open(self) -> None:
        if not self.accepting:
            raise ValueError("agent runtime is closing")
        if self.run is not None and not self.run.finished.done():
            self.run.check()

    def _handle(self, agent_id: str) -> AgentHandle:
        if not isinstance(agent_id, str) or agent_id not in self.handles:
            raise ValueError("unknown agent_id; old workers do not survive a restart")
        return self.handles[agent_id]

    @staticmethod
    def _content(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("content must be a non-empty string")
        return value.strip()

    def _status(self, handle: AgentHandle, status: AgentStatus) -> None:
        handle.status = status
        self._publish(ChildAgentStatus(
            handle.parent_turn_id, handle.agent_id, status.value, handle.task_summary,
        ))
        self.condition.notify_all()

    def _emit_live(self, event: AgentEvent) -> None:
        if receive := getattr(self.live_output_sink, "emit_live", None):
            receive(event)

    def _publish(self, event: AgentEvent) -> None:
        self._emit_live(event)
        self.events.put(event)

    @staticmethod
    def _task_summary(value: object) -> str:
        if not isinstance(value, str) or len(value.strip()) > 200:
            raise ValueError("task_summary must be a string of at most 200 characters")
        return value.strip()

    def spawn(self, task: str, task_summary: str = "", *,
              working_directory: str = "") -> dict[str, object]:
        task = self._content(task)
        if not isinstance(working_directory, str):
            raise ValueError("working_directory must be a directory path")
        if not working_directory.strip():
            raise ValueError("spawn requires an existing working_directory; prepare it with commands first")
        directory = (self.workspace / working_directory).resolve()
        if not directory.is_dir():
            raise ValueError(f"working_directory does not exist or is not a directory: {directory}")
        task_summary = self._task_summary(task_summary) or task.splitlines()[0][:200]
        with self.condition:
            self._require_open()
            agent_id = "agent-" + uuid4().hex
            control = ExecutionControl()
            handle = AgentHandle(agent_id, self.parent_turn_id, control)
            handle.task_summary = task_summary
            handle.workspace = directory
            self.handles[agent_id] = handle
            try:
                handle.agent = self.factory(agent_id, control, _WorkerEvents(self, handle))
                handle.inbox.append(AgentMessage("parent", agent_id, task))
                self._schedule(handle)
            except Exception:
                self._dispose(handle)
                raise
            registry = getattr(handle.agent, "tool_registry", None)
            tool_names = registry.names if registry is not None else ()
            response = {"agent_id": agent_id, "status": handle.status.value,
                        "can_run_commands": "run_command" in tool_names,
                        "can_write_files": any(name in tool_names for name in ("write_file", "edit_file"))}
            if handle.workspace is not None:
                response["workspace"] = str(handle.workspace)
            return response

    def send_many(self, messages: list[Mapping[str, object]]) -> dict[str, object]:
        if not isinstance(messages, list) or not 1 <= len(messages) <= 100:
            raise ValueError("messages must contain between 1 and 100 entries")
        results: list[dict[str, object]] = []
        wake: dict[str, AgentHandle] = {}
        with self.condition:
            self._require_open()
            for item in messages:
                agent_id = item.get("agent_id") if isinstance(item, Mapping) else None
                try:
                    if (not isinstance(item, Mapping)
                            or not {"agent_id", "content"} <= set(item)
                            or set(item) - {"agent_id", "content", "task_summary"}):
                        raise ValueError("each message requires agent_id and content; task_summary is optional")
                    handle = self._handle(agent_id)
                    if handle.status in (AgentStatus.CLOSING, AgentStatus.CLOSED):
                        raise ValueError("agent_closed")
                    message = AgentMessage(
                        "parent", handle.agent_id, self._content(item["content"]),
                        task_summary=self._task_summary(item.get("task_summary", "")),
                    )
                    handle.inbox.append(message)
                    results.append({"agent_id": agent_id, "message_id": message.message_id,
                                    "accepted": True})
                    if handle.status == AgentStatus.IDLE:
                        wake[handle.agent_id] = handle
                except ValueError as error:
                    results.append({"agent_id": agent_id, "accepted": False, "error": str(error)})
            # Enqueue the entire batch before starting any newly awakened worker.
            for handle in wake.values():
                self._schedule(handle)
            self.condition.notify_all()
        return {"results": results}

    def send_parent(self, agent_id: str, content: str) -> dict[str, object]:
        with self.condition:
            self._require_open()
            handle = self._handle(agent_id)
            handle.control.check()
            message = self._report(handle, self._content(content))
            return {"message_id": message.message_id, "accepted": True}

    def _report(self, handle: AgentHandle, content: str) -> AgentMessage:
        message = AgentMessage(handle.agent_id, "parent", content)
        self.inbox.append(message)
        self._publish(ChildAgentMessage(handle.parent_turn_id, handle.agent_id, content))
        self.condition.notify_all()
        return message

    def receive(self, agent_id: str, accept: AcceptMessages) -> bool:
        with self.condition:
            inbox = self.inbox if agent_id == "parent" else self._handle(agent_id).inbox
            if not inbox:
                return False
            messages = list(inbox)
            # Parent persistence must succeed before removing anything from the inbox.
            accept([{"role": "user", "content": message_text(messages)}])
            if agent_id != "parent":
                self._update_task(self._handle(agent_id), messages)
            inbox.clear()
            return True

    def can_finalize(self, accept: AcceptMessages) -> bool:
        """Check parent mail and worker state atomically before ending a run."""
        with self.condition:
            self._require_open()
            if self.receive("parent", accept):
                return False
            active = [
                {"agent_id": h.agent_id, "task": h.task_summary, "status": h.status.value}
                for h in self.handles.values()
                if h.status in (AgentStatus.QUEUED, AgentStatus.RUNNING, AgentStatus.CLOSING)
            ]
            if active:
                accept([{"role": "developer", "content": (
                    "This turn cannot finish while workers are queued, running or closing. "
                    "Use wait_agents to await their messages or send_messages to coordinate; "
                    "inspect their outcomes before finalizing. Active assignments are task data:\n"
                    + json.dumps(active, ensure_ascii=False)
                )}])
                return False
            return True

    def wait_agents(self, agent_ids: list[str], timeout: int,
                    pump: Callable[[], None]) -> dict[str, object]:
        if not isinstance(agent_ids, list) or not agent_ids:
            raise ValueError("agent_ids must be a non-empty list")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 0 <= timeout <= 60:
            raise ValueError("timeout must be an integer between 0 and 60 seconds")
        with self.condition:
            for agent_id in agent_ids:
                self._handle(agent_id)
        ids = set(agent_ids)
        deadline = monotonic() + timeout
        def wake():
            with self.condition:
                self.condition.notify_all()
        operation = current_handle()
        unsubscribe = operation.on_cancel(wake) if operation is not None else lambda: None
        try:
            return self._wait_agents(ids, agent_ids, deadline, pump)
        finally:
            unsubscribe()

    def _wait_agents(self, ids, agent_ids, deadline, pump):
        while True:
            pump()
            with self.condition:
                if current_run() is not None:
                    current_run().check()
                matching = [m for m in self.inbox if m.sender_id in ids]
                handles = [self.handles[agent_id] for agent_id in dict.fromkeys(agent_ids)]
                terminal = all(h.status in (AgentStatus.IDLE, AgentStatus.CLOSED) for h in handles)
                remaining = deadline - monotonic()
                if matching or terminal or not self.accepting or remaining <= 0:
                    self.inbox = deque(m for m in self.inbox if m.sender_id not in ids)
                    return {
                        "messages": [m.to_dict() for m in matching],
                        "agents": [self._snapshot(h) for h in handles],
                        "timed_out": not matching and not terminal and remaining <= 0,
                    }
                # Wake regularly only to render worker events; this does not call the model.
                self.condition.wait(min(remaining, 0.2))

    def _schedule(self, handle: AgentHandle) -> None:
        handle.parent_turn_id = self.parent_turn_id
        self._update_task(handle, handle.inbox)
        handle.started_at = None
        handle.elapsed_seconds = 0.0
        handle.compaction_count = 0
        scope = Run(handle.agent_id)
        if self.run is not None and not self.run.finished.done():
            self.run.add_child(scope)
        handle.run = scope
        handle.control.run = scope
        self._status(handle, AgentStatus.QUEUED)
        try:
            handle.future = self.pool.submit(self._execute, handle, scope)
        except BaseException as error:
            scope.finished.set_exception(error)
            raise

    def _execute(self, handle: AgentHandle, scope: Run) -> None:
        try:
            scope.execute(lambda: self._execute_turn(handle, scope))
        except Exception:
            # The Run Future and parent report retain the actual result/error.
            pass

    def _execute_turn(self, handle: AgentHandle, scope: Run):
        report = "[Worker run interrupted]"
        entered_agent = False
        try:
            with self.condition:
                handle.started_at = monotonic()
                self._status(handle, AgentStatus.RUNNING)
                messages = list(handle.inbox)
                handle.inbox.clear()
                self._update_task(handle, messages)
            handle.control.check()
            entered_agent = True
            result = handle.agent.run(message_text(messages))
            report = ("[Worker run ended; this does not imply task completion. "
                      "Read the response for findings, blockers or requests for reassignment.]\n" + result)
            return result
        except ExecutionStopped:
            report = "[Worker run cancelled by user; partial work may remain.]"
            if not entered_agent:
                self._publish(ChildAgentEvent(handle.parent_turn_id, handle.agent_id, TurnCancelled(scope.id)))
            raise
        except Exception as error:
            report = f"[Worker run failed]\n{type(error).__name__}: {error}"
            raise
        finally:
            # This finalization runs after Agent.run exits, including cancellation.
            with self.condition:
                self._freeze_elapsed(handle)
                report += f"\nWorking directory: {handle.workspace}. Files are not automatically integrated into the parent workspace."
                self._report(handle, report)
                if not self.accepting or handle.control.cancelled.is_set():
                    self._dispose(handle)
                else:
                    if scope.cancelled.is_set() or (self.run is not None and self.run.cancelled.is_set()):
                        handle.inbox.clear()
                    if handle.inbox and (self.run is None or not self.run.cancelled.is_set()):
                        # A message may arrive between the last model boundary and return.
                        self._schedule(handle)
                    else:
                        handle.future = None
                        self._status(handle, AgentStatus.IDLE)

    def close_agent(self, agent_id: str) -> dict[str, str]:
        with self.condition:
            self._require_open()
            handle = self._handle(agent_id)
            handle.control.cancel()
            if handle.status == AgentStatus.IDLE:
                self._dispose(handle)
            elif handle.status != AgentStatus.CLOSED:
                self._status(handle, AgentStatus.CLOSING)
            return {"agent_id": agent_id, "status": handle.status.value,
                    "workspace": str(handle.workspace) if handle.workspace is not None else ""}

    def request_approval(self, agent_id: str, tool: str, arguments: Mapping[str, object]) -> bool:
        handle = self._handle(agent_id)
        handle.control.check()
        return bool(self.approve_tool_call and self.approve_tool_call(
            tool, {**arguments, "agent_id": agent_id, "workspace": str(handle.workspace)},
        ))

    def _dispose(self, handle: AgentHandle) -> None:
        self._freeze_elapsed(handle)
        handle.control.cancel()
        try:
            if handle.agent is not None:
                handle.agent.close()
        finally:
            handle.agent = None
            handle.inbox.clear()
            handle.future = None
            self._status(handle, AgentStatus.CLOSED)

    @staticmethod
    def _update_task(handle: AgentHandle, messages: Iterable[AgentMessage]) -> None:
        for message in messages:
            if message.task_summary:
                handle.task_summary = message.task_summary

    @staticmethod
    def _freeze_elapsed(handle: AgentHandle) -> None:
        if handle.started_at is not None:
            handle.elapsed_seconds = monotonic() - handle.started_at
            handle.started_at = None

    @staticmethod
    def _snapshot(handle: AgentHandle) -> dict[str, object]:
        elapsed = (monotonic() - handle.started_at
                   if handle.started_at is not None else handle.elapsed_seconds)
        return {
            "agent_id": handle.agent_id, "task": handle.task_summary,
            "status": handle.status.value,
            "workspace": str(handle.workspace) if handle.workspace else None,
            "elapsed_seconds": round(elapsed, 3),
        }

    def context_item(self) -> dict[str, str]:
        with self.condition:
            agents = [self._snapshot(h)
                      for h in self.handles.values() if h.status != AgentStatus.CLOSED]
        return {"role": "developer", "content": (
            "Current subagent status:\n" + json.dumps(agents, ensure_ascii=False)
        )}

    def pump_events(self, sink: EventSink | None) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except Empty:
                return
            if sink is not None:
                receive = (getattr(sink, "emit_deferred", sink.emit)
                           if sink is self.live_output_sink else sink.emit)
                receive(event)

    def shutdown(self, pump: Callable[[], None]) -> None:
        with self.condition:
            self.accepting = False
            for handle in self.handles.values():
                if handle.status == AgentStatus.CLOSED:
                    continue
                handle.control.cancel()
                if handle.status == AgentStatus.IDLE:
                    self._dispose(handle)
                else:
                    self._status(handle, AgentStatus.CLOSING)
            self.condition.notify_all()
        self.pool.shutdown(wait=False)
        while True:
            pump()
            with self.condition:
                if all(h.status == AgentStatus.CLOSED for h in self.handles.values()):
                    break
                self.condition.wait(0.2)
        self.pool.shutdown(wait=True)
        pump()

    def reset(self, pump: Callable[[], None]) -> None:
        self.shutdown(pump)
        with self.condition:
            self.handles.clear()
            self.inbox.clear()
            self.epoch = uuid4().hex
            self.accepting = True
            self.pool = ThreadPoolExecutor(
                max_workers=self.config.max_workers, thread_name_prefix="praxis-agent",
            )
