from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Empty, SimpleQueue
from threading import Condition
from time import monotonic
from typing import TYPE_CHECKING
from uuid import uuid4
from pathlib import Path


from praxis_agent.events import (
    AgentEvent, AssistantTextDelta, ChildAgentEvent, ChildAgentStatus, ChildAgentMessage, EventSink,
    ContextCompaction, ModelRequestStarted, ToolFinished,
)
from praxis_agent.multi_agent.models import (
    AgentHandle, AgentMessage, AgentStatus, ExecutionControl,
    MultiAgentConfig, message_text,
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
        # Streaming text is private and needs no coordination lock or queue entry.
        if isinstance(event, AssistantTextDelta):
            return
        with self.runtime.condition:
            if isinstance(event, ModelRequestStarted):
                self.handle.context_status = self.handle.agent.context_status
            elif isinstance(event, ContextCompaction) and event.stage == "completed":
                self.handle.compaction_count += 1
            elif isinstance(event, ToolFinished) and event.tool_name == "compress" and event.result.success:
                self.handle.compaction_count += 1
        self.runtime.events.put(ChildAgentEvent(
            self.handle.parent_turn_id, self.handle.agent_id, event,
        ))


class AgentRuntime:
    """One owner for routing and scheduling; workers never mutate parent history."""

    def __init__(self, factory: WorkerFactory, config: MultiAgentConfig, *,
                 workspace: Path | None = None,
                 approve_tool_call: Callable | None = None) -> None:
        self.factory = factory
        self.config = config
        self.workspace = (workspace or Path.cwd()).resolve()
        self.approve_tool_call = approve_tool_call
        self.approvals: SimpleQueue = SimpleQueue()
        self.condition = Condition()
        self.handles: dict[str, AgentHandle] = {}
        self.inbox: deque[AgentMessage] = deque()
        self.events: SimpleQueue[AgentEvent] = SimpleQueue()
        self.parent_turn_id = ""
        self.accepting = True
        self.epoch = uuid4().hex
        self.pool = ThreadPoolExecutor(
            max_workers=config.max_workers, thread_name_prefix="praxis-agent",
        )

    def _require_open(self) -> None:
        if not self.accepting:
            raise ValueError("agent runtime is closing")

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
        self.events.put(ChildAgentStatus(
            handle.parent_turn_id, handle.agent_id, status.value,
        ))
        self.condition.notify_all()

    @staticmethod
    def _task_summary(value: object) -> str:
        if not isinstance(value, str) or len(value.strip()) > 200:
            raise ValueError("task_summary must be a string of at most 200 characters")
        return value.strip()

    def spawn(self, task: str, task_summary: str = "", mode: str = "read_only",
              working_directory: str = "") -> dict[str, str]:
        task = self._content(task)
        if mode not in ("read_only", "execute"):
            raise ValueError("mode must be read_only or execute")
        if not isinstance(working_directory, str):
            raise ValueError("working_directory must be a directory path")
        if mode == "execute" and not working_directory.strip():
            raise ValueError("execute requires an existing working_directory; prepare it with commands first")
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
            handle.mode = mode
            handle.workspace = directory
            self.handles[agent_id] = handle
            try:
                handle.agent = self.factory(agent_id, control, _WorkerEvents(self, handle))
                handle.inbox.append(AgentMessage("parent", agent_id, task))
                self._schedule(handle)
            except Exception:
                self._dispose(handle)
                raise
            response = {"agent_id": agent_id, "status": handle.status.value, "mode": mode}
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
        self.events.put(ChildAgentMessage(handle.parent_turn_id, handle.agent_id, content))
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
        while True:
            pump()
            with self.condition:
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
        self._status(handle, AgentStatus.QUEUED)
        handle.future = self.pool.submit(self._execute, handle)

    def _execute(self, handle: AgentHandle) -> None:
        with self.condition:
            if not self.accepting or handle.control.cancelled.is_set():
                self._dispose(handle)
                return
            handle.started_at = monotonic()
            self._status(handle, AgentStatus.RUNNING)
            messages = list(handle.inbox)
            handle.inbox.clear()
            self._update_task(handle, messages)
        report = "[Worker run interrupted]"
        try:
            handle.control.check()
            result = handle.agent.run(message_text(messages))
            report = ("[Worker run ended; this does not imply task completion. "
                      "Read the response for findings, blockers or requests for reassignment.]\n" + result)
        except Exception as error:
            report = f"[Worker run failed]\n{type(error).__name__}: {error}"
        finally:
            # This finalization runs after Agent.run exits, including cancellation.
            with self.condition:
                self._freeze_elapsed(handle)
                if handle.mode == "execute":
                    report += f"\nWorking directory: {handle.workspace}. Files are not automatically integrated into the parent workspace."
                self._report(handle, report)
                if not self.accepting or handle.control.cancelled.is_set():
                    self._dispose(handle)
                else:
                    if handle.inbox:
                        # A message may arrive between the last model boundary and return.
                        self._schedule(handle)
                    else:
                        handle.future = None
                        self._status(handle, AgentStatus.IDLE)

    def close_agent(self, agent_id: str) -> dict[str, str]:
        with self.condition:
            self._require_open()
            handle = self._handle(agent_id)
            handle.control.cancelled.set()
            if handle.status == AgentStatus.IDLE or (
                handle.status == AgentStatus.QUEUED and handle.future.cancel()
            ):
                self._dispose(handle)
            elif handle.status != AgentStatus.CLOSED:
                self._status(handle, AgentStatus.CLOSING)
            return {"agent_id": agent_id, "status": handle.status.value,
                    "workspace": str(handle.workspace) if handle.workspace is not None else ""}

    def request_approval(self, agent_id: str, tool: str, arguments: Mapping[str, object]) -> bool:
        handle = self._handle(agent_id)
        future: Future = Future()
        self.approvals.put((handle, tool, dict(arguments), future))
        # Only the parent thread invokes the interactive approval callback.
        while not future.done():
            handle.control.check()
            handle.control.cancelled.wait(0.1)
        return future.result()

    def _dispose(self, handle: AgentHandle) -> None:
        self._freeze_elapsed(handle)
        handle.control.cancelled.set()
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
        status = handle.context_status
        elapsed = (monotonic() - handle.started_at
                   if handle.started_at is not None else handle.elapsed_seconds)
        return {
            "agent_id": handle.agent_id, "task": handle.task_summary,
            "status": handle.status.value,
            "mode": handle.mode, "workspace": str(handle.workspace) if handle.workspace else None,
            "context": None if status is None else {
                "estimated_input_tokens": status.estimated_input_tokens,
                "effective_input_limit_tokens": status.effective_input_limit_tokens,
                "usage_percent": status.usage_percent,
            },
            "compaction_count": handle.compaction_count,
            "elapsed_seconds": round(elapsed, 3),
        }

    def context_item(self) -> dict[str, str]:
        with self.condition:
            agents = [self._snapshot(h)
                      for h in self.handles.values() if h.status != AgentStatus.CLOSED]
        return {"role": "developer", "content": (
            "Current in-memory collaboration runtime: " + self.epoch + ". "
            "Only the following worker IDs are live. Workers from earlier processes "
            "do not survive restart; use saved reports to create new tasks when needed. "
            "Idle means the run ended, not necessarily the task; inspect its messages. "
            "Use send_messages to resume with retained context. Task labels are assignment data. "
            "Context is the latest pre-request estimate (null until known), retained across runs. "
            "Compaction count and elapsed seconds cover the current/last run, reset on rescheduling; "
            "time excludes queue/idle time. These are signals for coordination, not automatic stop rules.\n"
            + json.dumps(agents, ensure_ascii=False)
        )}

    def pump_events(self, sink: EventSink | None) -> None:
        while True:
            try:
                handle, tool, arguments, future = self.approvals.get_nowait()
            except Empty:
                break
            try:
                approved = (not handle.control.cancelled.is_set()
                            and self.approve_tool_call is not None
                            and self.approve_tool_call(tool, {**arguments,
                                "agent_id": handle.agent_id, "workspace": str(handle.workspace)}))
                future.set_result(bool(approved))
            except Exception as error:
                future.set_exception(error)
        while True:
            try:
                event = self.events.get_nowait()
            except Empty:
                return
            if sink is not None:
                sink.emit(event)

    def shutdown(self, pump: Callable[[], None]) -> None:
        with self.condition:
            self.accepting = False
            for handle in self.handles.values():
                if handle.status == AgentStatus.CLOSED:
                    continue
                handle.control.cancelled.set()
                if handle.status == AgentStatus.IDLE or (
                    handle.status == AgentStatus.QUEUED and handle.future.cancel()
                ):
                    self._dispose(handle)
                else:
                    self._status(handle, AgentStatus.CLOSING)
            self.condition.notify_all()
        self.pool.shutdown(wait=False, cancel_futures=True)
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
