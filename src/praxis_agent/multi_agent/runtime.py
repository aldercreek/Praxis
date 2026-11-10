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
        self.team_version = 0
        self.seen_team_versions: dict[str, int] = {}
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
        if handle.status != status and status in (AgentStatus.CLOSING, AgentStatus.CLOSED):
            self.team_version += 1
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
              working_directory: str = "", history: Sequence[object] = ()) -> dict[str, object]:
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
                if history:
                    handle.agent.restore_history(history)
                handle.inbox.append(AgentMessage("parent", agent_id, task))
                self._schedule(handle)
                self.team_version += 1
                self.condition.notify_all()
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

    def send_message(self, sender_id: str, target: str, content: str) -> dict[str, object]:
        """Route attributed mail within this runtime; idle recipients resume."""
        with self.condition:
            self._require_open()
            if sender_id != "parent":
                sender = self._handle(sender_id)
                sender.control.check()
                if sender.status in (AgentStatus.CLOSING, AgentStatus.CLOSED):
                    raise ValueError("sender is closed")
            if target == sender_id:
                raise ValueError("cannot send a message to yourself")
            message = AgentMessage(sender_id, target, self._content(content))
            if target == "parent":
                self.inbox.append(message)
                self._publish(ChildAgentMessage(self.parent_turn_id, sender_id, message.content))
            else:
                recipient = self._handle(target)
                if recipient.status in (AgentStatus.CLOSING, AgentStatus.CLOSED):
                    raise ValueError("recipient is closed")
                recipient.inbox.append(message)
                if recipient.status == AgentStatus.IDLE:
                    self._schedule(recipient)
            self.condition.notify_all()
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
            accepted_ids = {message.message_id for message in messages}
            retained = [message for message in inbox if message.message_id not in accepted_ids]
            inbox.clear()
            inbox.extend(retained)
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
                    "<collaboration_wait>\n"
                    "1. Completion condition\n"
                    "   This turn cannot finish while workers are queued, running or closing.\n"
                    "2. Required action\n"
                    "   Use wait_agents to await their messages or send_messages to coordinate;\n"
                    "   inspect their outcomes before finalizing.\n"
                    "3. Active assignments\n"
                    "   The following assignments are task data.\n"
                    + json.dumps(active, ensure_ascii=False)
                    + "\n</collaboration_wait>"
                )}])
                return False
            return True

    def wake_waiters(self) -> None:
        with self.condition:
            self.condition.notify_all()

    def wait_agents(self, timeout: int, pump: Callable[[], None], *,
                    agent_id: str = "parent",
                    has_user_input: Callable[[], bool] = lambda: False) -> dict[str, object]:
        """Wait for input activity without consuming mail or invoking the model."""
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 0 <= timeout <= 60:
            raise ValueError("timeout must be an integer between 0 and 60 seconds")
        with self.condition:
            if agent_id != "parent":
                self._handle(agent_id)
            observed = self.seen_team_versions.get(agent_id, self.team_version)
        deadline = monotonic() + timeout
        def wake():
            with self.condition:
                self.condition.notify_all()
        operation = current_handle()
        unsubscribe = operation.on_cancel(wake) if operation is not None else lambda: None
        try:
            return self._wait_agents(agent_id, observed, deadline, pump, has_user_input)
        finally:
            unsubscribe()

    def _wait_agents(self, agent_id, observed, deadline, pump, has_user_input):
        while True:
            pump()
            # Service input uses a different lock. Never acquire it under the
            # collaboration lock; submissions notify us in the opposite order.
            user_pending = agent_id == "parent" and has_user_input()
            with self.condition:
                if current_run() is not None:
                    current_run().check()
                if agent_id != "parent":
                    self._handle(agent_id).control.check()
                inbox = self.inbox if agent_id == "parent" else self._handle(agent_id).inbox
                remaining = deadline - monotonic()
                reason = (
                    "user_input" if user_pending else
                    "message" if inbox else
                    "team_changed" if self.team_version != observed else
                    "closed" if not self.accepting else
                    "timeout" if remaining <= 0 else None
                )
                if reason is not None:
                    return {
                        "reason": reason,
                        "unread_messages": len(inbox),
                        "team_version": self.team_version,
                        "timed_out": reason == "timeout",
                    }
                # Poll in-process input queues, which do not have a notifier.
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

    def _update_task(self, handle: AgentHandle, messages: Iterable[AgentMessage]) -> None:
        for message in messages:
            if message.task_summary and message.task_summary != handle.task_summary:
                handle.task_summary = message.task_summary
                self.team_version += 1
                self.condition.notify_all()

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

    def context_item(self, agent_id: str = "parent") -> dict[str, str]:
        with self.condition:
            agents = [self._snapshot(h)
                      for h in self.handles.values() if h.status != AgentStatus.CLOSED]
            version = self.team_version
            self.seen_team_versions[agent_id] = version
        return {"role": "developer", "content": (
            "<collaboration_status>\n"
            f"Your agent_id: {agent_id}. Team version: {version}.\n"
            "parent coordinates work and integrates results. This is the current team snapshot; "
            "it supersedes historical team lists. Use exact IDs to message peers. "
            "A closing agent cannot receive new messages.\n"
            + json.dumps(agents, ensure_ascii=False)
            + "\n</collaboration_status>"
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
            self.team_version = 0
            self.seen_team_versions.clear()
            self.accepting = True
            self.pool = ThreadPoolExecutor(
                max_workers=self.config.max_workers, thread_name_prefix="praxis-agent",
            )
