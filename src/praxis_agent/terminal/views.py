"""In-memory agent pages. Producers enqueue; only the UI thread changes views."""

from __future__ import annotations

import json
from praxis_agent.mcp_results import decode_tool_output
from dataclasses import dataclass, replace
from queue import Empty, SimpleQueue
from time import monotonic

from rich.console import Group, RenderableType
from rich.text import Text

from praxis_agent.events import (
    AgentEvent,
    ApprovalRequested,
    ApprovalResolved,
    AssistantTextCompleted,
    AssistantTextDelta,
    ChildAgentEvent,
    ChildAgentMessage,
    ChildAgentStatus,
    ContextCompaction,
    MessagesReceived,
    ModelRequestCompleted,
    ModelRequestStarted,
    PlanUpdated,
    ToolCallPrepared,
    ToolFinished,
    ToolOutputDelta,
    ToolStarted,
    TurnCancelled,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
)
from praxis_agent.planning import PlanStep, PlanStepStatus
from praxis_agent.terminal.cells import (
    CollaborationBlock,
    MessageBlock,
    collaboration_messages,
    plan_cell,
    tool_description,
    tool_details,
    tool_title,
)
from praxis_agent.terminal.command_output import CommandOutputs
from praxis_agent.terminal.context import capacity_footer
from praxis_agent.tools import ToolResult
from praxis_agent.tools.filesystem import FileChange


@dataclass(frozen=True)
class ToolBlock:
    """One tool record, replaced in place as execution progresses."""

    title: str
    arguments: str
    output: str
    state: str = "completed"
    name: str = ""
    detail: str = ""
    file_change: FileChange | None = None
    structured_output: dict[str, object] | None = None

    @property
    def failed(self) -> bool:
        return self.state in {"failed", "spawn_failed", "timed_out"}

    @property
    def exploration(self) -> bool:
        return self.name in {"read_file", "list_files", "find_files", "search_text"}


class AgentView:
    """Reduce one agent's events into display state without terminal I/O."""

    def __init__(self, agent_id: str, label: str) -> None:
        self.agent_id = agent_id
        self.label = label
        self.task = ""
        self.committed_blocks: list[RenderableType | ToolBlock] = []
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.state = "idle"
        self.scroll_position = 0.0
        self.follow_output = True
        self._phase = "Waiting for a task"
        self._capacity: RenderableType | None = None
        self._prepared_calls: dict[str, ToolCallPrepared] = {}
        self._running_tools: dict[str, int] = {}
        self._approval_calls: set[str] = set()
        self._command_output = CommandOutputs()
        self._plan_seen = False
        self._response: MessageBlock | None = None

    @property
    def has_active_tools(self) -> bool:
        return bool(self._running_tools)

    @property
    def phase(self) -> str:
        if self._approval_calls:
            return "Waiting for approval"
        if len(self._running_tools) > 1:
            return f"Running {len(self._running_tools)} tools"
        if self._running_tools:
            name = self.committed_blocks[next(iter(self._running_tools.values()))].name
            return {"run_command": "Running command", "read_file": "Reading file",
                    "edit_file": "Editing file", "write_file": "Writing file"}.get(name, tool_description(name, {})[0])
        return self._phase

    def elapsed(self, now: float | None = None) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, (self.finished_at if self.finished_at is not None
                         else now if now is not None else monotonic()) - self.started_at)

    def consume(self, event: AgentEvent) -> None:
        if isinstance(event, TurnStarted):
            self.started_at, self.finished_at = event.occurred_at, None
            self.state = "running"
            self._running_tools.clear()
            self._prepared_calls.clear()
            self._approval_calls.clear()
            self._command_output.clear()
            self._capacity = None
            self._plan_seen = False
            self._response = None
            self._append_message(event.task, user=True, collaboration=self.agent_id != "main")
            if self.agent_id == "main":
                self.task = event.task.splitlines()[0][:200] if event.task else ""
            self._phase = "Waiting for model response"
        elif isinstance(event, ModelRequestStarted):
            self._response = None
            self._phase = "Waiting for model response"
            self._capacity = capacity_footer(event.context_status)
        elif isinstance(event, AssistantTextDelta):
            if event.text:
                self._phase = "Generating response"
                self._update_response((self._response.text if self._response else "") + event.text,
                                      streaming=True)
        elif isinstance(event, ModelRequestCompleted):
            self._update_response(event.response.output_text if event.response else (
                self._response.text if self._response else ""))
        elif isinstance(event, AssistantTextCompleted):
            self._update_response(event.text)
        elif isinstance(event, ToolCallPrepared):
            self._prepared_calls[event.call_id] = event
        elif isinstance(event, ToolStarted):
            self._start_tool(event.call_id, event.tool_name)
        elif isinstance(event, ToolOutputDelta):
            if event.text and not event.finished:
                index = self._start_tool(event.call_id, "run_command")
                self._command_output.update(event)
                output = self._command_output.outputs[event.call_id]
                block = self.committed_blocks[index]
                self.committed_blocks[index] = replace(
                    block, output=output.text(), detail=(
                        event.command if event.command and event.call_id not in self._prepared_calls else block.detail
                    ),
                )
        elif isinstance(event, ToolFinished):
            self._on_tool_finished(event)
        elif isinstance(event, ApprovalRequested):
            self._approval_calls.add(event.call_id)
        elif isinstance(event, ApprovalResolved):
            self._approval_calls.discard(event.call_id)
        elif isinstance(event, MessagesReceived):
            for item in event.items:
                if isinstance(item, dict) and item.get("role") != "developer" and isinstance(item.get("content"), str):
                    self._append_message(item["content"], user=event.source == "user",
                                         collaboration=event.source != "user")
        elif isinstance(event, PlanUpdated):
            self._append_plan(event.plan)
        elif isinstance(event, ContextCompaction):
            self._on_compaction(event)
        elif isinstance(event, (TurnCompleted, TurnFailed, TurnCancelled)):
            if self._response is not None:
                self._response.update(self._response.text)
            self.finished_at = event.occurred_at
            for index in self._running_tools.values():
                self.committed_blocks[index] = replace(self.committed_blocks[index], state="cancelled")
            self._running_tools.clear()
            self._prepared_calls.clear()
            self._approval_calls.clear()
            self._command_output.clear()
            self.state = "failed" if isinstance(event, TurnFailed) else "ended"
            self._phase = "Execution failed" if self.state == "failed" else "Turn finished"
            if isinstance(event, TurnCancelled):
                self.state = "cancelled"
                self._phase = "Stopped"
                self.committed_blocks.append(Text.assemble(("• ", "red"), ("Turn stopped", "dim")))
            if isinstance(event, TurnFailed):
                self.committed_blocks.append(Text(f"{event.error_type}: {event.message}", style="red"))
            if isinstance(event, TurnCompleted) and self.elapsed() >= 1:
                self.committed_blocks.append(Text(f"─ Turn took {self.elapsed():.0f}s ─", style="dim"))

    def _append_message(self, text: str, *, user: bool, collaboration: bool = False) -> None:
        messages = collaboration_messages(text) if collaboration else None
        if messages is None:
            self.committed_blocks.append(MessageBlock(text, user=user))
        elif self.agent_id != "main":
            self.committed_blocks.extend(MessageBlock(message["content"], user=user) for message in messages)
        # Main-page deliveries were already recorded by ChildAgentMessage.

    def _append_plan(self, plan) -> None:
        complete = all(step.status.value == "completed" for step in plan)
        self.committed_blocks.append(plan_cell(plan, compact=self._plan_seen and not complete))
        self._plan_seen = True

    def restore_history(self, items: list[object]) -> None:
        """Replay saved conversation data into the same display blocks as live events."""
        calls = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            kind, role = item.get("type"), item.get("role")
            if role in {"user", "assistant"}:
                content = item.get("content", "")
                text = content if isinstance(content, str) else "\n".join(
                    part.get("text", "") for part in content if part.get("type") in {"input_text", "output_text"})
                if text:
                    messages = collaboration_messages(text) if role == "user" else None
                    if messages is not None and self.agent_id == "main":
                        self.committed_blocks.extend(CollaborationBlock("Subagent", message["content"])
                                                     for message in messages)
                    else:
                        self._append_message(text, user=role == "user", collaboration=True)
                if role == "user":
                    self._plan_seen = False
            elif kind == "function_call":
                calls[item["call_id"]] = item
            elif kind == "function_call_output" and item["call_id"] in calls:
                call = calls.pop(item["call_id"])
                result = decode_tool_output(item["output"])
                try:
                    arguments = json.loads(call["arguments"])
                except ValueError:
                    arguments = None
                if not isinstance(arguments, dict):
                    # Invalid model arguments are legitimate failed calls, retained verbatim.
                    self.committed_blocks.append(ToolBlock(
                        tool_description(call["name"], {})[0], call["arguments"],
                        result["output"], "failed", call["name"],
                    ))
                    continue
                self._prepared_calls[item["call_id"]] = ToolCallPrepared(
                    "", item["call_id"], call["name"], arguments, False)
                output = result["output"]
                structured = output if isinstance(output, dict) else None
                self._on_tool_finished(ToolFinished(
                    "", item["call_id"], call["name"], ToolResult(
                        result["success"], json.dumps(output, ensure_ascii=False) if structured is not None else output,
                        structured_output=structured)))
                if call["name"] == "update_plan" and result["success"]:
                    self._append_plan([PlanStep(step["step"], PlanStepStatus(step["status"]))
                                       for step in arguments["plan"]])
        self._phase = "Waiting for a task"

    def _on_compaction(self, event: ContextCompaction) -> None:
        if event.stage == "started":
            self.committed_blocks.append(Text(f"· Context exceeds budget; compacting history (≈ {event.before_tokens:,} tokens)", style="dim"))
            self._phase = "Compacting context"
        elif event.stage == "chunk_started":
            phase = "Summary" if event.round_number == 1 else f"Shortening pass {event.round_number - 1}"
            progress = f"{event.chunk_number}/{event.chunk_count}" if event.chunk_count else f"Chunk {event.chunk_number}"
            self._phase = f"Compacting context · {phase} {progress}"
        elif event.stage == "completed":
            self.committed_blocks.append(Text(
                f"• Context compacted · ≈ {event.before_tokens:,} → {event.after_tokens:,} tokens", style="dim",
            ))
            self._phase = "Waiting for model response"
        elif event.stage == "cancelled":
            self.committed_blocks.append(Text("· Context compaction stopped", style="dim"))
            self._phase = "Stopping"
        elif event.stage == "failed":
            self.committed_blocks.append(Text(
                f"× Context compaction failed · {event.elapsed_seconds:.1f}s · {event.error}", style="red",
            ))
            self._phase = "Context compaction failed"

    def _update_response(self, text: str, *, streaming: bool = False) -> None:
        if not text and self._response is None:
            return
        if self._response is None:
            self._response = MessageBlock(text, streaming=streaming)
            self.committed_blocks.append(self._response)
        else:
            self._response.update(text, streaming=streaming)

    def set_status(self, event: ChildAgentStatus) -> None:
        self.task = event.task_summary or self.task
        if event.status == "queued":
            self.started_at = self.finished_at = None
            self.state = "queued"
            self._phase = "Queued"
        elif event.status == "running":
            self.started_at = event.occurred_at
            self.finished_at = None
            self.state = "running"
            self._phase = "Waiting for model response"
        elif event.status in {"idle", "closed"}:
            if self.finished_at is None:
                self.finished_at = event.occurred_at
            if self.state not in {"ended", "failed", "cancelled"}:
                self.state = "ended"
                self._phase = "Closed" if event.status == "closed" else "Turn finished"
        elif event.status == "closing" and self.state not in {"ended", "failed", "cancelled"}:
            self.state = "closing"
            self._phase = "Closing"

    def _start_tool(self, call_id: str, name: str) -> int:
        if call_id not in self._running_tools:
            call = self._prepared_calls.get(call_id)
            arguments = call.arguments if call else {}
            title, detail = tool_description(name, arguments)
            self._running_tools[call_id] = len(self.committed_blocks)
            self.committed_blocks.append(ToolBlock(
                title, json.dumps(arguments, ensure_ascii=False, indent=2), "", "running", name, detail,
            ))
        return self._running_tools[call_id]

    def _on_tool_finished(self, event: ToolFinished) -> None:
        index = self._start_tool(event.call_id, event.tool_name)
        self._command_output.update(ToolOutputDelta(
            event.turn_id, event.call_id, "", "", finished=True,
        ))
        self._prepared_calls.pop(event.call_id, None)
        self._running_tools.pop(event.call_id, None)
        block = self.committed_blocks[index]
        name = block.name
        failed = not event.result.success
        if name == "send_messages" and event.result.success:
            try:
                delivery = json.loads(event.result.output)
            except ValueError:
                delivery = None  # A truncated result must still remain readable.
            if isinstance(delivery, dict):
                failed = any(not item["accepted"] for item in delivery.get("results", []))
        state = "failed" if failed else "completed"
        command_result = event.result.structured_output if name == "run_command" else None
        if command_result is not None:
            state = "completed" if command_result["status"] == "exited" else command_result["status"]
        self.committed_blocks[index] = replace(
            block, output=event.result.output, state=state, structured_output=command_result,
            file_change=event.result.file_change,
        )
        if not self._running_tools:
            self._phase = "Working"

    def display_blocks(self) -> list[RenderableType | tuple[ToolBlock, ...]]:
        """Group adjacent exploration from its start, independent of completion order."""
        blocks = []
        for block in self.committed_blocks:
            if isinstance(block, ToolBlock):
                if block.name == "update_plan" and not block.failed:
                    continue  # PlanUpdated already contains the useful result.
                if block.exploration and blocks and isinstance(blocks[-1], tuple) and blocks[-1][-1].exploration:
                    blocks[-1] += (block,)
                else:
                    blocks.append((block,))
            else:
                blocks.append(block)
        return blocks

    def transcript_content(self) -> RenderableType:
        blocks = []
        for block in self.committed_blocks:
            if isinstance(block, ToolBlock):
                blocks.extend((tool_title((block,)), tool_details((block,))))
            elif isinstance(block, CollaborationBlock):
                blocks.extend((block, Text(block.text)))
            else:
                blocks.append(block)
            blocks.append(Text(""))
        return Group(*blocks)


class ViewStore:
    def __init__(self) -> None:
        self.views = {"main": AgentView("main", "Main agent")}
        self._hidden_agents: set[str] = set()

    @property
    def visible_agent_ids(self) -> list[str]:
        return [agent_id for agent_id in self.views if agent_id not in self._hidden_agents]

    def get(self, agent_id: str) -> AgentView:
        if agent_id not in self.views:
            number = len(self.views)
            self.views[agent_id] = AgentView(agent_id, f"Agent {number}")
        return self.views[agent_id]

    def accept(self, event: AgentEvent) -> str:
        if isinstance(event, TurnStarted):
            # Keep history, but do not carry finished worker tabs into a new task.
            self._hidden_agents.update(
                agent_id for agent_id, view in self.views.items()
                if agent_id != "main" and view.state not in {"queued", "running", "closing"}
            )
        if isinstance(event, ChildAgentEvent):
            agent_id, inner = event.agent_id, event.event
            if isinstance(inner, TurnStarted):
                self._hidden_agents.discard(agent_id)
        elif isinstance(event, ToolOutputDelta):
            agent_id, inner = event.agent_id or "main", event
        elif isinstance(event, ChildAgentStatus):
            if event.status in {"queued", "running"}:
                self._hidden_agents.discard(event.agent_id)
            self.get(event.agent_id).set_status(event)
            return event.agent_id
        elif isinstance(event, ChildAgentMessage):
            child = self.get(event.agent_id)
            # Keep the full report in details without interrupting the root's stream.
            self.views["main"].committed_blocks.append(
                CollaborationBlock(child.label, event.content),
            )
            return "main"
        else:
            agent_id, inner = "main", event
        self.get(agent_id).consume(inner)
        return agent_id


class ViewEventSink:
    """The queue is the sole cross-thread interface; no Rich/Textual work here."""

    def __init__(self) -> None:
        self.events: SimpleQueue[AgentEvent] = SimpleQueue()

    def emit(self, event: AgentEvent) -> None:
        self.events.put(event)

    emit_live = emit

    def emit_deferred(self, event: AgentEvent) -> None:
        # Worker events were already queued through emit_live.
        pass

    def discard_pending(self) -> None:
        """After runtime reset has stopped producers, discard old display events."""
        while True:
            try:
                self.events.get_nowait()
            except Empty:
                return

    def drain(self, store: ViewStore, limit: int = 1000) -> set[str]:
        changed = set()
        for _ in range(limit):
            try:
                event = self.events.get_nowait()
            except Empty:
                break
            changed.add(store.accept(event))
        return changed
