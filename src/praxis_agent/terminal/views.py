"""In-memory agent pages. Producers enqueue; only the UI thread changes views."""

from __future__ import annotations

import json
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from time import monotonic

from rich.console import Group, RenderableType
from rich.text import Text

from praxis_agent.events import (
    AgentEvent, ApprovalRequested, ApprovalResolved, AssistantTextCompleted, AssistantTextDelta,
    ChildAgentEvent, ChildAgentMessage, ChildAgentStatus, ContextCompaction,
    MessagesReceived, ModelRequestCompleted, ModelRequestStarted, PlanUpdated, ToolCallPrepared,
    ToolFinished, ToolOutputDelta, ToolStarted, TurnCompleted, TurnFailed, TurnStarted,
)
from praxis_agent.terminal.cells import (
    MessageBlock, plan_cell, tool_activity_cell, tool_description, tool_details, tool_title, working_cell,
)
from praxis_agent.terminal.command_output import CommandOutputPreview
from praxis_agent.terminal.context import CONTEXT_TOOLS, capacity_footer, context_activity


@dataclass(frozen=True)
class ToolBlock:
    """Keep full arguments/results; the widget initially shows a compact title."""

    title: str
    arguments: str
    output: str
    failed: bool = False
    name: str = ""
    detail: str = ""

    @property
    def exploration(self) -> bool:
        return not self.failed and self.name in {"read_file", "list_files", "find_files", "search_text"}


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
        self._phase = "等待任务"
        self._active: RenderableType = ""
        self._capacity: RenderableType | None = None
        self._prepared_calls: dict[str, ToolCallPrepared] = {}
        self._tool_started: dict[str, float] = {}
        self._tool_names: dict[str, str] = {}
        self._approval_calls: set[str] = set()
        self._command_output = CommandOutputPreview()
        self._plan_seen = False
        self._response: MessageBlock | None = None

    @property
    def phase(self) -> str:
        if self._approval_calls:
            return "等待审批"
        if len(self._tool_started) > 1:
            return f"执行 {len(self._tool_started)} 个工具"
        if self._tool_started:
            name = self._tool_names[next(iter(self._tool_started))]
            return {"run_command": "执行命令", "read_file": "读取文件",
                    "edit_file": "编辑文件", "write_file": "写入文件"}.get(name, f"执行 {name}")
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
            self._tool_started.clear()
            self._tool_names.clear()
            self._prepared_calls.clear()
            self._approval_calls.clear()
            self._command_output.clear()
            self._capacity = None
            self._plan_seen = False
            self._response = None
            self._commit(MessageBlock(event.task, user=True))
            self._set_phase("等待模型响应")
            self._active = working_cell()
        elif isinstance(event, ModelRequestStarted):
            self._response = None
            self._set_phase("等待模型响应")
            self._capacity = capacity_footer(event.context_status)
            self._active = working_cell()
        elif isinstance(event, AssistantTextDelta):
            if event.text:
                self._set_phase("正在生成回答")
                self._update_response((self._response.text if self._response else "") + event.text)
        elif isinstance(event, ModelRequestCompleted):
            self._update_response(event.response.output_text if event.response else "")
            self._active = working_cell() if event.response and event.response.function_calls else ""
        elif isinstance(event, AssistantTextCompleted):
            self._update_response(event.text)
            self._active = ""
        elif isinstance(event, ToolCallPrepared):
            self._prepared_calls[event.call_id] = event
        elif isinstance(event, ToolStarted):
            self._tool_started[event.call_id] = event.occurred_at
            self._tool_names[event.call_id] = event.tool_name
            self._restore_active_state()
        elif isinstance(event, ToolOutputDelta):
            self._command_output.update(event)
        elif isinstance(event, ToolFinished):
            self._on_tool_finished(event)
        elif isinstance(event, ApprovalRequested):
            self._approval_calls.add(event.call_id)
        elif isinstance(event, ApprovalResolved):
            self._approval_calls.discard(event.call_id)
            self._restore_active_state()
        elif isinstance(event, MessagesReceived):
            for item in event.items:
                if isinstance(item, dict) and isinstance(item.get("content"), str):
                    if event.source == "user":
                        self._commit(MessageBlock(item["content"], user=True))
                    else:
                        self._commit(Group(Text("收到协作消息", style="cyan"), Text(item["content"])))
        elif isinstance(event, PlanUpdated):
            complete = all(step.status.value == "completed" for step in event.plan)
            self._commit(plan_cell(event.plan, compact=self._plan_seen and not complete))
            self._plan_seen = True
            self._restore_active_state()
        elif isinstance(event, ContextCompaction):
            self._on_compaction(event)
        elif isinstance(event, (TurnCompleted, TurnFailed)):
            self.finished_at = event.occurred_at
            self._tool_started.clear()
            self._approval_calls.clear()
            self._command_output.clear()
            self._active = ""
            self.state = "failed" if isinstance(event, TurnFailed) else "ended"
            self._set_phase("执行出错" if self.state == "failed" else "本轮结束")
            if isinstance(event, TurnFailed):
                self._commit(Text(f"{event.error_type}: {event.message}", style="red"))

    def _on_compaction(self, event: ContextCompaction) -> None:
        if event.stage == "started":
            self._commit(Text(f"· 上下文超出预算，正在整理历史（约 {event.before_tokens:,} tokens）", style="dim"))
            self._set_phase("正在压缩上下文")
            self._active = working_cell()
        elif event.stage == "chunk_started":
            phase = "摘要" if event.round_number == 1 else f"收短第 {event.round_number - 1} 轮"
            progress = f"{event.chunk_number}/{event.chunk_count}" if event.chunk_count else f"第 {event.chunk_number} 块"
            self._set_phase(f"压缩上下文 · {phase} {progress}")
            self._active = working_cell()
        elif event.stage == "completed":
            self._commit(Text(
                f"✓ 上下文已压缩 · 约 {event.before_tokens:,} → {event.after_tokens:,} tokens"
                f" · {event.request_count} 次摘要请求 · {event.elapsed_seconds:.1f}s", style="green",
            ))
            self._set_phase("等待模型响应")
            self._active = working_cell()
        elif event.stage == "failed":
            self._commit(Text(
                f"× 上下文压缩失败 · {event.elapsed_seconds:.1f}s · {event.error}", style="red",
            ))
            self._set_phase("上下文压缩失败")

    def _update_response(self, text: str) -> None:
        if not text:
            return
        if self._response is None:
            self._response = MessageBlock(text)
            self._commit(self._response)
        else:
            self._response.update(text)
        self._active = ""

    def set_status(self, event: ChildAgentStatus) -> None:
        self.task = event.task_summary or self.task
        if event.status == "queued":
            self.started_at = self.finished_at = None
            self.state = "queued"
            self._set_phase("排队中")
        elif event.status == "running":
            self.started_at = event.occurred_at
            self.finished_at = None
            self.state = "running"
            self._set_phase("等待模型响应")
        elif event.status in {"idle", "closed"}:
            if self.finished_at is None:
                self.finished_at = event.occurred_at
            if self.state != "failed":
                self.state = "ended"
                self._set_phase("已关闭" if event.status == "closed" else "本轮结束")
            self._active = ""
        elif event.status == "closing":
            self.state = "closing"
            self._set_phase("正在关闭")

    def _on_tool_finished(self, event: ToolFinished) -> None:
        self._command_output.update(ToolOutputDelta(
            event.turn_id, event.call_id, "", "", finished=True,
        ))
        call = self._prepared_calls.pop(event.call_id, None)
        self._tool_started.pop(event.call_id, None)
        self._tool_names.pop(event.call_id, None)
        arguments = call.arguments if call is not None else {}
        name = call.tool_name if call is not None else event.tool_name
        title, detail = tool_description(name, arguments)
        failed = not event.result.success or (
            name == "run_command" and event.result.output.startswith("exit_code: ")
            and not event.result.output.startswith("exit_code: 0\n")
        )
        self._commit(ToolBlock(
            title, json.dumps(arguments, ensure_ascii=False, indent=2),
            event.result.output, failed, name, detail,
        ))
        if not self._tool_started:
            self._set_phase("处理中")
        self._restore_active_state()

    def _restore_active_state(self) -> None:
        if len(self._tool_started) == 1:
            call_id = next(iter(self._tool_started))
            name = self._tool_names[call_id]
            call = self._prepared_calls.get(call_id)
            arguments = call.arguments if call else {}
            if name in CONTEXT_TOOLS:
                self._active = context_activity(name, arguments)
            else:
                title, detail = tool_description(name, arguments)
                self._active = tool_activity_cell(f"正在{title}", detail)
        elif self._tool_started:
            self._active = tool_activity_cell(self.phase, ", ".join(self._tool_names.values()))
        else:
            self._active = ""

    def _set_phase(self, phase: str) -> None:
        self._phase = phase

    def _commit(self, renderable: RenderableType | ToolBlock) -> None:
        self.committed_blocks.append(renderable)
        self._active = ""

    def active_content(self) -> RenderableType:
        return Group(self._active, *self._command_output.render())

    def display_blocks(self) -> list[RenderableType | tuple[ToolBlock, ...]]:
        """Group adjacent exploration for display while retaining every original result."""
        blocks = []
        for block in self.committed_blocks:
            if isinstance(block, ToolBlock):
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
            else:
                blocks.append(block)
            blocks.append(Text(""))
        return Group(*blocks, self.active_content())


class ViewStore:
    def __init__(self) -> None:
        self.views = {"main": AgentView("main", "主 agent")}

    def get(self, agent_id: str) -> AgentView:
        if agent_id not in self.views:
            number = len(self.views)
            self.views[agent_id] = AgentView(agent_id, f"Agent {number}")
        return self.views[agent_id]

    def accept(self, event: AgentEvent) -> str:
        if isinstance(event, ChildAgentEvent):
            agent_id, inner = event.agent_id, event.event
        elif isinstance(event, ToolOutputDelta):
            agent_id, inner = event.agent_id or "main", event
        elif isinstance(event, ChildAgentStatus):
            self.get(event.agent_id).set_status(event)
            return event.agent_id
        elif isinstance(event, ChildAgentMessage):
            child = self.get(event.agent_id)
            # The full message appears when accepted, or in wait_agents' result.
            # A background notification must not clear the root's active stream.
            self.views["main"].committed_blocks.append(
                Text(f"{child.label} → 主 agent：收到新消息", style="cyan"),
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
