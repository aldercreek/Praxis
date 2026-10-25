from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Lock
from typing import TYPE_CHECKING

from praxis_agent.config import DEFAULT_MAX_PARALLEL_TOOLS
from praxis_agent.context import (
    ContextManager,
    ContextStatus,
    ConversationHistory,
)
from praxis_agent.context.operations import ContextOperation
from praxis_agent.events import (
    AgentEvent,
    AssistantTextCompleted,
    EventSink,
    MessagesReceived,
    TurnCancelled,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
)
from praxis_agent.execution import ExecutionStopped, Run, current_run
from praxis_agent.model import (
    FunctionCall,
    ModelClient,
)
from praxis_agent.multi_agent.models import ExecutionControl
from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.planning import RunState
from praxis_agent.tool_execution import (
    ToolApprovalHandler,
)
from praxis_agent.tools import (
    ToolRegistry,
    ToolResult,
)
from praxis_agent.tools.command_sessions import CommandSessionManager

if TYPE_CHECKING:
    from collections.abc import Sequence

    from praxis_agent.sessions import SQLiteSessionStore


class AgentError(RuntimeError):
    """表示 Agent 无法正常完成当前任务。"""


@dataclass(slots=True)
class Agent:
    model_client: ModelClient
    tool_registry: ToolRegistry
    workspace: Path
    name: str = "Praxis"
    event_sink: EventSink | None = None
    approve_tool_call: ToolApprovalHandler | None = None
    bypass_approvals: bool = False
    session_store: SQLiteSessionStore | None = None
    session_id: str | None = None
    session_model: str | None = None
    context_manager: ContextManager = field(default_factory=ContextManager)
    max_parallel_tools: int = DEFAULT_MAX_PARALLEL_TOOLS
    collaboration: AgentRuntime | None = None
    receive_messages: Callable[[Callable[[Sequence[object]], None]], bool] | None = None
    user_input: SimpleQueue[str] | None = None
    input_source: object | None = None
    execution_control: ExecutionControl | None = None
    close_callbacks: tuple[Callable[[], None], ...] = ()
    mcp_catalog: str = ""
    reset_code_worker: Callable[[], None] | None = None
    command_sessions: CommandSessionManager = field(default_factory=CommandSessionManager, init=False, repr=False)
    _run_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _closed: bool = field(default=False, init=False)
    _pending_calls: list[FunctionCall] = field(default_factory=list, init=False, repr=False)
    run_state: RunState | None = field(default=None, init=False)
    context_status: ContextStatus | None = field(default=None, init=False)
    _history: ConversationHistory = field(
        default_factory=ConversationHistory,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.max_parallel_tools < 1:
            raise ValueError("max_parallel_tools must be at least 1")
        workspace_root = self.workspace.resolve()
        if not workspace_root.is_dir():
            raise NotADirectoryError(
                f"workspace is not a directory: {self.workspace}"
            )
        self.workspace = workspace_root

        persistence_fields = (self.session_id, self.session_model)
        if self.session_store is None and any(persistence_fields):
            raise ValueError(
                "session_store is required with persisted session metadata"
            )
        if self.session_store is not None and not all(persistence_fields):
            raise ValueError(
                "session_id and session_model are required with session_store"
            )
        if self.session_store is not None:
            self.context_manager.operations.replay(
                self.session_store.load_context_operations(self._session_id())
            )
            self.context_manager.operations.persist = self._persist_context_operation

    def run(self, task: str) -> str:
        """运行一轮任务，并保留内容供同一会话的后续任务使用。"""
        scope = current_run()
        if scope is None:
            return Run(self.name).execute(lambda: self.run(task))
        if not self._run_lock.acquire(blocking=False):
            raise AgentError("this agent is already running")
        try:
            if self._closed:
                raise AgentError("this agent is closed")
            if self.execution_control is None:
                self.execution_control = ExecutionControl()
            self.execution_control.run = scope
            if self.collaboration is not None:
                self.collaboration.run = scope
            return self._run(task)
        finally:
            self._run_lock.release()

    def _run(self, task: str) -> str:
        normalized_task = task.strip()
        if not normalized_task:
            raise ValueError("task must not be empty")

        turn_id = current_run().id
        if self.collaboration is not None:
            self.collaboration.parent_turn_id = turn_id
        self.context_manager.turn_id = turn_id
        self.context_manager.event_sink = self.event_sink
        run_state = RunState(original_task=normalized_task)
        self.run_state = run_state
        user_item = {"role": "user", "content": normalized_task}
        run_persisted = False

        try:
            with self._atomic():
                if self.session_store is not None:
                    self.session_store.start_run(
                        self._session_id(),
                        turn_id,
                        run_state,
                        user_item,
                    )
                    run_persisted = True
                    if self.input_source is not None:
                        self.input_source.consume_initial(turn_id)
                self._emit(TurnStarted(turn_id, normalized_task))
            self._history.append(user_item)
            current_run().check()
            from praxis_agent.agent_loop import run_model_loop
            output_text = run_model_loop(self,
                turn_id,
                run_state,
            )
            current_run().settle()
            run_state.complete()
            with self._atomic():
                self._finish_persisted_run(turn_id, run_state)
                self._emit(AssistantTextCompleted(turn_id, output_text))
                self._emit(TurnCompleted(turn_id))
            return output_text
        except (Exception, KeyboardInterrupt) as error:
            current_run().settle(error)
            self._pump_worker_events()
            if isinstance(error, ExecutionStopped):
                run_state.cancel()
            else:
                run_state.fail(error)
            try:
                self._finish_failed_run(turn_id, run_state, error, run_persisted)
            except Exception as persistence_error:
                persistence_error.add_note(f"While finalizing {type(error).__name__}: {error}")
                error = persistence_error
                run_state.fail(error)
                self._emit(TurnFailed(turn_id, type(error).__name__, str(error)))
            finally:
                self._pending_calls.clear()
            raise error

    def _finish_failed_run(self, turn_id, run_state, error, run_persisted):
        # Repair protocol pairs, save the result and publish its event together.
        with self._atomic():
            for call in self._pending_calls:
                item = _build_function_call_output(call, ToolResult(
                    False, "Execution interrupted; completion is unknown. Inspect evidence before retrying.",
                ))
                if self.session_store is not None:
                    self.session_store.append_tool_result(self._session_id(), turn_id, item, run_state)
                self._history.append(item)
            if run_persisted:
                self._finish_persisted_run(turn_id, run_state)
            self._emit(TurnCancelled(turn_id) if isinstance(error, ExecutionStopped) else
                       TurnFailed(turn_id, type(error).__name__, str(error)))

    def clear_history(self) -> None:
        """保留旧会话，并切换到一段新的空会话。"""
        if self.collaboration is not None:
            self.collaboration.reset(self._pump_worker_events)
        if self.session_store is not None:
            session = self.session_store.create_session(
                self.workspace,
                self._session_model(),
            )
            self.session_id = session.id
        if self.reset_code_worker is not None:
            self.reset_code_worker()
        self.command_sessions.close()
        self.command_sessions = CommandSessionManager()
        self._history.clear()
        self.context_manager.reset()
        self.context_manager.operations.clear()
        self.run_state = None
        self.context_status = None

    def close(self) -> None:
        """Close children before releasing this agent's model clients."""
        if self._closed:
            return
        self._closed = True
        if self.execution_control is not None:
            self.execution_control.cancel()
        try:
            self.command_sessions.close()
            if self.collaboration is not None:
                self.collaboration.shutdown(self._pump_worker_events)
        finally:
            with self._run_lock:
                for close in self.close_callbacks:
                    try:
                        close()
                    except Exception:
                        logging.getLogger(__name__).warning("Could not close a model client", exc_info=True)

    def _pump_worker_events(self) -> None:
        if self.collaboration is not None:
            self.collaboration.pump_events(self.event_sink)

    def _receive_messages(self, turn_id: str) -> bool:
        if self.receive_messages is None:
            return False

        return self.receive_messages(lambda items: self._accept_messages(turn_id, items))

    def _receive_user_input(self, turn_id: str, *, final=False) -> bool:
        if self.input_source is not None:
            return self.input_source.receive(turn_id, lambda items: self._accept_messages(turn_id, items, source="user"), final=final)
        if self.user_input is None:
            return False
        items = []
        while True:
            try:
                text = self.user_input.get_nowait()
            except Empty:
                break
            items.append({"role": "user", "content": text})
        if not items:
            return False
        self._accept_messages(turn_id, items, source="user")
        return True

    def _accept_messages(self, turn_id: str, items: Sequence[object], *, source: str = "agent") -> None:
        if self.session_store is not None:
            self.session_store.append_response_items(self._session_id(), turn_id, items)
        self._history.extend(items)
        if self.event_sink is not None:
            self.event_sink.emit(MessagesReceived(turn_id, tuple(items), source=source))

    @property
    def history_item_count(self) -> int:
        """返回当前会话的完整历史条目数量。"""
        return len(self._history)

    def restore_history(self, items: Sequence[object]) -> None:
        """把持久化上下文装载到尚未开始工作的 Agent。"""
        if len(self._history) > 0:
            raise RuntimeError("conversation history is already initialized")
        self._history.extend(items)

    def _atomic(self):
        return self.session_store.atomic() if self.session_store is not None else nullcontext()

    def _persist_context_operation(self, operation: ContextOperation) -> None:
        if self.session_store is not None:
            self.session_store.append_context_operation(
                self._session_id(), operation,
            )

    def _finish_persisted_run(
        self,
        turn_id: str,
        run_state: RunState,
    ) -> None:
        if self.session_store is not None:
            self.session_store.finish_run(
                self._session_id(),
                turn_id,
                run_state,
            )

    def _session_id(self) -> str:
        if self.session_id is None:
            raise RuntimeError("persisted Agent has no session_id")
        return self.session_id

    def _session_model(self) -> str:
        if self.session_model is None:
            raise RuntimeError("persisted Agent has no session_model")
        return self.session_model

    def _emit(self, event: AgentEvent) -> None:
        """把事件发送给外部消费者；未配置消费者时保持静默。"""
        self._pump_worker_events()
        if self.event_sink is not None:
            self.event_sink.emit(event)


def _build_function_call_output(
    function_call: FunctionCall,
    result: ToolResult,
) -> dict[str, object]:
    """把工具结果转换成模型接口要求的输入项。"""
    if result.model_content is not None:
        return {
            "type": "function_call_output", "call_id": function_call.call_id,
            "output": [{"type": "input_text", "text": json.dumps({"success": result.success})},
                       *result.model_content],
        }
    return {
        "type": "function_call_output",
        "call_id": function_call.call_id,
        "output": json.dumps(
            {
                "success": result.success,
                "output": result.structured_output if function_call.name in {"run_command", "wait_command"} and result.structured_output is not None else result.output,
            },
            ensure_ascii=False,
        ),
    }
