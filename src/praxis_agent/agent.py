from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from praxis_agent.config import ModelConfig
from praxis_agent.context import (
    ContextManager,
    ContextStatus,
    ConversationHistory,
)
from praxis_agent.context.operations import ContextOperation
from praxis_agent.context.summarizer import ModelContextSummarizer, SUMMARY_INSTRUCTIONS
from praxis_agent.events import (
    AgentEvent,
    AssistantTextDelta,
    AssistantTextCompleted,
    EventSink,
    ModelRequestCompleted,
    ModelRequestStarted,
    PlanUpdated,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
)
from praxis_agent.model import (
    FunctionCall,
    ModelClient,
    OpenAIModelClient,
)
from praxis_agent.planning import RunState, build_run_state_context
from praxis_agent.prompts import build_instructions
from praxis_agent.runtime import collect_runtime_context
from praxis_agent.tool_execution import (
    ToolApprovalHandler,
    ToolCallExecutor,
)
from praxis_agent.tools import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    create_default_registry,
)

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
    max_turns: int = 30
    name: str = "Praxis"
    event_sink: EventSink | None = None
    approve_tool_call: ToolApprovalHandler | None = None
    session_store: SQLiteSessionStore | None = None
    session_id: str | None = None
    session_model: str | None = None
    context_manager: ContextManager = field(default_factory=ContextManager)
    run_state: RunState | None = field(default=None, init=False)
    context_status: ContextStatus | None = field(default=None, init=False)
    _history: ConversationHistory = field(
        default_factory=ConversationHistory,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")

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
        normalized_task = task.strip()
        if not normalized_task:
            raise ValueError("task must not be empty")

        turn_id = uuid4().hex
        run_state = RunState(original_task=normalized_task)
        self.run_state = run_state
        user_item = {"role": "user", "content": normalized_task}
        run_persisted = False

        try:
            if self.session_store is not None:
                self.session_store.start_run(
                    self._session_id(),
                    turn_id,
                    run_state,
                    user_item,
                )
                run_persisted = True
            self._history.append(user_item)
            self._emit(TurnStarted(turn_id, normalized_task))
            output_text = self._run_model_loop(
                turn_id,
                run_state,
            )
            run_state.complete()
            self._finish_persisted_run(turn_id, run_state)
            self._emit(AssistantTextCompleted(turn_id, output_text))
            self._emit(TurnCompleted(turn_id))
            return output_text
        except Exception as error:
            run_state.fail(error)
            if run_persisted:
                try:
                    self._finish_persisted_run(turn_id, run_state)
                except Exception as persistence_error:
                    error.add_note(
                        "Additionally failed to persist the failed Run: "
                        f"{persistence_error}"
                    )
            self._emit(
                TurnFailed(
                    turn_id,
                    type(error).__name__,
                    str(error),
                )
            )
            raise

    def clear_history(self) -> None:
        """保留旧会话，并切换到一段新的空会话。"""
        if self.session_store is not None:
            session = self.session_store.create_session(
                self.workspace,
                self._session_model(),
            )
            self.session_id = session.id
        self._history.clear()
        self.context_manager.reset()
        self.context_manager.operations.clear()
        self.run_state = None
        self.context_status = None

    @property
    def history_item_count(self) -> int:
        """返回当前会话的完整历史条目数量。"""
        return len(self._history)

    def restore_history(self, items: Sequence[object]) -> None:
        """把持久化上下文装载到尚未开始工作的 Agent。"""
        if len(self._history) > 0:
            raise RuntimeError("conversation history is already initialized")
        self._history.extend(items)

    def _run_model_loop(
        self,
        turn_id: str,
        run_state: RunState,
    ) -> str:
        """持续请求模型并执行工具，直到模型生成最终文本。"""
        tool_definitions = self.tool_registry.definitions()
        tool_executor = ToolCallExecutor(
            tool_registry=self.tool_registry,
            context=ToolContext(
                self.workspace, run_state,
                context_manager=self.context_manager,
            ),
            publish_event=self._emit,
            approve_tool_call=self.approve_tool_call,
        )

        for request_number in range(1, self.max_turns + 1):
            context_view = self.context_manager.build_view(
                self._history,
                transient_items=(build_run_state_context(run_state),),
                tool_definitions=tool_definitions,
            )
            self.context_status = context_view.context_status
            self._emit(
                ModelRequestStarted(
                    turn_id,
                    request_number,
                    context_view.items,
                )
            )
            response = self.model_client.create_response(
                context_view.items,
                tool_definitions,
                on_text_delta=lambda text: self._emit(
                    AssistantTextDelta(
                        turn_id,
                        request_number,
                        text,
                    )
                ),
            )
            self.context_manager.record_actual_input_tokens(
                None
                if response.usage is None
                else response.usage.input_tokens
            )
            if self.session_store is not None:
                self.session_store.append_response_items(
                    self._session_id(),
                    turn_id,
                    response.output_items,
                )
            # 官方协议要求把模型本轮输出原样加入下一轮输入。
            self._history.extend(response.output_items)
            self._emit(
                ModelRequestCompleted(
                    turn_id,
                    request_number,
                    response,
                )
            )

            if not response.function_calls:
                if response.output_text:
                    return response.output_text
                raise AgentError(
                    "model returned no text and no function calls"
                )

            # 一次响应可能包含多个工具调用，需要逐个执行并回传结果。
            for function_call in response.function_calls:
                result = tool_executor.execute(turn_id, function_call)
                _record_changed_file(run_state, function_call, result)
                output_item = _build_function_call_output(
                    function_call,
                    result,
                )
                if self.session_store is not None:
                    self.session_store.append_tool_result(
                        self._session_id(),
                        turn_id,
                        output_item,
                        run_state,
                    )
                self._history.append(output_item)
                if (
                    result.success
                    and function_call.name == "update_plan"
                ):
                    self._emit(PlanUpdated(turn_id, tuple(run_state.plan)))

        raise AgentError(
            f"agent exceeded the maximum of {self.max_turns} model turns"
        )

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
        if self.event_sink is not None:
            self.event_sink.emit(event)


def _record_changed_file(
    run_state: RunState,
    function_call: FunctionCall,
    result: ToolResult,
) -> None:
    """根据成功的写文件工具调用更新任务状态。"""
    if (
        not result.success
        or function_call.name not in {"edit_file", "create_file"}
    ):
        return

    arguments = json.loads(function_call.arguments)
    relative_path = arguments.get("relative_path")
    if isinstance(relative_path, str):
        run_state.record_changed_file(relative_path)


def _build_function_call_output(
    function_call: FunctionCall,
    result: ToolResult,
) -> dict[str, object]:
    """把工具结果转换成模型接口要求的输入项。"""
    return {
        "type": "function_call_output",
        "call_id": function_call.call_id,
        "output": json.dumps(
            {
                "success": result.success,
                "output": result.output,
            },
            ensure_ascii=False,
        ),
    }


def create_default_agent(
    workspace: Path,
    *,
    event_sink: EventSink | None = None,
    approve_tool_call: ToolApprovalHandler | None = None,
    resume_latest: bool = False,
    resume_session_id: str | None = None,
    session_database_path: Path | None = None,
) -> Agent:
    """使用环境配置和内置工具创建默认 Agent。"""
    if resume_latest and resume_session_id is not None:
        raise ValueError(
            "resume_latest and resume_session_id are mutually exclusive"
        )

    from praxis_agent.sessions import (
        SQLiteSessionStore,
        SessionNotFoundError,
    )

    config = ModelConfig.from_env()
    runtime_context = collect_runtime_context(workspace)
    instructions = build_instructions(runtime_context)
    session_store = SQLiteSessionStore(session_database_path)
    history: list[object] = []

    if resume_session_id is not None:
        session = session_store.resolve_session(
            resume_session_id,
            workspace=runtime_context.workspace_root,
        )
    elif resume_latest:
        session = session_store.latest_session(runtime_context.workspace_root)
        if session is None:
            raise SessionNotFoundError(
                "no session found for the current workspace"
            )
    else:
        session = session_store.create_session(
            runtime_context.workspace_root,
            config.model,
        )

    if resume_latest or resume_session_id is not None:
        session_store.recover_interrupted_runs(session.id)
        history = session_store.load_conversation_items(session.id)
        config = replace(config, model=session.model)

    agent = Agent(
        model_client=OpenAIModelClient(
            config,
            instructions=instructions,
        ),
        tool_registry=create_default_registry(),
        workspace=runtime_context.workspace_root,
        event_sink=event_sink,
        approve_tool_call=approve_tool_call,
        session_store=session_store,
        session_id=session.id,
        session_model=session.model,
        context_manager=ContextManager(
            model_limits=config.model_limits,
            policy=config.context_policy,
            instructions=instructions,
            summarizer=ModelContextSummarizer(
                OpenAIModelClient(config, instructions=SUMMARY_INSTRUCTIONS),
            ),
        ),
    )
    agent.restore_history(history)
    return agent
