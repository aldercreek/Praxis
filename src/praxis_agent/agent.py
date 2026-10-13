from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING
from uuid import uuid4

from praxis_agent.config import DEFAULT_MAX_PARALLEL_TOOLS, ModelConfig
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
from praxis_agent.skills import discover_skills, create_skill_tool
from praxis_agent.mcp import MCPConnections
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
from praxis_agent.multi_agent.models import ExecutionControl, MultiAgentConfig
from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.multi_agent.tools import (
    PARENT_INSTRUCTIONS, WORKER_INSTRUCTIONS,
    register_parent_tools, register_worker_tools,
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
    execution_control: ExecutionControl | None = None
    close_callbacks: tuple[Callable[[], None], ...] = ()
    mcp_catalog: str = ""
    code_session_context: Callable[[], dict[str, str]] | None = None
    clear_code_sessions: Callable[[], None] | None = None
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
        if not self._run_lock.acquire(blocking=False):
            raise AgentError("this agent is already running")
        try:
            if self._closed:
                raise AgentError("this agent is closed")
            return self._run(task)
        finally:
            self._run_lock.release()

    def _run(self, task: str) -> str:
        normalized_task = task.strip()
        if not normalized_task:
            raise ValueError("task must not be empty")

        turn_id = uuid4().hex
        if self.collaboration is not None:
            self.collaboration.parent_turn_id = turn_id
        self.context_manager.turn_id = turn_id
        self.context_manager.event_sink = self.event_sink
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
        except (Exception, KeyboardInterrupt) as error:
            run_state.fail(error)
            # Interrupted batches must not leave unmatched tool calls on a follow-up.
            try:
                for call in self._pending_calls:
                    item = _build_function_call_output(call, ToolResult(
                        False, "Execution interrupted; completion is unknown. Inspect evidence before retrying.",
                    ))
                    if self.session_store is not None:
                        self.session_store.append_tool_result(
                            self._session_id(), turn_id, item, run_state,
                        )
                    self._history.append(item)
            except Exception as repair_error:
                error.add_note(f"Could not repair interrupted tool results: {repair_error}")
            finally:
                self._pending_calls.clear()
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
        if self.collaboration is not None:
            self.collaboration.reset(self._pump_worker_events)
        if self.session_store is not None:
            session = self.session_store.create_session(
                self.workspace,
                self._session_model(),
            )
            self.session_id = session.id
        if self.clear_code_sessions is not None:
            self.clear_code_sessions()
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
            self.execution_control.cancelled.set()
        try:
            if self.collaboration is not None:
                self.collaboration.shutdown(self._pump_worker_events)
        finally:
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

    def _accept_messages(self, turn_id: str, items: Sequence[object]) -> None:
        if self.session_store is not None:
            self.session_store.append_response_items(self._session_id(), turn_id, items)
        self._history.extend(items)

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
        def check_execution():
            if self._closed:
                raise RuntimeError("this agent is closing")
            if self.execution_control is not None:
                self.execution_control.check()

        tool_executor = ToolCallExecutor(
            tool_registry=self.tool_registry,
            context=ToolContext(
                self.workspace, run_state,
                context_manager=self.context_manager,
                history=self._history,
            ),
            publish_event=self._emit,
            approve_tool_call=self.approve_tool_call,
            bypass_approvals=self.bypass_approvals,
            max_parallel_tools=self.max_parallel_tools,
            check_execution=check_execution,
        )

        request_number = 0
        while True:
            if self._closed:
                raise AgentError("this agent is closing")
            if self.execution_control is not None:
                self.execution_control.check()
            self._pump_worker_events()
            self._receive_messages(turn_id)
            tool_definitions = self.tool_registry.definitions()
            request_number += 1
            transient_items = (build_run_state_context(run_state),)
            if self.code_session_context is not None:
                transient_items += (self.code_session_context(),)
            if self.mcp_catalog:
                transient_items += ({"role": "developer", "content": self.mcp_catalog},)
            if self.collaboration is not None:
                transient_items += (self.collaboration.context_item(),)
            context_view = self.context_manager.build_view(
                self._history,
                transient_items=transient_items,
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
            self._pending_calls = list(response.function_calls)
            self._emit(
                ModelRequestCompleted(
                    turn_id,
                    request_number,
                    response,
                )
            )

            if not response.function_calls:
                if response.output_text:
                    if self.collaboration is not None:
                        if not self.collaboration.can_finalize(
                            lambda items: self._accept_messages(turn_id, items),
                        ):
                            continue
                    elif self._receive_messages(turn_id):
                        continue
                    return response.output_text
                raise AgentError(
                    "model returned no text and no function calls"
                )

            # 组内可并行，但结果按调用顺序交回主线程；保存完本组才执行下一组。
            for function_call, result in tool_executor.execute_many(turn_id, response.function_calls):
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
                self._pending_calls.remove(function_call)
                if (
                    result.success
                    and function_call.name == "update_plan"
                ):
                    self._emit(PlanUpdated(turn_id, tuple(run_state.plan)))

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
    bypass_approvals: bool = False,
    resume_latest: bool = False,
    resume_session_id: str | None = None,
    session_database_path: Path | None = None,
    enable_multi_agent: bool = True,
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
    skills = discover_skills(runtime_context.workspace_root)
    instructions = build_instructions(
        runtime_context, skills=skills,
        collaboration_instructions=PARENT_INSTRUCTIONS if enable_multi_agent else "",
    )
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
        # 恢复的模型可能属于另一档；重新选默认值，同时保留环境中的显式覆盖。
        config = ModelConfig.from_env(model=session.model)

    model_client = OpenAIModelClient(config, instructions=instructions)
    summary_client = OpenAIModelClient(config, instructions=SUMMARY_INSTRUCTIONS, reasoning_effort="none")
    agent = Agent(
        model_client=model_client,
        tool_registry=create_default_registry(config.context_policy),
        workspace=runtime_context.workspace_root,
        event_sink=event_sink,
        approve_tool_call=approve_tool_call,
        bypass_approvals=bypass_approvals,
        max_parallel_tools=config.max_parallel_tools,
        session_store=session_store,
        session_id=session.id,
        session_model=session.model,
        context_manager=_create_context_manager(config, instructions, summary_client),
        close_callbacks=(model_client.close, summary_client.close),
    )

    from praxis_agent.tools.code import CodeRuntime
    code_runtime = CodeRuntime()
    agent.tool_registry.register(code_runtime.tool("parent"))
    agent.code_session_context = lambda: code_runtime.context_item("parent")
    agent.clear_code_sessions = lambda: code_runtime.release_agent("parent")
    agent.close_callbacks = (code_runtime.close, *agent.close_callbacks)
    agent.tool_registry.register(create_skill_tool(skills, max_chars=config.context_policy.max_tool_output_chars))
    mcp = MCPConnections()
    try:
        mcp.start()
        release_mcp_tools = mcp.register(agent.tool_registry)
        agent.mcp_catalog = mcp.catalog()
    except BaseException:
        mcp.close()
        agent.close()
        raise
    agent.close_callbacks = (release_mcp_tools, mcp.close, *agent.close_callbacks)

    if not enable_multi_agent:
        agent.restore_history(history)
        return agent

    def create_worker(agent_id: str, control: ExecutionControl, sink: EventSink) -> Agent:
        handle = collaboration.handles[agent_id]
        # Skill discovery belongs to the session project, not a worker's output directory.
        worker_instructions = build_instructions(
            collect_runtime_context(handle.workspace), skills=skills,
            collaboration_instructions=WORKER_INSTRUCTIONS,
        )
        registry = ToolRegistry()
        for tool in create_default_registry(config.context_policy, check_execution=control.check).tools:
            registry.register(tool)
        registry.register(code_runtime.tool(agent_id))
        registry.register(create_skill_tool(skills, max_chars=config.context_policy.max_tool_output_chars))
        release_worker_tools = mcp.register(registry, check_execution=control.check)
        register_worker_tools(registry, collaboration, agent_id)
        client = OpenAIModelClient(
            config, instructions=worker_instructions, execution_control=control,
        )
        summary = OpenAIModelClient(
            config, instructions=SUMMARY_INSTRUCTIONS, reasoning_effort="none",
            execution_control=control,
        )
        return Agent(
            model_client=client, tool_registry=registry,
            workspace=handle.workspace, name=agent_id,
            mcp_catalog=mcp.catalog(),
            code_session_context=lambda: code_runtime.context_item(agent_id),
            clear_code_sessions=lambda: code_runtime.release_agent(agent_id),
            event_sink=sink, max_parallel_tools=config.max_parallel_tools,
            execution_control=control,
            approve_tool_call=lambda name, arguments: collaboration.request_approval(agent_id, name, arguments),
            bypass_approvals=agent.bypass_approvals,
            context_manager=_create_context_manager(config, worker_instructions, summary),
            receive_messages=lambda accept: collaboration.receive(agent_id, accept),
            close_callbacks=(lambda: code_runtime.release_agent(agent_id), release_worker_tools, client.close, summary.close),
        )

    collaboration = AgentRuntime(create_worker, MultiAgentConfig.from_env(), workspace=runtime_context.workspace_root,
                                 approve_tool_call=approve_tool_call)
    agent.collaboration = collaboration
    agent.receive_messages = lambda accept: collaboration.receive("parent", accept)
    register_parent_tools(agent.tool_registry, collaboration, agent._pump_worker_events)
    agent.restore_history(history)
    return agent


def _create_context_manager(
    config: ModelConfig, instructions: str, summary_client: ModelClient,
) -> ContextManager:
    return ContextManager(
        model_limits=config.model_limits,
        policy=config.context_policy,
        instructions=instructions,
        summarizer=ModelContextSummarizer(
            summary_client,
            model_limits=config.model_limits,
            input_limit_tokens=config.context_policy.effective_input_limit(config.model_limits),
            safety_ratio=config.context_policy.safety_ratio,
        ),
    )
