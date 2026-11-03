from __future__ import annotations

import json
from typing import TYPE_CHECKING

from praxis_agent.events import (
    AssistantTextDelta,
    ModelRequestCompleted,
    ModelRequestStarted,
    PlanUpdated,
)
from praxis_agent.execution import run_operation
from praxis_agent.model_errors import IncompleteResponseError, PromptTooLongError, SummaryTooLongError
from praxis_agent.planning import RunState, build_run_state_context
from praxis_agent.tool_execution import (
    ToolCallExecutor,
)
from praxis_agent.tools import (
    ToolContext,
)

if TYPE_CHECKING:
    from praxis_agent.agent import Agent


from praxis_agent.agent import AgentError, _build_function_call_output


def run_model_loop(
    agent: Agent,
    turn_id: str,
    run_state: RunState,
) -> str:
    """持续请求模型并执行工具，直到模型生成最终文本。"""

    def check_execution():
        if agent._closed:
            raise RuntimeError("this agent is closing")
        if agent.execution_control is not None:
            agent.execution_control.check()

    tool_executor = ToolCallExecutor(
        tool_registry=agent.tool_registry,
        context=ToolContext(
            agent.workspace,
            run_state,
            context_manager=agent.context_manager,
            history=agent._history,
            command_sessions=agent.command_sessions,
            defer_command_ack=True,
        ),
        publish_event=agent._emit,
        approve_tool_call=agent.approve_tool_call,
        bypass_approvals=agent.bypass_approvals,
        max_parallel_tools=agent.max_parallel_tools,
        check_execution=check_execution,
    )

    request_number = 0
    turn_start = len(agent._history) - 1
    while True:
        if agent._closed:
            raise AgentError("this agent is closing")
        if agent.execution_control is not None:
            agent.execution_control.check()
        agent._pump_worker_events()
        agent._receive_messages(turn_id)
        agent._receive_user_input(turn_id)
        tool_definitions = agent.tool_registry.definitions()
        run_context = build_run_state_context(run_state)
        transient_items = (run_context,)
        # Select existing user inputs for this turn, including live corrections.
        # Keep the task and progress available in every summary chunk and retry.
        summary_context = tuple(
            item for item in agent._history.snapshot()[turn_start:]
            if isinstance(item, dict) and item.get("role") == "user"
        ) + (run_context,)
        commands = agent.command_sessions.summaries()
        if commands:
            transient_items += (
                {
                    "role": "developer",
                    "content": "Command sessions owned by this agent (use wait_command; do not restart): "
                    + json.dumps(commands, ensure_ascii=False),
                },
            )
        mcp_catalog = agent.mcp_catalog() if agent.mcp_catalog else ""
        if mcp_catalog:
            transient_items += ({"role": "developer", "content": mcp_catalog},)
        if agent.collaboration is not None:
            transient_items += (agent.collaboration.context_item(),)
        recovery_rounds = 0
        recovery_summary = None
        response_retries = 0
        response_hint = ()
        while True:
            request_number += 1
            context_view = run_operation(
                "context",
                lambda handle: agent.context_manager.build_view(
                    agent._history, transient_items=transient_items + response_hint,
                    tool_definitions=tool_definitions,
                    summary_context=summary_context,
                ),
            )
            agent.context_status = context_view.context_status
            agent._emit(ModelRequestStarted(
                turn_id, request_number, context_view.items,
                context_status=context_view.context_status,
            ))
            try:
                response = run_operation(
                    "model",
                    lambda handle: agent.model_client.create_response(
                        context_view.items, tool_definitions,
                        on_text_delta=lambda text: agent._emit(
                            AssistantTextDelta(turn_id, request_number, text)
                        ),
                    ),
                )
                if not response.function_calls and not response.output_text.strip():
                    raise IncompleteResponseError("model returned no text and no function calls")
            except PromptTooLongError as error:
                failure = error
                while recovery_rounds < 2:
                    recovery_rounds += 1
                    try:
                        recovery_summary = run_operation(
                            "context", lambda handle: agent.context_manager.recover_from_ptl(
                                failure, previous_summary=recovery_summary,
                            ),
                        )
                        break
                    except SummaryTooLongError as oversized:
                        # The complete candidate is temporary until it fits. Refining it
                        # uses the next main recovery round, without replaying history.
                        recovery_summary = oversized.summary
                        failure = oversized
                else:
                    raise failure
                continue
            except IncompleteResponseError:
                if response_retries == 2:
                    raise
                response_retries += 1
                response_hint = ({"role": "developer", "content": (
                    "The previous response was incomplete; none of its tool calls were executed. "
                    "Return a complete, concise response with fewer or smaller tool calls. "
                    "For a large file, create a small skeleton and fill it with separate edit_file calls; "
                    "write_file replaces the whole file."
                )},)
                continue
            break
        agent.context_manager.record_actual_input_tokens(
            None if response.usage is None else response.usage.input_tokens
        )
        with agent._atomic():
            if agent.session_store is not None:
                agent.session_store.append_response_items(
                    agent._session_id(),
                    turn_id,
                    response.output_items,
                )
            # 官方协议要求把模型本轮输出原样加入下一轮输入。
            agent._history.extend(response.output_items)
            agent._pending_calls = list(response.function_calls)
            agent._emit(
                ModelRequestCompleted(
                    turn_id,
                    request_number,
                    response,
                )
            )
        check_execution()

        if not response.function_calls:
            if agent.collaboration is not None:
                if not agent.collaboration.can_finalize(
                    lambda items: agent._accept_messages(turn_id, items),
                ):
                    continue
            elif agent._receive_messages(turn_id):
                continue
            if agent._receive_user_input(turn_id, final=True):
                continue
            return response.output_text

        # 组内可并行，但结果按调用顺序交回主线程；保存完本组才执行下一组。
        for function_call, result in tool_executor.execute_many(
            turn_id, response.function_calls
        ):
            output_item = _build_function_call_output(
                function_call,
                result,
            )
            if agent.session_store is not None:
                agent.session_store.append_tool_result(
                    agent._session_id(),
                    turn_id,
                    output_item,
                    run_state,
                )
            agent._history.append(output_item)
            agent._pending_calls.remove(function_call)
            if result.on_saved is not None:
                result.on_saved()
            if result.success and function_call.name == "update_plan":
                agent._emit(PlanUpdated(turn_id, tuple(run_state.plan)))
