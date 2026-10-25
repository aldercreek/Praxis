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
    while True:
        if agent._closed:
            raise AgentError("this agent is closing")
        if agent.execution_control is not None:
            agent.execution_control.check()
        agent._pump_worker_events()
        agent._receive_messages(turn_id)
        agent._receive_user_input(turn_id)
        tool_definitions = agent.tool_registry.definitions()
        request_number += 1
        transient_items = (build_run_state_context(run_state),)
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
        context_view = run_operation(
            "context",
            lambda handle: agent.context_manager.build_view(
                agent._history,
                transient_items=transient_items,
                tool_definitions=tool_definitions,
            ),
        )
        agent.context_status = context_view.context_status
        agent._emit(
            ModelRequestStarted(
                turn_id,
                request_number,
                context_view.items,
                context_status=context_view.context_status,
            )
        )
        response = run_operation(
            "model",
            lambda handle: agent.model_client.create_response(
                context_view.items,
                tool_definitions,
                on_text_delta=lambda text: agent._emit(
                    AssistantTextDelta(
                        turn_id,
                        request_number,
                        text,
                    )
                ),
            ),
        )
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
            if response.output_text:
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
            raise AgentError("model returned no text and no function calls")

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
