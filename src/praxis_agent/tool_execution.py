import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from praxis_agent.events import (
    AgentEvent,
    ApprovalRequested,
    ApprovalResolved,
    ToolCallPrepared,
    ToolFinished,
    ToolStarted,
)
from praxis_agent.model import FunctionCall
from praxis_agent.tools import ToolContext, ToolRegistry, ToolResult


ToolApprovalHandler = Callable[[str, Mapping[str, object]], bool]
EventPublisher = Callable[[AgentEvent], None]


class _InvalidToolArgumentsError(ValueError):
    """表示模型生成的工具参数无法用于工具调用。"""


def _parse_arguments(encoded_arguments: str) -> dict[str, object]:
    """把模型生成的 JSON 参数解析成关键字参数字典。"""
    try:
        arguments = json.loads(encoded_arguments)
    except json.JSONDecodeError as error:
        raise _InvalidToolArgumentsError(
            f"invalid JSON arguments: {error.msg} "
            f"（第 {error.lineno} 行，第 {error.colno} 列，字符位置 {error.pos + 1}）；请修正 JSON 格式。"
        ) from error

    if not isinstance(arguments, dict):
        raise _InvalidToolArgumentsError(
            "tool arguments must be a JSON object"
        )
    return arguments


@dataclass(slots=True)
class ToolCallExecutor:
    """处理一次完整的工具调用，包括审批和工具生命周期事件。"""

    tool_registry: ToolRegistry
    context: ToolContext
    publish_event: EventPublisher
    approve_tool_call: ToolApprovalHandler | None = None
    bypass_approvals: bool = False

    def execute(
        self,
        turn_id: str,
        function_call: FunctionCall,
    ) -> ToolResult:
        """执行模型请求的工具调用，并返回统一的工具结果。"""
        try:
            arguments = _parse_arguments(function_call.arguments)
        except _InvalidToolArgumentsError as error:
            return self._finish(
                turn_id,
                function_call,
                self.tool_registry.argument_error(function_call.name, str(error)),
            )

        requires_approval = (
            not self.bypass_approvals
            and self.tool_registry.requires_approval(function_call.name)
        )
        self.publish_event(
            ToolCallPrepared(
                turn_id,
                function_call.call_id,
                function_call.name,
                dict(arguments),
                requires_approval,
            )
        )

        if requires_approval and not self._request_approval(
            turn_id,
            function_call,
            arguments,
        ):
            return self._finish(
                turn_id,
                function_call,
                self._approval_failure(),
            )

        self.publish_event(
            ToolStarted(
                turn_id,
                function_call.call_id,
                function_call.name,
            )
        )
        result = self.tool_registry.execute(
            function_call.name,
            arguments,
            context=self.context,
        )
        return self._finish(turn_id, function_call, result)

    def _request_approval(
        self,
        turn_id: str,
        function_call: FunctionCall,
        arguments: Mapping[str, object],
    ) -> bool:
        """请求执行许可，并发布完整的审批事件。"""
        self.publish_event(
            ApprovalRequested(
                turn_id,
                function_call.call_id,
                function_call.name,
            )
        )
        approved = (
            self.approve_tool_call is not None
            and self.approve_tool_call(function_call.name, arguments)
        )
        self.publish_event(
            ApprovalResolved(
                turn_id,
                function_call.call_id,
                function_call.name,
                approved,
            )
        )
        return approved

    def _approval_failure(self) -> ToolResult:
        """区分缺少审批处理器和用户主动拒绝。"""
        if self.approve_tool_call is None:
            message = "tool execution requires user approval"
        else:
            message = (
                "tool execution was declined by the user; do not retry"
            )
        return ToolResult(False, message)

    def _finish(
        self,
        turn_id: str,
        function_call: FunctionCall,
        result: ToolResult,
    ) -> ToolResult:
        """发布工具调用的最终事件，并原样返回结果。"""
        self.publish_event(
            ToolFinished(
                turn_id,
                function_call.call_id,
                function_call.name,
                result,
            )
        )
        return result
