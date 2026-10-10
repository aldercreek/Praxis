import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from uuid import uuid4

from praxis_agent.config import DEFAULT_MAX_PARALLEL_TOOLS
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
    """主线程调度、审批和发布事件，只有标记安全的 handler 进入线程池。"""

    tool_registry: ToolRegistry
    context: ToolContext
    publish_event: EventPublisher
    approve_tool_call: ToolApprovalHandler | None = None
    bypass_approvals: bool = False
    max_parallel_tools: int = DEFAULT_MAX_PARALLEL_TOOLS
    check_execution: Callable[[], None] | None = None
    live_registry: ToolRegistry | None = None

    def execute_many(
        self, turn_id: str, calls: Sequence[FunctionCall],
    ) -> Iterator[tuple[FunctionCall, ToolResult]]:
        """连续只读调用组成一组；本组结果全部交给调用方后才启动下一组。"""
        batch = replace(self, tool_registry=self.tool_registry.snapshot(),
                        live_registry=self.tool_registry)
        yield from batch._execute_batch(turn_id, calls)

    def _execute_batch(self, turn_id, calls):
        index = 0
        while index < len(calls):
            call = calls[index]
            if not self.tool_registry.is_parallel_safe(call.name):
                yield call, self.execute(turn_id, call)
                index += 1
                continue
            end = index + 1
            while end < len(calls) and self.tool_registry.is_parallel_safe(calls[end].name):
                end += 1
            group = calls[index:end]
            results = self._execute_parallel(turn_id, group)
            yield from zip(group, results)
            index = end

    def _execute_parallel(self, turn_id: str, calls: Sequence[FunctionCall]) -> list[ToolResult]:
        results: dict[int, ToolResult] = {}
        ready = []
        # 启动本组前，在主线程完成解析、校验和审批。
        for index, call in enumerate(calls):
            prepared = self._prepare(turn_id, call)
            if isinstance(prepared, ToolResult):
                results[index] = self._finish(turn_id, call, prepared)
            else:
                ready.append((index, prepared))
        if ready:
            with ThreadPoolExecutor(max_workers=self.max_parallel_tools,
                                    thread_name_prefix="praxis-tool") as pool:
                waiting = iter(ready)
                pending = {}

                def fill_slots() -> None:
                    while len(pending) < self.max_parallel_tools:
                        item = next(waiting, None)
                        if item is None:
                            break
                        index, handler = item
                        self._start(turn_id, calls[index])
                        pending[pool.submit(handler)] = index

                fill_slots()
                while pending:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        index = pending.pop(future)
                        result = self._result(future.result)
                        results[index] = self._finish(turn_id, calls[index], result)
                    fill_slots()
        return [results[index] for index in range(len(calls))]

    def execute(
        self,
        turn_id: str,
        function_call: FunctionCall,
    ) -> ToolResult:
        """执行模型请求的工具调用，并返回统一的工具结果。"""
        prepared = self._prepare(turn_id, function_call)
        if isinstance(prepared, ToolResult):
            return self._finish(turn_id, function_call, prepared)
        self._start(turn_id, function_call)
        return self._finish(turn_id, function_call, self._result(prepared))

    @staticmethod
    def _result(invoke: Callable[[], str]) -> ToolResult:
        """主线程收集返回值或异常，单个 handler 失败不打断其余调用。"""
        try:
            result = invoke()
            return result if isinstance(result, ToolResult) else ToolResult(True, result)
        except Exception as error:
            return ToolResult(False, f"{type(error).__name__}: {error}")

    def _prepare(
        self, turn_id: str, function_call: FunctionCall,
    ) -> Callable[[], str] | ToolResult:
        if self.check_execution is not None:
            try:
                self.check_execution()
            except RuntimeError as error:
                # Still return a result for every model call so history stays paired.
                return ToolResult(False, f"Not executed: {error}")
        try:
            arguments = _parse_arguments(function_call.arguments)
        except _InvalidToolArgumentsError as error:
            return self.tool_registry.argument_error(function_call.name, str(error))

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

        def invoke_tool(name: str, values: dict) -> ToolResult:
            executor = replace(self, tool_registry=self.live_registry or self.tool_registry)
            return executor.execute(turn_id, FunctionCall(
                f"{function_call.call_id}:{uuid4().hex}", name, json.dumps(values)))

        def invoke_programmatic_tool(name: str, values: dict) -> ToolResult:
            registry = self.live_registry or self.tool_registry
            if not registry.allows_programmatic(name):
                return ToolResult(False, f"Tool unavailable for programmatic calls: {name}")
            return invoke_tool(name, values)

        context = replace(self.context, invoke_tool=invoke_tool,
                          invoke_programmatic_tool=invoke_programmatic_tool,
                          check_execution=self.check_execution)
        prepared = self.tool_registry.prepare(function_call.name, arguments, context=context)
        if isinstance(prepared, ToolResult):
            return prepared
        if requires_approval and not self._request_approval(
            turn_id,
            function_call,
            arguments,
        ):
            return self._approval_failure()
        if self.check_execution is None:
            return prepared

        def invoke() -> str:
            # A parallel handler may have waited for a slot since preparation.
            self.check_execution()
            return prepared()

        return invoke

    def _start(self, turn_id: str, function_call: FunctionCall) -> None:
        self.publish_event(
            ToolStarted(
                turn_id,
                function_call.call_id,
                function_call.name,
            )
        )

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
        if result.success and function_call.name in {"create_file", "edit_file"}:
            path = json.loads(function_call.arguments).get("relative_path")
            if isinstance(path, str):
                self.context.run_state.record_changed_file(path)
        self.publish_event(
            ToolFinished(
                turn_id,
                function_call.call_id,
                function_call.name,
                result,
            )
        )
        return result
