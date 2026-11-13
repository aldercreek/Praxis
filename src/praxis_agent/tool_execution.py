import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass, field, replace
from copy import deepcopy
from time import monotonic
from uuid import uuid4

from praxis_agent.config import DEFAULT_MAX_PARALLEL_TOOLS
from praxis_agent.events import (
    AgentEvent,
    ApprovalRequested,
    ApprovalResolved,
    ToolCallPrepared,
    ToolFinished,
    ToolOutputDelta,
    ToolStarted,
)
from praxis_agent.hooks import HookError, HookManager
from praxis_agent.hook_types import HookEvent
from praxis_agent.execution import current_run, run_operation
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
            f"(line {error.lineno}, column {error.colno}, character {error.pos + 1}); correct the JSON syntax."
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
    hooks: HookManager = field(default_factory=HookManager)
    session_id: str | None = None
    agent_id: str = "Praxis"
    parent_call_id: str | None = None

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
                results[index] = self._finish(turn_id, call, self._unexecuted(prepared))
            else:
                ready.append((index, prepared))
        if ready:
            with ThreadPoolExecutor(max_workers=self.max_parallel_tools,
                                    thread_name_prefix="praxis-tool") as pool:
                waiting = iter(ready)
                pending = {}

                def fill_slots() -> None:
                    if current_run() is not None and current_run().cancelled.is_set():
                        return
                    while len(pending) < self.max_parallel_tools:
                        item = next(waiting, None)
                        if item is None:
                            break
                        index, handler = item
                        self._start(turn_id, calls[index])
                        pending[pool.submit(copy_context().run, handler)] = index

                try:
                    fill_slots()
                    while pending:
                        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in completed:
                            index = pending.pop(future)
                            result = self._result(future.result)
                            results[index] = self._finish(turn_id, calls[index], result)
                        fill_slots()
                except BaseException:
                    if current_run() is not None:
                        current_run().cancel()
                    raise
            for index, call in enumerate(calls):
                if index not in results:
                    results[index] = self._finish(
                        turn_id, call, ToolResult(False, "Not executed: run cancelled"),
                    )
        return [results[index] for index in range(len(calls))]

    def execute(
        self,
        turn_id: str,
        function_call: FunctionCall,
    ) -> ToolResult:
        """执行模型请求的工具调用，并返回统一的工具结果。"""
        prepared = self._prepare(turn_id, function_call)
        if isinstance(prepared, ToolResult):
            return self._finish(turn_id, function_call, self._unexecuted(prepared))
        self._start(turn_id, function_call)
        try:
            result = self._result(prepared)
        finally:
            if function_call.name in {"run_command", "wait_command"}:
                self.publish_event(ToolOutputDelta(turn_id, function_call.call_id, "", "", finished=True))
        return self._finish(turn_id, function_call, result)

    @staticmethod
    def _result(invoke: Callable[[], str]) -> ToolResult:
        """主线程收集返回值或异常，单个 handler 失败不打断其余调用。"""
        try:
            result = invoke()
            return result if isinstance(result, ToolResult) else ToolResult(True, result)
        except Exception as error:
            # Preserve a cancelled tool's partial output in the transcript/history.
            # The next dispatch or model-loop checkpoint propagates cancellation.
            return ToolResult(False, f"{type(error).__name__}: {error}")

    def _prepare(
        self, turn_id: str, function_call: FunctionCall,
    ) -> Callable[[], str] | ToolResult:
        if self.check_execution is not None:
            self.check_execution()
        try:
            arguments = _parse_arguments(function_call.arguments)
        except _InvalidToolArgumentsError as error:
            return self.tool_registry.argument_error(function_call.name, str(error))

        if self.context.programmatic_call and not self.tool_registry.allows_programmatic(function_call.name):
            return ToolResult(False, f"Tool unavailable for programmatic calls: {function_call.name}")
        tool = next((tool for tool in self.tool_registry.tools if tool.name == function_call.name), None)
        if tool is None:
            return self.tool_registry.argument_error(function_call.name, "unknown tool")
        event = HookEvent(
            event="before_tool_call", tool_name=tool.name, description=tool.description,
            parameters=deepcopy(tool.parameters), requires_approval=tool.requires_approval,
            session_id=self.session_id, agent_id=self.agent_id, turn_id=turn_id,
            call_id=function_call.call_id, parent_call_id=self.parent_call_id,
            workspace=self.context.workspace, programmatic_call=self.context.programmatic_call,
            original_arguments=deepcopy(arguments), arguments=deepcopy(arguments),
        )
        try:
            event, denied = self.hooks.before(event)
        except HookError as error:
            return ToolResult(False, str(error), presentation={"execution_status": "not_executed", "blocked_by": "hook"})
        arguments = deepcopy(event.arguments)
        # An explicit hook approval requirement survives the normal bypass setting.
        requires_approval = event.approval_requested or (tool.requires_approval and not self.bypass_approvals)
        presentation = dict(source=tool.source, display_name=tool.display_name or tool.name,
                            description=tool.description, arguments_modified=event.arguments != event.original_arguments,
                            approval_hooks=list(event.approval_hooks))
        self.publish_event(ToolCallPrepared(turn_id, function_call.call_id, function_call.name,
                                            deepcopy(arguments), bool(requires_approval and not denied), presentation))
        if denied:
            return ToolResult(False, denied, presentation={**presentation,
                              "execution_status": "not_executed", "blocked_by": "hook"})

        def invoke_tool(name: str, values: dict) -> ToolResult:
            executor = replace(self, tool_registry=self.live_registry or self.tool_registry,
                               parent_call_id=function_call.call_id)
            return executor.execute(turn_id, FunctionCall(
                f"{function_call.call_id}:{uuid4().hex}", name, json.dumps(values)))

        def invoke_programmatic_tool(name: str, values: dict) -> ToolResult:
            registry = self.live_registry or self.tool_registry
            if not registry.allows_programmatic(name):
                return ToolResult(False, f"Tool unavailable for programmatic calls: {name}")
            executor = replace(self, tool_registry=registry, parent_call_id=function_call.call_id,
                               context=replace(self.context, programmatic_call=True))
            return executor.execute(turn_id, FunctionCall(
                f"{function_call.call_id}:{uuid4().hex}", name, json.dumps(values)))

        def invoke_parallel_tools(requests: list[dict], max_workers: int) -> list[ToolResult]:
            registry = (self.live_registry or self.tool_registry).snapshot()
            scope = current_run()

            def check():
                if scope is not None:
                    scope.check()
                if self.check_execution is not None:
                    self.check_execution()

            executor = replace(
                self, tool_registry=registry, live_registry=None, parent_call_id=function_call.call_id,
                context=replace(self.context, programmatic_call=True, parallel_tool_call=True),
                max_parallel_tools=max_workers, check_execution=check,
            )
            calls, positions, results = [], [], {}
            for index, item in enumerate(requests):
                check()
                call_id = f"{function_call.call_id}:{uuid4().hex}"
                try:
                    arguments = json.dumps(item["arguments"])
                except (TypeError, ValueError, RecursionError) as error:
                    # No JSON arguments exist for this item; finish it without
                    # submitting it for approval or executing its handler.
                    call = FunctionCall(call_id, item["name"], "")
                    failure = registry.argument_error(item["name"],
                        f"Arguments cannot be encoded as JSON: {error}")
                    results[index] = executor._finish(turn_id, call, failure)
                else:
                    calls.append(FunctionCall(call_id, item["name"], arguments))
                    positions.append(index)
            # Explicit caller-selected concurrency; do not consult parallel_safe.
            results.update(zip(positions, executor._execute_parallel(turn_id, calls)))
            return [results[index] for index in range(len(requests))]

        context = replace(self.context, invoke_tool=invoke_tool,
                          invoke_programmatic_tool=invoke_programmatic_tool,
                          invoke_parallel_tools=invoke_parallel_tools,
                          check_execution=self.check_execution,
                          on_command_output=(None if self.hooks.handles_results(function_call.name) else
                                             lambda stream, text: self.publish_event(ToolOutputDelta(
                              turn_id, function_call.call_id, stream, text,
                              command=arguments.get("command") or arguments.get("session_id") or " ".join(str(part) for part in arguments.get("argv", [])),
                          ))))
        prepared = self.tool_registry.prepare(function_call.name, arguments, context=context)
        if isinstance(prepared, ToolResult):
            return prepared
        if requires_approval and not self._request_approval(
            turn_id,
            function_call,
            deepcopy(arguments),
        ):
            return self._approval_failure()
        def invoke() -> str:
            # A parallel handler may have waited for a slot since preparation.
            if self.check_execution is not None:
                self.check_execution()
            started = monotonic()
            result = self._result(lambda: run_operation(function_call.name, lambda handle: prepared()))
            if result.success and function_call.name in {"write_file", "edit_file"}:
                path = arguments.get("relative_path")
                if isinstance(path, str):
                    self.context.run_state.record_changed_file(path)
            status = "success" if result.success else "failure"
            metadata = {**presentation, "execution_status": status, "result_processing": "unchanged"}
            if function_call.name in {"run_command", "wait_command"}:
                command_status = (result.structured_output or {}).get("status")
                if command_status in {"running", "exited", "failed", "spawn_failed", "timed_out", "cancelled"}:
                    metadata["command_status"] = command_status
                metadata["command_exit_code"] = (result.structured_output or {}).get("exit_code")
                metadata["command_session_id"] = (result.structured_output or {}).get("session_id")
            if function_call.name == "run_code":
                try:
                    code_result = result.structured_output or json.loads(result.output)
                except (ValueError, TypeError):
                    code_result = {}
                if isinstance(code_result, dict) and code_result.get("error_kind") == "cancelled":
                    metadata["cancelled"] = True
            if result.output.startswith("ExecutionStopped:"):
                metadata["cancelled"] = True
            result = replace(result, presentation=metadata)
            return self.hooks.after(event, result, (monotonic() - started) * 1000)

        return invoke

    @staticmethod
    def _unexecuted(result):
        return result if result.presentation else replace(result, presentation={
            "execution_status": "not_executed", "blocked_by": "invalid_arguments"})

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
        approved = False
        try:
            approved = bool(self.approve_tool_call and run_operation(
                "approval", lambda handle: self.approve_tool_call(function_call.name, arguments),
            ))
        finally:
            self.publish_event(ApprovalResolved(
                turn_id, function_call.call_id, function_call.name, approved,
            ))
        return approved

    def _approval_failure(self) -> ToolResult:
        """区分缺少审批处理器和用户主动拒绝。"""
        if self.approve_tool_call is None:
            message = "tool execution requires user approval"
        else:
            message = (
                "tool execution was declined by the user; do not retry"
            )
        return ToolResult(False, message, presentation={"execution_status": "not_executed", "blocked_by": "approval"})

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
