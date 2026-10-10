from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from langfuse import Langfuse, propagate_attributes

from praxis_agent.config import LangfuseConfig
from praxis_agent.events import (
    AgentEvent,
    ChildAgentEvent,
    ContextCompaction,
    AssistantTextCompleted,
    ModelRequestCompleted,
    ModelRequestStarted,
    ToolCallPrepared,
    ToolFinished,
    ToolStarted,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
)


_LOGGER = logging.getLogger(__name__)
_MAX_TASK_CHARS = 4_000
_MAX_MODEL_INPUT_CHARS = 8_000
_MAX_MODEL_OUTPUT_CHARS = 8_000
_MAX_TOOL_ARGUMENT_CHARS = 4_000
_MAX_TOOL_RESULT_CHARS = 4_000
_MAX_VALUE_CHARS = 2_000
_ObservationType = Literal[
    "span",
    "agent",
    "tool",
    "generation",
]


class _Observation(Protocol):
    """描述本模块使用到的 Langfuse observation 最小接口。"""

    def start_observation(
        self,
        *,
        name: str,
        as_type: _ObservationType = "span",
        input: Any | None = None,
        metadata: Any | None = None,
    ) -> "_Observation": ...

    def update(self, **values: Any) -> object: ...

    def end(self, *, end_time: int | None = None) -> object: ...

    def set_trace_io(
        self,
        *,
        input: Any | None = None,
        output: Any | None = None,
    ) -> object: ...


class _LangfuseClient(Protocol):
    """描述事件消费者依赖的 Langfuse 客户端最小接口。"""

    def start_as_current_observation(
        self,
        *,
        trace_context: Mapping[str, str],
        name: str,
        as_type: _ObservationType,
        input: Any | None = None,
        metadata: Any | None = None,
    ) -> AbstractContextManager[_Observation]: ...

    def flush(self) -> None: ...


@dataclass(slots=True)
class _TurnTrace:
    observation: _Observation
    observation_context: AbstractContextManager[_Observation]
    attributes_context: AbstractContextManager[Any]


@dataclass(frozen=True, slots=True)
class _PreparedToolCall:
    arguments: Mapping[str, object]
    requires_approval: bool


def _create_langfuse_client(config: LangfuseConfig) -> Langfuse:
    """创建 Langfuse 客户端，并在 Windows 上使用系统证书库。"""
    if sys.platform == "win32":
        import truststore

        # 公司 CA 已由 Windows 信任，但不一定包含在 certifi 证书库中。
        truststore.inject_into_ssl()

    return Langfuse(
        public_key=config.public_key,
        secret_key=config.secret_key,
        base_url=config.base_url,
        environment=config.environment,
    )


class LangfuseEventSink:
    """把 Praxis 事件转换成 Langfuse Trace 和 observation。"""

    def __init__(
        self,
        config: LangfuseConfig,
        workspace: Path,
        *,
        client: _LangfuseClient | None = None,
        session_id: str | None = None,
    ) -> None:
        self._client = (
            client
            if client is not None
            else _create_langfuse_client(config)
        )
        self._workspace = workspace.resolve()
        self._environment = config.environment
        self._session_id = session_id or uuid4().hex
        self._turns: dict[str, _TurnTrace] = {}
        self._generations: dict[tuple[str, int], _Observation] = {}
        self._prepared_tools: dict[tuple[str, str], _PreparedToolCall] = {}
        self._tools: dict[tuple[str, str], _Observation] = {}
        self._compactions: dict[str, _Observation] = {}
        self._summary_requests: dict[tuple[str, int, int], _Observation] = {}
        self._worker_spans: dict[str, _Observation] = {}
        self._worker_requests: dict[tuple[str, int], _Observation] = {}
        self._worker_tools: dict[tuple[str, str], _Observation] = {}
        self._disabled = False

    @property
    def session_id(self) -> str:
        """返回当前 CLI 会话在 Langfuse 中使用的标识。"""
        return self._session_id

    def emit(self, event: AgentEvent) -> None:
        """处理一个事件；观测异常不会中断 Agent 主流程。"""
        if self._disabled:
            return

        try:
            self._handle_event(event)
        except Exception as error:
            self._disabled = True
            _LOGGER.warning(
                "Langfuse observability disabled after an error: %s",
                error,
            )
            self._close_open_traces()

    def close(self) -> None:
        """关闭残留 Trace，并把缓存中的事件发送给 Langfuse。"""
        self._close_open_traces()
        try:
            self._client.flush()
        except Exception as error:
            _LOGGER.warning("Failed to flush Langfuse events: %s", error)

    def _handle_event(self, event: AgentEvent) -> None:
        if isinstance(event, ChildAgentEvent):
            self._handle_worker_event(event)
        elif isinstance(event, TurnStarted):
            self._start_turn(event)
        elif isinstance(event, ContextCompaction):
            self._compaction_progress(event)
        elif isinstance(event, ModelRequestStarted):
            self._start_generation(event)
        elif isinstance(event, ModelRequestCompleted):
            self._finish_generation(event)
        elif isinstance(event, ToolCallPrepared):
            self._remember_tool_call(event)
        elif isinstance(event, ToolStarted):
            self._start_tool(event)
        elif isinstance(event, ToolFinished):
            self._finish_tool(event)
        elif isinstance(event, AssistantTextCompleted):
            self._set_turn_output(event)
        elif isinstance(event, TurnCompleted):
            self._finish_turn(event.turn_id)
        elif isinstance(event, TurnFailed):
            self._fail_turn(event)

    def _handle_worker_event(self, envelope: ChildAgentEvent) -> None:
        """Use explicit parent observations, never overlapping context-manager stacks."""
        event = envelope.event
        worker_id = event.turn_id
        if isinstance(event, TurnStarted):
            parent = self._turns.get(envelope.turn_id)
            if parent is not None:
                self._worker_spans[worker_id] = parent.observation.start_observation(
                    name="praxis-worker", as_type="agent",
                    input=_truncate_text(event.task, _MAX_TASK_CHARS),
                    metadata={"agent_id": envelope.agent_id, "run_id": worker_id,
                              "parent_turn_id": envelope.turn_id},
                )
            return
        worker = self._worker_spans.get(worker_id)
        if worker is None:
            return
        if isinstance(event, ModelRequestStarted):
            self._worker_requests[(worker_id, event.request_number)] = worker.start_observation(
                name="worker-model", as_type="generation",
                input=_bounded_payload(event.input_items, _MAX_MODEL_INPUT_CHARS),
            )
        elif isinstance(event, ModelRequestCompleted):
            observation = self._worker_requests.pop((worker_id, event.request_number), None)
            if observation is not None:
                response = event.response
                if response is not None:
                    values = {"model": response.model,
                              "output": _truncate_text(response.output_text, _MAX_MODEL_OUTPUT_CHARS)}
                    if response.usage is not None:
                        values["usage_details"] = {"input": response.usage.input_tokens,
                                                   "output": response.usage.output_tokens,
                                                   "total": response.usage.total_tokens}
                    observation.update(**values)
                observation.end()
        elif isinstance(event, ToolStarted):
            self._worker_tools[(worker_id, event.call_id)] = worker.start_observation(
                name=event.tool_name, as_type="tool", metadata={"call_id": event.call_id},
            )
        elif isinstance(event, ToolFinished):
            observation = self._worker_tools.pop((worker_id, event.call_id), None)
            if observation is not None:
                observation.update(output=_bounded_payload(asdict(event.result), _MAX_TOOL_RESULT_CHARS))
                observation.end()
        elif isinstance(event, AssistantTextCompleted):
            worker.update(output=_truncate_text(event.text, _MAX_MODEL_OUTPUT_CHARS))
        elif isinstance(event, (TurnCompleted, TurnFailed)):
            if isinstance(event, TurnFailed):
                worker.update(level="ERROR", status_message=_truncate_text(event.message, _MAX_VALUE_CHARS))
            self._finish_worker(worker_id)

    def _finish_worker(self, worker_id: str) -> None:
        for observations in (self._worker_requests, self._worker_tools):
            for key in list(observations):
                if key[0] == worker_id:
                    observation = observations.pop(key)
                    observation.update(level="ERROR", status_message="Worker stopped before completion")
                    observation.end()
        worker = self._worker_spans.pop(worker_id, None)
        if worker is not None:
            worker.end()

    def _compaction_progress(self, event: ContextCompaction) -> None:
        turn = self._turns.get(event.turn_id)
        if turn is None:
            return
        metadata = asdict(event)
        metadata["error"] = _truncate_text(event.error, _MAX_VALUE_CHARS)
        if event.stage == "started":
            self._compactions[event.turn_id] = turn.observation.start_observation(
                name="context-compaction", as_type="span", metadata=metadata,
            )
            return
        compaction = self._compactions.get(event.turn_id)
        if compaction is None:
            return
        key = (event.turn_id, event.round_number, event.chunk_number)
        if event.stage == "chunk_started":
            self._summary_requests[key] = compaction.start_observation(
                name=f"summary-round-{event.round_number}-chunk-{event.chunk_number}",
                as_type="generation", metadata=metadata,
            )
        elif event.stage == "chunk_completed":
            generation = self._summary_requests.pop(key, None)
            if generation is not None:
                values: dict[str, object] = {"metadata": metadata}
                if event.model is not None:
                    values["model"] = event.model
                if event.input_tokens is not None:
                    values["usage_details"] = {"input": event.input_tokens, "output": event.output_tokens}
                generation.update(**values)
                generation.end()
        elif event.stage in {"completed", "failed"}:
            error = metadata["error"] if event.stage == "failed" else ""
            self._close_summary_requests(event.turn_id, str(error))
            values = {"metadata": metadata, "output": {
                "estimated_before_tokens": event.before_tokens,
                "estimated_after_tokens": event.after_tokens,
                "elapsed_seconds": event.elapsed_seconds,
            }}
            if error:
                values.update(level="ERROR", status_message=error)
            compaction.update(**values)
            compaction.end()
            del self._compactions[event.turn_id]

    def _close_summary_requests(self, turn_id: str, error: str) -> None:
        for key, observation in list(self._summary_requests.items()):
            if key[0] != turn_id:
                continue
            observation.update(level="ERROR", status_message=error or "Summary request interrupted")
            observation.end()
            del self._summary_requests[key]

    def _start_turn(self, event: TurnStarted) -> None:
        task = _bounded_payload(event.task, _MAX_TASK_CHARS)
        observation_context = self._client.start_as_current_observation(
            trace_context={"trace_id": event.turn_id},
            name="praxis-turn",
            as_type="agent",
            input={"task": task},
            metadata={
                "turn_id": event.turn_id,
                "workspace": str(self._workspace),
            },
        )
        observation = observation_context.__enter__()
        attributes_context = propagate_attributes(
            session_id=self._session_id,
            trace_name="praxis-turn",
            environment=self._environment,
            tags=["praxis", "coding-agent"],
        )
        try:
            attributes_context.__enter__()
        except Exception:
            observation_context.__exit__(None, None, None)
            raise

        self._turns[event.turn_id] = _TurnTrace(
            observation=observation,
            observation_context=observation_context,
            attributes_context=attributes_context,
        )
        observation.set_trace_io(input={"task": task})

    def _start_generation(self, event: ModelRequestStarted) -> None:
        turn = self._turns.get(event.turn_id)
        if turn is None:
            return

        key = (event.turn_id, event.request_number)
        generation = turn.observation.start_observation(
            name=f"model-request-{event.request_number}",
            as_type="generation",
            input=_bounded_payload(
                event.input_items,
                _MAX_MODEL_INPUT_CHARS,
            ),
            metadata={"request_number": event.request_number},
        )
        self._generations[key] = generation

    def _finish_generation(self, event: ModelRequestCompleted) -> None:
        generation = self._generations.pop(
            (event.turn_id, event.request_number),
            None,
        )
        if generation is None:
            return

        response = event.response
        if response is not None:
            values: dict[str, object] = {
                "output": _bounded_payload(
                    {
                        "text": response.output_text,
                        "tool_calls": [
                            {
                                "call_id": call.call_id,
                                "name": call.name,
                            }
                            for call in response.function_calls
                        ],
                    },
                    _MAX_MODEL_OUTPUT_CHARS,
                ),
                "metadata": {
                    "request_number": event.request_number,
                    "response_id": response.response_id or "",
                },
            }
            if response.model is not None:
                values["model"] = response.model
            if response.usage is not None:
                values["usage_details"] = {
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                    "total": response.usage.total_tokens,
                }
            generation.update(**values)
        generation.end()

    def _remember_tool_call(self, event: ToolCallPrepared) -> None:
        self._prepared_tools[(event.turn_id, event.call_id)] = (
            _PreparedToolCall(
                arguments=dict(event.arguments),
                requires_approval=event.requires_approval,
            )
        )

    def _start_tool(self, event: ToolStarted) -> None:
        turn = self._turns.get(event.turn_id)
        if turn is None:
            return

        key = (event.turn_id, event.call_id)
        prepared = self._prepared_tools.get(key)
        arguments = {} if prepared is None else prepared.arguments
        requires_approval = (
            False if prepared is None else prepared.requires_approval
        )
        self._tools[key] = turn.observation.start_observation(
            name=event.tool_name,
            as_type="tool",
            input=_bounded_payload(arguments, _MAX_TOOL_ARGUMENT_CHARS),
            metadata={
                "call_id": event.call_id,
                "requires_approval": requires_approval,
                "executed": True,
            },
        )

    def _finish_tool(self, event: ToolFinished) -> None:
        key = (event.turn_id, event.call_id)
        observation = self._tools.pop(key, None)
        prepared = self._prepared_tools.pop(key, None)
        executed = observation is not None

        # 审批被拒绝时不会产生 ToolStarted，但仍应留下记录。
        if observation is None:
            turn = self._turns.get(event.turn_id)
            if turn is None:
                return
            arguments = {} if prepared is None else prepared.arguments
            observation = turn.observation.start_observation(
                name=event.tool_name,
                as_type="tool",
                input=_bounded_payload(
                    arguments,
                    _MAX_TOOL_ARGUMENT_CHARS,
                ),
                metadata={
                    "call_id": event.call_id,
                    "requires_approval": (
                        False
                        if prepared is None
                        else prepared.requires_approval
                    ),
                    "executed": False,
                },
            )

        values: dict[str, object] = {
            "output": _bounded_payload(
                {
                    "success": event.result.success,
                    "output": event.result.output,
                },
                _MAX_TOOL_RESULT_CHARS,
            ),
            "metadata": {
                "call_id": event.call_id,
                "success": event.result.success,
                "requires_approval": (
                    False
                    if prepared is None
                    else prepared.requires_approval
                ),
                "executed": executed,
            },
        }
        if not event.result.success:
            values["level"] = "ERROR"
            values["status_message"] = _truncate_text(
                event.result.output,
                _MAX_VALUE_CHARS,
            )
        observation.update(**values)
        observation.end()

    def _set_turn_output(self, event: AssistantTextCompleted) -> None:
        turn = self._turns.get(event.turn_id)
        if turn is None:
            return
        output = {"text": _truncate_text(event.text, _MAX_MODEL_OUTPUT_CHARS)}
        turn.observation.update(output=output)
        turn.observation.set_trace_io(output=output)

    def _fail_turn(self, event: TurnFailed) -> None:
        turn = self._turns.get(event.turn_id)
        if turn is not None:
            message = _truncate_text(event.message, _MAX_VALUE_CHARS)
            output = {
                "error_type": event.error_type,
                "message": message,
            }
            turn.observation.update(
                output=output,
                level="ERROR",
                status_message=message,
            )
            turn.observation.set_trace_io(output=output)
        self._finish_turn(event.turn_id, incomplete=True)

    def _finish_turn(
        self,
        turn_id: str,
        *,
        incomplete: bool = False,
    ) -> None:
        self._close_turn_children(turn_id, incomplete=incomplete)
        turn = self._turns.pop(turn_id, None)
        if turn is None:
            return

        turn.attributes_context.__exit__(None, None, None)
        turn.observation_context.__exit__(None, None, None)

    def _close_turn_children(
        self,
        turn_id: str,
        *,
        incomplete: bool,
    ) -> None:
        status = "Turn ended before this observation completed"
        self._close_summary_requests(turn_id, status)
        compaction = self._compactions.pop(turn_id, None)
        if compaction is not None:
            compaction.update(level="ERROR", status_message=status)
            compaction.end()
        for key, observation in list(self._generations.items()):
            if key[0] != turn_id:
                continue
            if incomplete:
                observation.update(level="ERROR", status_message=status)
            observation.end()
            del self._generations[key]

        for key, observation in list(self._tools.items()):
            if key[0] != turn_id:
                continue
            if incomplete:
                observation.update(level="ERROR", status_message=status)
            observation.end()
            del self._tools[key]

        for key in list(self._prepared_tools):
            if key[0] == turn_id:
                del self._prepared_tools[key]

    def _close_open_traces(self) -> None:
        for worker_id in list(self._worker_spans):
            self._finish_worker(worker_id)
        for turn_id in list(self._turns):
            try:
                turn = self._turns[turn_id]
                message = "Praxis exited before the turn completed"
                turn.observation.update(
                    level="ERROR",
                    status_message=message,
                )
                self._finish_turn(turn_id, incomplete=True)
            except Exception as error:
                _LOGGER.warning(
                    "Failed to close a Langfuse trace: %s",
                    error,
                )


def _bounded_payload(value: object, max_chars: int) -> object:
    """转换成 JSON 兼容的数据，并限制一次上报的整体大小。"""
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    if len(encoded) <= max_chars:
        return json.loads(encoded)
    return {
        "truncated": True,
        "preview": _truncate_text(encoded, max_chars),
    }


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + "\n... Langfuse payload truncated ...\n"
        + text[-half:]
    )


__all__ = ["LangfuseEventSink"]
