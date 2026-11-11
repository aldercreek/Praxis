from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from time import monotonic
from uuid import uuid4

from praxis_agent.context.capacity import (
    ContextPolicy,
    ContextStatus,
    HeuristicTokenEstimator,
    TokenEstimator,
)
from praxis_agent.context.model_metadata import ModelLimits
from praxis_agent.context.compaction import trim_tool_messages
from praxis_agent.context.model_context import build_context_status_context
from praxis_agent.context.models import (
    ConversationHistory,
    ModelContextView,
)
from praxis_agent.context.operations import (
    ContextOperation,
    ContextOperations,
    compression_reasoning_ids,
    is_editable,
    render_message,
)
from praxis_agent.context.summarizer import ContextSummarizer
from praxis_agent.context.tool_output import preview_tool_outputs
from praxis_agent.events import ContextCompaction, EventSink
from praxis_agent.execution import ExecutionStopped
from praxis_agent.model_errors import ModelClientError, PromptTooLongError, SummaryError, SummaryTooLongError


@dataclass(frozen=True, slots=True)
class _InputTokenAnchor:
    """关联一次请求的真实输入量和本地估算量。"""

    actual_input_tokens: int
    heuristic_input_tokens: int


class ContextManager:
    """根据完整历史构建一次模型请求使用的上下文视图。"""

    def __init__(
        self,
        *,
        model_limits: ModelLimits | None = None,
        policy: ContextPolicy | None = None,
        instructions: str = "",
        token_estimator: TokenEstimator | None = None,
        summarizer: ContextSummarizer | None = None,
    ) -> None:
        self._model_limits = model_limits
        self._policy = policy or (
            ContextPolicy.for_model(model_limits) if model_limits else ContextPolicy()
        )
        self._instructions = instructions
        self._token_estimator = token_estimator or HeuristicTokenEstimator()
        self._input_token_anchor: _InputTokenAnchor | None = None
        self._pending_heuristic_input_tokens: int | None = None
        self.operations = ContextOperations()
        self._summarizer = summarizer
        self._current_messages: list[dict[str, object]] = []
        self._view_operation_count = 0
        self._history_snapshot = ()
        self._transient_items = ()
        self._summary_context = ()
        self._tool_definitions = ()
        self.event_sink: EventSink | None = None
        self.turn_id = ""

    def record_actual_input_tokens(
        self,
        actual_input_tokens: int | None,
    ) -> None:
        """使用模型返回值校准最近一次构建的上下文视图。"""
        heuristic_input_tokens = self._pending_heuristic_input_tokens
        self._pending_heuristic_input_tokens = None
        if actual_input_tokens is None or heuristic_input_tokens is None:
            return
        if actual_input_tokens < 0:
            raise ValueError("actual_input_tokens must not be negative")
        self._input_token_anchor = _InputTokenAnchor(
            actual_input_tokens=actual_input_tokens,
            heuristic_input_tokens=heuristic_input_tokens,
        )

    def reset(self) -> None:
        """清除旧视图和输入 Token 校准状态；下一轮重新构建。"""
        self._input_token_anchor = None
        self._pending_heuristic_input_tokens = None

        self._current_messages = []
        self._view_operation_count = 0
        self._summary_context = ()

    def build_view(
        self,
        history: ConversationHistory,
        *,
        transient_items: Sequence[object] = (),
        tool_definitions: Sequence[dict[str, object]] = (),
        summary_context: Sequence[dict[str, object]] | None = None,
    ) -> ModelContextView:
        """构建完整历史、运行时状态和可选容量状态组成的视图。"""
        if self._view_operation_count != len(self.operations.records):
            self.reset()
        self._history_snapshot = history.snapshot()
        self._transient_items = tuple(transient_items)
        self._tool_definitions = tuple(tool_definitions)
        self._current_messages = self.operations.project(self._history_snapshot)
        if summary_context is None:
            latest_user = next((m["payload"] for m in reversed(self._current_messages)
                                if m["payload"].get("role") == "user"), None)
            summary_context = (() if latest_user is None else (latest_user,)) + tuple(transient_items)
        self._summary_context = tuple(summary_context)
        self._view_operation_count = len(self.operations.records)
        base_items = self._render_current_messages() + tuple(transient_items)
        if self._model_limits is None:
            self._pending_heuristic_input_tokens = None
            return ModelContextView(items=base_items)

        view, heuristic_input_tokens = self._estimate_view(base_items, tool_definitions)
        if view.context_status.estimated_input_tokens >= self._trigger_tokens(view.context_status):
            self._observe_compaction(
                transient_items, tool_definitions, view.context_status.estimated_input_tokens,
            )
            self._current_messages = self.operations.project(self._history_snapshot)
            self._view_operation_count = len(self.operations.records)
            view, heuristic_input_tokens = self._estimate_view(
                self._render_current_messages() + tuple(transient_items), tool_definitions,
            )
        self._pending_heuristic_input_tokens = heuristic_input_tokens
        return view

    def _render_current_messages(self) -> tuple[object, ...]:
        origins = {op.replacement["message_id"]: op.operation_id
                   for op in self.operations.records if op.replacement is not None}
        previews = preview_tool_outputs(self._current_messages, self._policy.max_tool_output_chars)
        return tuple(render_message(m, origins.get(m["message_id"])) for m in previews)

    def _trigger_tokens(self, status: ContextStatus) -> int:
        return int(status.effective_input_limit_tokens * self._policy.auto_compact_ratio)

    def recover_from_ptl(
        self, error: PromptTooLongError | SummaryTooLongError, *, previous_summary: str | None = None,
    ) -> str:
        """Return a complete recovery summary; later rounds only refine that summary."""
        if self._model_limits is None:
            raise ValueError("PTL recovery requires model context limits")
        before = self._estimate_view(
            self._render_current_messages() + self._transient_items, self._tool_definitions,
        )[0].context_status.estimated_input_tokens
        summary = self._observe_compaction(
            self._transient_items, self._tool_definitions, before,
            error=error, previous_summary=previous_summary,
        )
        assert summary is not None
        return summary

    def _observe_compaction(self, transient_items, tool_definitions, before_tokens,
                            *, error=None, previous_summary=None) -> str | None:
        started = monotonic()
        request_count = 0

        def emit(stage, **values):
            if self.event_sink is not None:
                self.event_sink.emit(ContextCompaction(
                    self.turn_id, stage, before_tokens,
                    elapsed_seconds=monotonic() - started, request_count=request_count, **values,
                ))

        def progress(round_number, chunk_number, chunk_count, response):
            nonlocal request_count
            if response is None:
                request_count += 1
            usage = response.usage if response is not None else None
            emit("chunk_started" if response is None else "chunk_completed",
                 round_number=round_number, chunk_number=chunk_number, chunk_count=chunk_count,
                 input_tokens=usage.input_tokens if usage else None,
                 output_tokens=usage.output_tokens if usage else None,
                 model=response.model if response else None)

        emit("started")
        try:
            status, summary = self._compact_context(
                transient_items, tool_definitions, progress,
                on_fallback=lambda failure: emit("fallback", error=str(failure)),
                error=error, previous_summary=previous_summary,
            )
        except ExecutionStopped:
            emit("cancelled")
            raise
        except Exception as error:
            emit("failed", error=str(error))
            raise
        emit("completed", after_tokens=status.estimated_input_tokens)
        return summary

    def _compact_context(self, transient_items, tool_definitions, on_progress,
                         *, on_fallback, error, previous_summary) -> tuple[ContextStatus, str | None]:
        """Trim every tool body, then summarize all history except the latest user request."""
        force_summary = error is not None
        messages = self._current_messages
        pending = set()
        for message in messages:
            payload = message["payload"]
            if payload.get("type") == "function_call":
                pending.add(payload.get("call_id"))
            elif payload.get("type") == "function_call_output":
                pending.discard(payload.get("call_id"))
        if pending:
            raise ValueError("cannot compact before pending tool results")
        latest_user = next((m for m in reversed(messages) if m["payload"].get("role") == "user"), None)
        kept = [m for m in messages if m is latest_user or m["payload"].get("role") in {"system", "developer"}]
        selected = [m for m in messages if m not in kept]
        selected_ids = {m["message_id"] for m in selected}
        trimmed = trim_tool_messages(messages)
        updates = tuple(new for old, new in zip(messages, trimmed) if new != old)
        if updates or force_summary:
            self._input_token_anchor = None
        items = tuple(render_message(m) for m in trimmed) + tuple(transient_items)
        status = self._estimate_view(items, tool_definitions)[0].context_status

        def save_trim():
            if updates:
                self.operations.record(ContextOperation(
                    self.operations.next_id, "trim", tuple(m["message_id"] for m in updates), updates=updates,
                ))
            return status, None

        if not force_summary:
            history_tokens = self._token_estimator.estimate_request(
                instructions="", tools=(),
                input_items=tuple(render_message(m) for m in trimmed if m["message_id"] in selected_ids),
            )
            if (history_tokens <= int(self._model_limits.context_window_tokens * self._policy.engineering_sufficiency_ratio)
                    and status.estimated_input_tokens < self._trigger_tokens(status)):
                return save_trim()
        try:
            try:
                return self._summarize_context(
                    selected, kept, transient_items, tool_definitions, on_progress,
                    error=error, previous_summary=previous_summary,
                )
            except SummaryTooLongError as oversized:
                if force_summary:
                    raise  # Main PTL recovery owns its two rounds in the request loop.
                return self._summarize_context(
                    selected, kept, transient_items, tool_definitions, on_progress,
                    error=oversized, previous_summary=oversized.summary,
                )
        except ModelClientError as failure:
            if force_summary or status.estimated_input_tokens >= self._trigger_tokens(status):
                raise
            # Automatic compaction can continue with a complete trimmed request.
            # Provider PTL recovery must still produce a valid summary.
            on_fallback(failure)
            return save_trim()

    def _summarize_context(self, selected, kept, transient_items, tool_definitions, on_progress,
                           *, error, previous_summary) -> tuple[ContextStatus, str]:
        if self._summarizer is None:
            raise ValueError("context requires compression but no summary model is configured")
        if not selected:
            raise ValueError("current user request and fixed context cannot be compacted; shorten the input")
        ids = tuple(m["message_id"] for m in selected)
        header = (f"Conversation summary of {len(ids)} records (first {ids[0]}, last {ids[-1]}). "
                  "Use search_history/read_history for original details.\n")
        replacement = {"message_id": f"m{uuid4().hex}",
                       "payload": {"role": "assistant", "content": header}}

        def estimate(summary=""):
            replacement["payload"]["content"] = header + summary
            items = (render_message(replacement, self.operations.next_id),
                     *(render_message(m) for m in kept), *transient_items)
            return self._estimate_view(items, tool_definitions, calibrate=False)[0].context_status

        self._input_token_anchor = None
        fixed = estimate()
        room = self._trigger_tokens(fixed) - fixed.estimated_input_tokens - 1
        output_limit = self._policy.summary_output_tokens
        if self._model_limits.max_output_tokens is not None:
            output_limit = min(output_limit, self._model_limits.max_output_tokens)
        target_tokens = min(room, int(self._model_limits.context_window_tokens * self._policy.engineering_sufficiency_ratio))
        if target_tokens < 1:
            raise SummaryError("fixed context leaves insufficient summary space; shorten the current input or fixed context")
        sources = preview_tool_outputs(selected, self._policy.max_tool_output_chars)
        if previous_summary is not None:
            previous_status = estimate(previous_summary)
            sources = [{"role": "assistant", "content": previous_summary}]
            summary_tokens = self._token_estimator.estimate_request(
                instructions="", input_items=sources, tools=(),
            )
            required = error.required_reduction
            if required is None:
                required = max(0, previous_status.estimated_input_tokens - self._trigger_tokens(previous_status) + 1)
            if summary_tokens <= required:
                raise SummaryError("fixed context is too large; even removing the entire summary cannot fit the request")
            target_tokens = min(max(1, target_tokens // 2), max(1, summary_tokens // 2), summary_tokens - required)
        summary = self._summarizer.summarize(
            sources, "", on_progress=on_progress,
            reference_context=self._summary_context,
            target_tokens=target_tokens, max_output_tokens=output_limit,
        )
        if not summary.strip():
            raise SummaryError("summary model returned an empty summary")
        summary = summary.strip()
        status = estimate(summary)
        if status.estimated_input_tokens >= self._trigger_tokens(status):
            raise SummaryTooLongError(summary, status.estimated_input_tokens - self._trigger_tokens(status) + 1)
        self.operations.record(ContextOperation(
            self.operations.next_id, "compress", ids, replacement=replacement, prepend=True,
        ))
        return status, summary

    def _estimate_view(
        self,
        base_items: Sequence[object],
        tool_definitions: Sequence[dict[str, object]],
        *, calibrate: bool = True,
    ) -> tuple[ModelContextView, int]:
        """计入最终提醒正文，跨提醒等级时重新核算新增的文字。"""
        status, _ = self._estimate_status(
            base_items,
            tool_definitions, calibrate=calibrate,
        )
        while True:
            marker = build_context_status_context(status, self._policy)
            items = (*base_items, marker)
            measured, heuristic_input_tokens = self._estimate_status(
                items, tool_definitions, calibrate=calibrate,
            )
            # A rounded percentage can shorten its own text by a character; retain
            # the conservative estimate so boundary rounding cannot oscillate.
            status = self._policy.assess(self._model_limits, max(
                status.estimated_input_tokens, measured.estimated_input_tokens,
            ))
            if build_context_status_context(status, self._policy) == marker:
                break
        return ModelContextView(
            items=items,
            context_status=status,
        ), heuristic_input_tokens

    def _select(
        self, message_ids: list[str],
    ) -> list[dict[str, object]]:
        if not isinstance(message_ids, list) or not message_ids or any(
            not isinstance(mid, str) or not mid for mid in message_ids
        ):
            raise ValueError("message_ids must be a non-empty list of strings")
        targets = set(message_ids)
        messages = self._current_messages
        by_id = {message["message_id"]: message for message in messages}
        # 同一轮内不重新生成视图：已经处理过的目标需要等下一轮再选择。
        changed_ids = set()
        for operation in self.operations.records[self._view_operation_count:]:
            changed_ids.update(operation.message_ids)
        for mid in message_ids:
            if mid not in by_id:
                raise ValueError(f"message is not in the current view: {mid}")
            if mid in changed_ids:
                raise ValueError(f"message {mid} was already changed this turn; wait for the next view")
            if not is_editable(by_id[mid]["payload"]):
                raise ValueError(
                    f"message {mid} cannot be processed: select user/assistant text "
                    "or a text tool result"
                )
        return [message for message in messages if message["message_id"] in targets]

    def change_messages(
        self, action: str, message_ids: list[str], *, instructions: str | None = None,
        summary: str | None = None,
    ) -> str:
        """读取本轮可见消息，生成替代内容并保存操作；下一轮才应用。"""
        if action not in {"delete", "compress"}:
            raise ValueError("unsupported context operation")
        if action == "compress":
            if (instructions is None) == (summary is None):
                raise ValueError("provide exactly one of summary or instructions")
            supplied = summary if summary is not None else instructions
            if not isinstance(supplied, str) or not supplied.strip():
                raise ValueError("summary or instructions must be non-empty text")
        elif summary is not None or instructions is not None:
            raise ValueError("delete does not accept summary or instructions")
        selected = self._select(message_ids)
        if action == "compress":
            # reasoning 不供摘要模型阅读，但整组输出完成压缩后一起移出视图。
            selected_ids = {message["message_id"] for message in selected}
            selected_ids.update(compression_reasoning_ids(self._current_messages, selected_ids))
            selected = [message for message in self._current_messages
                        if message["message_id"] in selected_ids]
            calls = {
                message["payload"].get("call_id"): message["payload"]
                for message in self._current_messages
                if message["payload"].get("type") == "function_call"
            }
            sources = []
            for message in selected:
                source = {"message_id": message["message_id"], "message": message["payload"]}
                if message["payload"].get("type") == "function_call_output":
                    source["tool_call"] = calls.get(message["payload"].get("call_id"))
                sources.append(source)
            if summary is None:
                if self._summarizer is None:
                    raise ValueError("summary model is not configured")
                summary = self._summarizer.summarize(
                    sources, instructions, reference_context=self._summary_context,
                    max_output_tokens=self._policy.summary_output_tokens,
                )
                if not isinstance(summary, str) or not summary.strip():
                    raise ValueError("summary model returned an empty summary")
                summary = summary.strip()

        operation_id = self.operations.next_id
        selected_ids = tuple(message["message_id"] for message in selected)
        content = (
            f"Messages {', '.join(selected_ids)} summarized:\n{summary}"
            if action == "compress"
            else f"Messages {', '.join(selected_ids)} hidden."
        )
        replacement = {
            "message_id": f"m{uuid4().hex}",
            "payload": {"role": "assistant", "content": content},
        }
        operation = ContextOperation(
            operation_id, action, selected_ids,
            instructions=instructions or "", replacement=replacement,
        )
        if action == "compress" and self._model_limits is not None:
            candidate = ContextOperations(records=[*self.operations.records, operation])
            projected = preview_tool_outputs(
                candidate.project(self._history_snapshot), self._policy.max_tool_output_chars,
            )
            origins = {op.replacement["message_id"]: op.operation_id
                       for op in candidate.records if op.replacement is not None}
            items = tuple(render_message(m, origins.get(m["message_id"])) for m in projected)
            # 摘要改变输入结构，用新的完整请求估算，避免沿用旧校准偏差。
            status = self._estimate_view(
                (*items, *self._transient_items), self._tool_definitions, calibrate=False,
            )[0].context_status
            if status.estimated_input_tokens > status.effective_input_limit_tokens:
                raise ValueError("compacted context still exceeds the budget; no operation was saved")
        self.operations.record(operation)
        return json.dumps({
            "operation_id": operation.operation_id,
            "message_id": replacement["message_id"],
        })

    def _estimate_status(
        self,
        items: Sequence[object],
        tool_definitions: Sequence[dict[str, object]],
        *, calibrate: bool = True,
    ) -> tuple[ContextStatus, int]:
        assert self._model_limits is not None
        heuristic_input_tokens = self._token_estimator.estimate_request(
            instructions=self._instructions,
            input_items=items,
            tools=tool_definitions,
        )
        estimated_input_tokens = self._calibrate(heuristic_input_tokens) if calibrate else heuristic_input_tokens
        return (
            self._policy.assess(
                self._model_limits,
                estimated_input_tokens,
            ),
            heuristic_input_tokens,
        )

    def _calibrate(self, heuristic_input_tokens: int) -> int:
        """用真实用量补偿低估，但不让上一轮的高估降低本轮预算。"""
        anchor = self._input_token_anchor
        if anchor is None:
            return heuristic_input_tokens

        estimated_new_tokens = (
            heuristic_input_tokens - anchor.heuristic_input_tokens
        )
        return max(anchor.actual_input_tokens + estimated_new_tokens, heuristic_input_tokens)
