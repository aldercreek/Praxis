from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from praxis_agent.context.capacity import (
    ContextPolicy,
    ContextStatus,
    HeuristicTokenEstimator,
    TokenEstimator,
)
from praxis_agent.context.model_catalog import ModelLimits
from praxis_agent.context.model_context import build_context_status_context
from praxis_agent.context.models import (
    ConversationHistory,
    ModelContextView,
)
from praxis_agent.context.operations import (
    ContextOperation,
    ContextOperations,
    is_editable,
    render_message,
)
from praxis_agent.context.summarizer import ContextSummarizer


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
        self._policy = policy or ContextPolicy()
        self._instructions = instructions
        self._token_estimator = token_estimator or HeuristicTokenEstimator()
        self._input_token_anchor: _InputTokenAnchor | None = None
        self._pending_heuristic_input_tokens: int | None = None
        self.operations = ContextOperations()
        self._summarizer = summarizer
        self._current_messages: list[dict[str, object]] = []
        self._view_operation_count = 0

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

    def build_view(
        self,
        history: ConversationHistory,
        *,
        transient_items: Sequence[object] = (),
        tool_definitions: Sequence[dict[str, object]] = (),
    ) -> ModelContextView:
        """构建完整历史、运行时状态和可选容量状态组成的视图。"""
        if self._view_operation_count != len(self.operations.records):
            self.reset()
        self._current_messages = self.operations.project(history.snapshot())
        self._view_operation_count = len(self.operations.records)
        origins = {
            op.replacement["message_id"]: op.operation_id
            for op in self.operations.records if op.replacement is not None
        }
        rendered = tuple(
            render_message(message, origins.get(message["message_id"]))
            for message in self._current_messages
        )
        base_items = (*rendered, *transient_items)
        if self._model_limits is None:
            self._pending_heuristic_input_tokens = None
            return ModelContextView(items=base_items)

        initial_status, _ = self._estimate_status(
            base_items,
            tool_definitions,
        )
        items_with_status = (
            *base_items,
            build_context_status_context(initial_status),
        )
        final_status, heuristic_input_tokens = self._estimate_status(
            items_with_status,
            tool_definitions,
        )

        # 使用完整请求的结果更新容量消息，不再为这条小消息循环估算。
        items = (*base_items, build_context_status_context(final_status))
        if final_status.estimated_input_tokens > final_status.effective_input_limit_tokens:
            raise ValueError("model input exceeds the context budget; shorten the input or start a new session")
        self._pending_heuristic_input_tokens = heuristic_input_tokens
        return ModelContextView(
            items=items,
            context_status=final_status,
        )

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
        self, action: str, message_ids: list[str], *, instructions: str = "",
    ) -> str:
        """读取本轮可见消息，生成替代内容并保存操作；下一轮才应用。"""
        if action not in {"delete", "compress"}:
            raise ValueError("unsupported context operation")
        selected = self._select(message_ids)
        summary = ""
        if action == "compress":
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError("compression instructions must not be empty")
            if self._summarizer is None:
                raise ValueError("summary model is not configured")
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
            summary = self._summarizer.summarize(sources, instructions)
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
            instructions=instructions, replacement=replacement,
        )
        self.operations.record(operation)
        return json.dumps({
            "operation_id": operation.operation_id,
            "message_id": replacement["message_id"],
        })

    def _estimate_status(
        self,
        items: Sequence[object],
        tool_definitions: Sequence[dict[str, object]],
    ) -> tuple[ContextStatus, int]:
        assert self._model_limits is not None
        heuristic_input_tokens = self._token_estimator.estimate_request(
            instructions=self._instructions,
            input_items=items,
            tools=tool_definitions,
        )
        estimated_input_tokens = self._calibrate(heuristic_input_tokens)
        return (
            self._policy.assess(
                self._model_limits,
                estimated_input_tokens,
            ),
            heuristic_input_tokens,
        )

    def _calibrate(self, heuristic_input_tokens: int) -> int:
        """以上一轮真实输入量加本轮估算增量得到当前估算值。"""
        anchor = self._input_token_anchor
        if anchor is None:
            return heuristic_input_tokens

        estimated_new_tokens = (
            heuristic_input_tokens - anchor.heuristic_input_tokens
        )
        return max(anchor.actual_input_tokens + estimated_new_tokens, 0)
