from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
    ) -> None:
        self._model_limits = model_limits
        self._policy = policy or ContextPolicy()
        self._instructions = instructions
        self._token_estimator = token_estimator or HeuristicTokenEstimator()
        self._input_token_anchor: _InputTokenAnchor | None = None
        self._pending_heuristic_input_tokens: int | None = None

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
        """清除不再适用于新会话的输入 Token 校准状态。"""
        self._input_token_anchor = None
        self._pending_heuristic_input_tokens = None

    def build_view(
        self,
        history: ConversationHistory,
        *,
        transient_items: Sequence[object] = (),
        tool_definitions: Sequence[dict[str, object]] = (),
    ) -> ModelContextView:
        """构建完整历史、运行时状态和可选容量状态组成的视图。"""
        base_items = (*history.snapshot(), *transient_items)
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
        self._pending_heuristic_input_tokens = heuristic_input_tokens
        return ModelContextView(
            items=items,
            context_status=final_status,
        )

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
