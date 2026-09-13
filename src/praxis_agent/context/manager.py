from __future__ import annotations

from collections.abc import Sequence

from praxis_agent.context.models import (
    ConversationHistory,
    ModelContextView,
)


class ContextManager:
    """根据完整历史构建一次模型请求使用的上下文视图。"""

    def build_view(
        self,
        history: ConversationHistory,
        *,
        transient_items: Sequence[object] = (),
    ) -> ModelContextView:
        """保留全部历史，并在末尾加入不持久化的运行时上下文。"""
        return ModelContextView(
            items=(*history.snapshot(), *transient_items),
        )
