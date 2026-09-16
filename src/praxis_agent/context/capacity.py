from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from praxis_agent.context.model_catalog import ModelLimits


class ContextPressure(StrEnum):
    """表示当前模型输入相对于可用窗口的压力级别。"""

    NORMAL = "normal"
    SOFT = "soft"
    HARD = "hard"
    EMERGENCY = "emergency"


@dataclass(frozen=True, slots=True)
class ContextStatus:
    """保存一次模型请求发送前估算出的上下文容量状态。"""

    estimated_input_tokens: int
    # 有效输入 = context_window * safetyRatio - output_token
    effective_input_limit_tokens: int
    usage_percent: float
    pressure: ContextPressure


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """定义 Praxis 使用模型上下文容量的策略。"""

    desired_output_tokens: int = 16_000
    safety_ratio: float = 0.95
    soft_ratio: float = 0.65
    hard_ratio: float = 0.80
    emergency_ratio: float = 0.92

    def __post_init__(self) -> None:
        if self.desired_output_tokens < 0:
            raise ValueError("desired_output_tokens must not be negative")
        if not 0 < self.safety_ratio <= 1:
            raise ValueError("safety_ratio must be greater than 0 and at most 1")
        if not (
            0 < self.soft_ratio < self.hard_ratio
            < self.emergency_ratio < 1
        ):
            raise ValueError(
                "context pressure ratios must be ordered between 0 and 1"
            )

    def effective_input_limit(self, limits: ModelLimits) -> int:
        """计算应用安全余量和输出预留后的实际输入上限。"""
        output_budget = self.desired_output_tokens
        if limits.max_output_tokens is not None:
            output_budget = min(output_budget, limits.max_output_tokens)

        safe_context_window = math.floor(
            limits.context_window_tokens * self.safety_ratio
        )
        effective_limit = safe_context_window - output_budget
        if limits.max_input_tokens is not None:
            effective_limit = min(effective_limit, limits.max_input_tokens)
        if effective_limit < 1:
            raise ValueError(
                "context policy leaves no capacity for model input"
            )
        return effective_limit

    def assess(
        self,
        limits: ModelLimits,
        estimated_input_tokens: int,
    ) -> ContextStatus:
        """根据输入估算值生成容量状态。"""
        if estimated_input_tokens < 0:
            raise ValueError("estimated_input_tokens must not be negative")

        effective_limit = self.effective_input_limit(limits)
        ratio = estimated_input_tokens / effective_limit
        if ratio >= self.emergency_ratio:
            pressure = ContextPressure.EMERGENCY
        elif ratio >= self.hard_ratio:
            pressure = ContextPressure.HARD
        elif ratio >= self.soft_ratio:
            pressure = ContextPressure.SOFT
        else:
            pressure = ContextPressure.NORMAL

        return ContextStatus(
            estimated_input_tokens=estimated_input_tokens,
            effective_input_limit_tokens=effective_limit,
            usage_percent=round(ratio * 100, 1),
            pressure=pressure,
        )


class TokenEstimator(Protocol):
    """定义模型请求 token 估算器所需的接口。"""

    def estimate_request(
        self,
        *,
        instructions: str,
        input_items: Sequence[object],
        tools: Sequence[dict[str, object]],
    ) -> int:
        """估算一份完整模型请求的输入 token 数。"""
        ...


@dataclass(frozen=True, slots=True)
class HeuristicTokenEstimator:
    """使用保守字符换算估算中英文混合模型输入。"""

    ascii_characters_per_token: float = 3.0
    non_ascii_tokens_per_character: float = 1.0
    safety_factor: float = 1.2

    def estimate_request(
        self,
        *,
        instructions: str,
        input_items: Sequence[object],
        tools: Sequence[dict[str, object]],
    ) -> int:
        request_payload = {
            "instructions": instructions,
            "input": input_items,
            "tools": tools,
        }
        text = json.dumps(
            request_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )
        non_ascii_count = sum(ord(character) > 127 for character in text)
        ascii_count = len(text) - non_ascii_count
        raw_estimate = (
            ascii_count / self.ascii_characters_per_token
            + non_ascii_count * self.non_ascii_tokens_per_character
        )
        return max(1, math.ceil(raw_estimate * self.safety_factor))


def _json_default(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return str(value)
