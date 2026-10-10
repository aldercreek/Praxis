from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from praxis_agent.context.model_catalog import ModelLimits
from praxis_agent.context.budgets import DEFAULT_BUDGETS, context_budgets


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

    desired_output_tokens: int = DEFAULT_BUDGETS["desired_output_tokens"]
    safety_ratio: float = 0.95
    soft_ratio: float = 0.65
    hard_ratio: float = 0.80
    emergency_ratio: float = 0.92
    # 仅限制模型视图中的单条工具结果正文，完整历史仍保留原文。
    max_tool_output_chars: int = DEFAULT_BUDGETS["max_tool_output_chars"]
    summary_target_tokens: int = DEFAULT_BUDGETS["summary_target_tokens"]
    summary_max_output_tokens: int = DEFAULT_BUDGETS["summary_max_output_tokens"]
    max_file_read_chars: int = DEFAULT_BUDGETS["max_file_read_chars"]
    max_history_read_chars: int = DEFAULT_BUDGETS["max_history_read_chars"]

    @classmethod
    def for_model(cls, limits: ModelLimits, **overrides: int | float) -> ContextPolicy:
        values = context_budgets(limits.context_window_tokens)
        values.update(overrides)
        return cls(**values)

    def __post_init__(self) -> None:
        for name in DEFAULT_BUDGETS:
            if name != "desired_output_tokens" and getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
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
    """按实际文本做字符换算；安全余量由 ContextPolicy 单独预留。"""

    ascii_characters_per_token: float = 3.0
    non_ascii_tokens_per_character: float = 1.0

    def estimate_request(
        self,
        *,
        instructions: str,
        input_items: Sequence[object],
        tools: Sequence[dict[str, object]],
    ) -> int:
        return max(1, sum(self.estimate_breakdown(
            instructions=instructions, input_items=input_items, tools=tools,
        ).values()))

    def estimate_breakdown(
        self,
        *,
        instructions: str,
        input_items: Sequence[object],
        tools: Sequence[dict[str, object]],
    ) -> dict[str, int]:
        """拆分估算来源便于诊断；不序列化整个请求，避免额外转义。"""
        parts: dict[str, list[str]] = {
            "body": [instructions], "reasoning": [], "tool_definitions": [],
            "tool_arguments": [], "tool_results": [],
        }
        for item in input_items:
            payload = item if isinstance(item, Mapping) else _json_default(item)
            if not isinstance(payload, Mapping):
                continue
            kind = payload.get("type")
            if kind == "reasoning":
                parts["reasoning"].extend(_content_text(payload.get("summary")))
                parts["reasoning"].extend(_content_text(payload.get("content")))
            elif kind == "function_call":
                parts["tool_arguments"].append(_content_json(payload.get("name")))
                parts["tool_arguments"].append(_content_json(payload.get("arguments")))
            elif kind == "function_call_output":
                output = payload.get("output")
                if isinstance(output, list):
                    parts["tool_results"].extend(_content_text(output))
                else:
                    parts["tool_results"].append(_content_json(output))
            else:
                parts["body"].extend(_content_text(payload.get("content")))
            # 某些兼容服务将 thinking 放在调用或消息上，按实际回传内容计入。
            parts["reasoning"].extend(_content_text(payload.get("reasoning_content")))
        for tool in tools:
            definition = tool.get("function", tool)
            for key in ("name", "description", "parameters"):
                parts["tool_definitions"].append(_content_json(definition.get(key)))
        result = {
            category: math.ceil(sum(self._estimate_text(text) for text in texts))
            for category, texts in parts.items()
        }
        # 角色、消息边界和工具定义边界的少量固定开销，并非协议精确 token 数。
        result["overhead"] = 4 * (bool(instructions) + len(input_items) + len(tools))
        return result

    def _estimate_text(self, text: str) -> float:
        non_ascii_count = sum(ord(character) > 127 for character in text)
        ascii_count = len(text) - non_ascii_count
        return (
            ascii_count / self.ascii_characters_per_token
            + non_ascii_count * self.non_ascii_tokens_per_character
        )


def _content_json(value: object) -> str:
    """原本就是文本的 JSON 不再序列化；结构化参数/schema 只序列化一次。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_json_default)


def _content_text(content: object) -> list[str]:
    """提取消息内容块的文本，不统计 id/status 或图片等非文本协议字段。"""
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [block["text"] for block in content
                if isinstance(block, Mapping) and isinstance(block.get("text"), str)]
    return []


def _json_default(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return str(value)
