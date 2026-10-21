from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class MetadataSource(StrEnum):
    """表示模型容量元数据的来源。"""

    USER_OVERRIDE = "user_override"
    PROVIDER = "provider"
    CATALOG = "catalog"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class ModelLimits:
    """描述模型或 Provider 声明的上下文硬限制。"""

    context_window_tokens: int
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    metadata_source: MetadataSource = MetadataSource.FALLBACK

    def __post_init__(self) -> None:
        if self.context_window_tokens < 1:
            raise ValueError("context_window_tokens must be at least 1")
        if self.max_input_tokens is not None and self.max_input_tokens < 1:
            raise ValueError("max_input_tokens must be at least 1")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")


@dataclass(frozen=True, slots=True)
class ModelLimitOverrides:
    """保存用户显式提供的模型容量字段。"""

    context_window_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "context_window_tokens",
            "max_input_tokens",
            "max_output_tokens",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1")

    @property
    def has_values(self) -> bool:
        """返回是否至少配置了一个覆盖字段。"""
        return any(
            value is not None
            for value in (
                self.context_window_tokens,
                self.max_input_tokens,
                self.max_output_tokens,
            )
        )


def _catalog_limits(
    context_window_tokens: int,
    *,
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
) -> ModelLimits:
    return ModelLimits(
        context_window_tokens=context_window_tokens,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        metadata_source=MetadataSource.CATALOG,
    )


# Values are taken from vendor documentation. A field stays None when the
# vendor does not publish a separate limit; the resolver never guesses it.
# OpenAI: https://developers.openai.com/api/docs/models
_OPENAI_LARGE = _catalog_limits(
    1_050_000,
    max_output_tokens=128_000,
)

# Anthropic: https://platform.claude.com/docs/en/models/overview
_CLAUDE_LARGE = _catalog_limits(
    1_000_000,
    max_output_tokens=128_000,
)
_CLAUDE_HAIKU = _catalog_limits(
    200_000,
    max_output_tokens=64_000,
)

# Google publishes independent input and output limits on each model page.
# https://ai.google.dev/gemini-api/docs/models
_GEMINI_LONG = _catalog_limits(
    1_048_576,
    max_input_tokens=1_048_576,
    max_output_tokens=65_536,
)

# DeepSeek: https://api-docs.deepseek.com/quick_start/pricing
# Verified 2026-09-17: deepseek-flash serves V4.1-Flash (1M context,
# 384K max output). Legacy Flash names route to the same model.
# ADA exposes this version as deepseek-v4.1-flash.
_DEEPSEEK_V4 = _catalog_limits(
    1_000_000,
    max_output_tokens=384_000,
)

# Qwen: https://help.aliyun.com/zh/model-studio/text-generation-model
_QWEN_LONG = _catalog_limits(1_000_000)
_QWEN_256K = _catalog_limits(256_000)


MODEL_LIMIT_CATALOG: Mapping[str, ModelLimits] = MappingProxyType(
    {
        # OpenAI
        "gpt-6-astra": _OPENAI_LARGE,
        "gpt-5.6": _OPENAI_LARGE,
        "gpt-5.6-sol": _OPENAI_LARGE,
        "gpt-5.6-terra": _OPENAI_LARGE,
        "gpt-5.6-luna": _OPENAI_LARGE,
        "gpt-5.5": _OPENAI_LARGE,
        # Anthropic
        "claude-fable-5-1": _CLAUDE_LARGE,
        "claude-opus-5": _CLAUDE_LARGE,
        "claude-sonnet-5": _CLAUDE_LARGE,
        "claude-haiku-4-5": _CLAUDE_HAIKU,
        # Google
        "gemini-3.8-flash": _GEMINI_LONG,
        "gemini-3.6-flash": _GEMINI_LONG,
        "gemini-3.5-flash": _GEMINI_LONG,
        "gemini-3.5-flash-lite": _GEMINI_LONG,
        "gemini-3.1-pro-preview": _GEMINI_LONG,
        "gemini-3.1-pro-preview-customtools": _GEMINI_LONG,
        "gemini-3-flash-preview": _GEMINI_LONG,
        "gemini-2.5-pro": _GEMINI_LONG,
        "gemini-2.5-flash": _GEMINI_LONG,
        "gemini-2.5-flash-lite": _GEMINI_LONG,
        # DeepSeek
        "deepseek-flash": _DEEPSEEK_V4,
        "deepseek-v4.1-flash": _DEEPSEEK_V4,
        "deepseek-v4-flash": _DEEPSEEK_V4,
        "deepseek-v4-flash-vision-exp": _DEEPSEEK_V4,
        "deepseek-v4-pro": _DEEPSEEK_V4,
        # Qwen
        "qwen3.8-max": _QWEN_LONG,
        "qwen3.8-flash": _QWEN_LONG,
        "qwen3.7-plus": _QWEN_LONG,
        "qwen3.7-flash": _QWEN_LONG,
        "qwen3.6-plus": _QWEN_LONG,
        "qwen3.6-flash": _QWEN_LONG,
        "qwen3.5-plus": _QWEN_LONG,
        "qwen3.5-flash": _QWEN_LONG,
        "qwen3-coder-plus": _QWEN_LONG,
        "qwen3-coder-flash": _QWEN_LONG,
        "qwen-plus": _QWEN_LONG,
        "qwen-flash": _QWEN_LONG,
        "qwen3-coder-next": _QWEN_256K,
        "qwen3-coder-480b-a35b-instruct": _QWEN_256K,
        "qwen3-coder-30b-a3b-instruct": _QWEN_256K,
    }
)

DEFAULT_MODEL_LIMITS = ModelLimits(context_window_tokens=128_000)


@dataclass(frozen=True, slots=True)
class ModelInputCapabilities:
    """Native model inputs; the active API can impose narrower restrictions."""

    image_mime_types: tuple[str, ...] = ()
    pdf: bool = False
    audio: bool = False


_IMAGES = ("image/png", "image/jpeg", "image/webp", "image/gif")
_VISION = ModelInputCapabilities(_IMAGES)
_VISION_PDF = ModelInputCapabilities(_IMAGES, pdf=True)
_GEMINI_INPUT = ModelInputCapabilities(
    ("image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"), pdf=True, audio=True,
)
_TEXT_INPUT = ModelInputCapabilities()

# Checked 2026-10-21. Exact model entries, not assumptions based on provider prefixes.
# https://developers.openai.com/api/docs/models
# https://platform.claude.com/docs/en/about-claude/models/overview
# https://ai.google.dev/gemini-api/docs/models
# https://api-docs.deepseek.com/quick_start/pricing
# https://help.aliyun.com/zh/model-studio/vision
# https://models.dev/api.json (cross-check of model-specific modalities)
MODEL_INPUT_CATALOG: Mapping[str, ModelInputCapabilities] = MappingProxyType({
    **dict.fromkeys((
        "gpt-6-astra", "gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5",
        "claude-fable-5-1", "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5",
        "qwen3.8-max",
    ), _VISION_PDF),
    **dict.fromkeys((
        "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview", "gemini-3.1-pro-preview-customtools", "gemini-3-flash-preview",
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
    ), _GEMINI_INPUT),
    **dict.fromkeys((
        "deepseek-flash", "deepseek-v4.1-flash", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp",
        "qwen3.8-flash", "qwen3.7-plus", "qwen3.7-flash", "qwen3.6-plus", "qwen3.6-flash",
        "qwen3.5-plus", "qwen3.5-flash",
    ), _VISION),
    **dict.fromkeys((
        "deepseek-v4-pro", "qwen3-coder-plus", "qwen3-coder-flash", "qwen-plus", "qwen-flash",
        "qwen3-coder-next", "qwen3-coder-480b-a35b-instruct", "qwen3-coder-30b-a3b-instruct",
    ), _TEXT_INPUT),
})


def resolve_model_inputs(model: str) -> ModelInputCapabilities:
    """Unknown models and automatic routing default to text, without guessing vision."""
    key = _catalog_model_name(model)
    return MODEL_INPUT_CATALOG.get(key, _TEXT_INPUT)

_SNAPSHOT_SUFFIXES = (
    re.compile(r"-\d{4}-\d{2}-\d{2}$"),
    re.compile(r"-\d{8}$"),
    re.compile(r"-\d{4}$"),
)


def resolve_model_limits(
    model: str,
    overrides: ModelLimitOverrides | None = None,
) -> ModelLimits:
    """按用户覆盖、本地目录、保守默认值的顺序解析容量。"""
    base_limits = _find_catalog_limits(model) or DEFAULT_MODEL_LIMITS
    if overrides is None or not overrides.has_values:
        return base_limits

    return ModelLimits(
        context_window_tokens=(
            overrides.context_window_tokens
            if overrides.context_window_tokens is not None
            else base_limits.context_window_tokens
        ),
        max_input_tokens=(
            overrides.max_input_tokens
            if overrides.max_input_tokens is not None
            else base_limits.max_input_tokens
        ),
        max_output_tokens=(
            overrides.max_output_tokens
            if overrides.max_output_tokens is not None
            else base_limits.max_output_tokens
        ),
        metadata_source=MetadataSource.USER_OVERRIDE,
    )


def _find_catalog_limits(model: str) -> ModelLimits | None:
    return MODEL_LIMIT_CATALOG.get(_catalog_model_name(model))


def _catalog_model_name(model: str) -> str:
    normalized_model = model.strip().casefold().rsplit("/", 1)[-1]
    if normalized_model in MODEL_LIMIT_CATALOG:
        return normalized_model

    for suffix in _SNAPSHOT_SUFFIXES:
        base_model = suffix.sub("", normalized_model)
        if base_model != normalized_model:
            return base_model
    return normalized_model
