import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from praxis_agent.context.capacity import ContextPolicy
from praxis_agent.context.budgets import DEFAULT_BUDGETS
from praxis_agent.context.model_catalog import (
    ModelLimitOverrides,
    ModelLimits,
    ModelInputCapabilities,
    resolve_model_inputs,
    resolve_model_limits,
)


_DEFAULT_ADA_BASE_URL = (
    "https://ada-cli-go.ctripcorp.com/coding-plan/openai/v1"
)
_DEFAULT_ADA_MODEL = "auto(free)"
DEFAULT_MAX_PARALLEL_TOOLS = 4


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """保存模型客户端及其上下文管理策略所需的配置。"""

    # repr=False 可以防止调试打印配置对象时泄露密钥。
    api_key: str = field(repr=False)
    model: str
    base_url: str | None = None
    model_limit_overrides: ModelLimitOverrides = field(
        default_factory=ModelLimitOverrides,
        repr=False,
    )
    context_policy: ContextPolicy | None = None
    max_parallel_tools: int = DEFAULT_MAX_PARALLEL_TOOLS
    input_capabilities: ModelInputCapabilities | None = None

    def __post_init__(self) -> None:
        if self.input_capabilities is None:
            object.__setattr__(self, "input_capabilities", resolve_model_inputs(self.model))
        if self.max_parallel_tools < 1:
            raise ValueError("max_parallel_tools must be at least 1")
        if self.context_policy is None:
            object.__setattr__(self, "context_policy", ContextPolicy.for_model(self.model_limits))

    @property
    def model_limits(self) -> ModelLimits:
        """解析当前模型最终使用的容量限制。"""
        return resolve_model_limits(self.model, self.model_limit_overrides)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *, model: str | None = None,
    ) -> "ModelConfig":
        """从环境变量读取并校验模型配置。"""
        # 参数主要用于测试；正常运行时直接读取当前进程的环境变量。
        values = os.environ if environ is None else environ
        api_key = values.get("ADA_API_KEY", "").strip()

        if not api_key:
            raise ValueError("ADA_API_KEY environment variable is required")

        # 模型和网关提供 ADA 默认值，需要时仍可通过环境变量覆盖。
        model = model or values.get("ADA_MODEL", "").strip() or _DEFAULT_ADA_MODEL
        base_url = (
            values.get("ADA_BASE_URL", "").strip() or _DEFAULT_ADA_BASE_URL
        )
        model_limit_overrides = ModelLimitOverrides(
            context_window_tokens=_read_optional_integer(
                values,
                "ADA_CONTEXT_WINDOW_TOKENS",
            ),
            max_input_tokens=_read_optional_integer(
                values,
                "ADA_MAX_INPUT_TOKENS",
            ),
            max_output_tokens=_read_optional_integer(
                values,
                "ADA_MAX_OUTPUT_TOKENS",
            ),
        )
        policy_overrides = {}
        for name in DEFAULT_BUDGETS:
            value = _read_optional_integer(values, f"ADA_{name.upper()}")
            if value is not None:
                policy_overrides[name] = value
        context_policy = ContextPolicy.for_model(
            resolve_model_limits(model, model_limit_overrides),
            **policy_overrides,
            safety_ratio=_read_float(
                values,
                "ADA_CONTEXT_SAFETY_RATIO",
                0.95,
            ),
        )
        max_parallel_tools = _read_optional_integer(values, "PRAXIS_MAX_PARALLEL_TOOLS")
        inputs = resolve_model_inputs(model)
        image_input = _read_optional_bool(values, "ADA_MODEL_IMAGE_INPUT")
        pdf_input = _read_optional_bool(values, "ADA_MODEL_PDF_INPUT")
        if image_input is not None:
            inputs = replace(inputs, image_mime_types=(
                (inputs.image_mime_types or ("image/png", "image/jpeg", "image/webp", "image/gif"))
                if image_input else ()
            ))
        if pdf_input is not None:
            inputs = replace(inputs, pdf=pdf_input)
        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            model_limit_overrides=model_limit_overrides,
            context_policy=context_policy,
            input_capabilities=inputs,
            max_parallel_tools=(DEFAULT_MAX_PARALLEL_TOOLS
                                if max_parallel_tools is None else max_parallel_tools),
        )


def _read_optional_bool(values: Mapping[str, str], name: str) -> bool | None:
    raw = values.get(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"true", "1"}:
        return True
    if raw in {"false", "0"}:
        return False
    raise ValueError(f"{name} must be true/false or 1/0")


def _read_optional_integer(
    values: Mapping[str, str],
    name: str,
) -> int | None:
    raw_value = values.get(name, "").strip()
    if not raw_value:
        return None
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _read_float(
    values: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    raw_value = values.get(name, "").strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
