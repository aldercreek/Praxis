import os
from collections.abc import Mapping
from dataclasses import dataclass, field


_DEFAULT_ADA_BASE_URL = (
    "https://ada-cli-go.ctripcorp.com/coding-plan/openai/v1"
)
_DEFAULT_ADA_MODEL = "gpt-5.6-terra-2026-07-09"
_DEFAULT_LANGFUSE_BASE_URL = "https://langfuse.flight.ctripcorp.com"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """保存创建模型客户端所需的配置。"""

    # repr=False 可以防止调试打印配置对象时泄露密钥。
    api_key: str = field(repr=False)
    model: str
    base_url: str | None = None

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "ModelConfig":
        """从环境变量读取并校验模型配置。"""
        # 参数主要用于测试；正常运行时直接读取当前进程的环境变量。
        values = os.environ if environ is None else environ
        api_key = values.get("ADA_API_KEY", "").strip()

        if not api_key:
            raise ValueError("ADA_API_KEY environment variable is required")

        # 模型和网关提供 ADA 默认值，需要时仍可通过环境变量覆盖。
        model = values.get("ADA_MODEL", "").strip() or _DEFAULT_ADA_MODEL
        base_url = (
            values.get("ADA_BASE_URL", "").strip() or _DEFAULT_ADA_BASE_URL
        )
        return cls(api_key=api_key, model=model, base_url=base_url)


@dataclass(frozen=True, slots=True)
class LangfuseConfig:
    """保存携程内部 Langfuse 客户端所需的配置。"""

    public_key: str = field(repr=False)
    secret_key: str = field(repr=False)
    base_url: str = _DEFAULT_LANGFUSE_BASE_URL
    environment: str = "development"

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "LangfuseConfig | None":
        """读取 Langfuse 配置；没有配置密钥时表示不启用观测。"""
        values = os.environ if environ is None else environ
        public_key = values.get("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = values.get("LANGFUSE_SECRET_KEY", "").strip()

        if not public_key and not secret_key:
            return None
        if not public_key:
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY is required when Langfuse is configured"
            )
        if not secret_key:
            raise ValueError(
                "LANGFUSE_SECRET_KEY is required when Langfuse is configured"
            )

        base_url = (
            values.get("LANGFUSE_BASE_URL", "").strip()
            or _DEFAULT_LANGFUSE_BASE_URL
        )
        environment = (
            values.get("LANGFUSE_TRACING_ENVIRONMENT", "").strip()
            or "development"
        )
        return cls(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            environment=environment,
        )
