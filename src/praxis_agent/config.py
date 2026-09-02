import os
from collections.abc import Mapping
from dataclasses import dataclass, field


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
        api_key = values.get("OPENAI_API_KEY", "").strip()
        model = values.get("PRAXIS_MODEL", "").strip()

        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not model:
            raise ValueError("PRAXIS_MODEL environment variable is required")

        base_url = values.get("OPENAI_BASE_URL", "").strip() or None
        return cls(api_key=api_key, model=model, base_url=base_url)
