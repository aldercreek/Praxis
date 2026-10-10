import pytest

from praxis_agent.config import LangfuseConfig, ModelConfig


def test_model_config_loads_environment_values() -> None:
    config = ModelConfig.from_env(
        {
            "ADA_API_KEY": "secret-key",
            "ADA_MODEL": "example-model",
            "ADA_BASE_URL": "https://example.com/v1",
        }
    )

    assert config.api_key == "secret-key"
    assert config.model == "example-model"
    assert config.base_url == "https://example.com/v1"


def test_model_config_uses_ada_defaults() -> None:
    config = ModelConfig.from_env({"ADA_API_KEY": "secret-key"})

    assert config.model == "gpt-5.6-terra-2026-07-09"
    assert (
        config.base_url
        == "https://ada-cli-go.ctripcorp.com/coding-plan/openai/v1"
    )


def test_model_config_uses_defaults_for_empty_overrides() -> None:
    config = ModelConfig.from_env(
        {
            "ADA_API_KEY": "secret-key",
            "ADA_MODEL": "   ",
            "ADA_BASE_URL": "   ",
        }
    )

    assert config.model == "gpt-5.6-terra-2026-07-09"
    assert (
        config.base_url
        == "https://ada-cli-go.ctripcorp.com/coding-plan/openai/v1"
    )


def test_model_config_requires_ada_api_key() -> None:
    with pytest.raises(ValueError, match="ADA_API_KEY"):
        ModelConfig.from_env({})


def test_model_config_repr_hides_api_key() -> None:
    config = ModelConfig(api_key="secret-key", model="example-model")

    assert "secret-key" not in repr(config)


def test_langfuse_config_is_disabled_without_keys() -> None:
    assert LangfuseConfig.from_env({}) is None


def test_langfuse_config_uses_company_defaults() -> None:
    config = LangfuseConfig.from_env(
        {
            "LANGFUSE_PUBLIC_KEY": "public-key",
            "LANGFUSE_SECRET_KEY": "secret-key",
        }
    )

    assert config is not None
    assert config.base_url == "https://langfuse.flight.ctripcorp.com"
    assert config.environment == "development"


def test_langfuse_config_loads_environment_overrides() -> None:
    config = LangfuseConfig.from_env(
        {
            "LANGFUSE_PUBLIC_KEY": "public-key",
            "LANGFUSE_SECRET_KEY": "secret-key",
            "LANGFUSE_BASE_URL": "https://langfuse.example.com",
            "LANGFUSE_TRACING_ENVIRONMENT": "test",
        }
    )

    assert config is not None
    assert config.base_url == "https://langfuse.example.com"
    assert config.environment == "test"


@pytest.mark.parametrize(
    "environ, missing_name",
    [
        ({"LANGFUSE_SECRET_KEY": "secret-key"}, "LANGFUSE_PUBLIC_KEY"),
        ({"LANGFUSE_PUBLIC_KEY": "public-key"}, "LANGFUSE_SECRET_KEY"),
    ],
)
def test_langfuse_config_requires_both_keys(
    environ: dict[str, str],
    missing_name: str,
) -> None:
    with pytest.raises(ValueError, match=missing_name):
        LangfuseConfig.from_env(environ)


def test_langfuse_config_repr_hides_keys() -> None:
    config = LangfuseConfig(
        public_key="public-key",
        secret_key="secret-key",
    )

    assert "public-key" not in repr(config)
    assert "secret-key" not in repr(config)
