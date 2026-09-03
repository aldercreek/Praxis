import pytest

from praxis_agent.config import ModelConfig


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
