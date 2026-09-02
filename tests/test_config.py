import pytest

from praxis_agent.config import ModelConfig


def test_model_config_loads_environment_values() -> None:
    config = ModelConfig.from_env(
        {
            "OPENAI_API_KEY": "secret-key",
            "PRAXIS_MODEL": "example-model",
            "OPENAI_BASE_URL": "https://example.com/v1",
        }
    )

    assert config.api_key == "secret-key"
    assert config.model == "example-model"
    assert config.base_url == "https://example.com/v1"


def test_model_config_treats_empty_base_url_as_missing() -> None:
    config = ModelConfig.from_env(
        {
            "OPENAI_API_KEY": "secret-key",
            "PRAXIS_MODEL": "example-model",
            "OPENAI_BASE_URL": "   ",
        }
    )

    assert config.base_url is None


@pytest.mark.parametrize("missing_name", ["OPENAI_API_KEY", "PRAXIS_MODEL"])
def test_model_config_requires_environment_values(missing_name: str) -> None:
    environ = {
        "OPENAI_API_KEY": "secret-key",
        "PRAXIS_MODEL": "example-model",
    }
    del environ[missing_name]

    with pytest.raises(ValueError, match=missing_name):
        ModelConfig.from_env(environ)


def test_model_config_repr_hides_api_key() -> None:
    config = ModelConfig(api_key="secret-key", model="example-model")

    assert "secret-key" not in repr(config)
