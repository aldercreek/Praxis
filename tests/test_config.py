import pytest

from praxis_agent.config import ModelConfig


@pytest.mark.parametrize("model", ["gpt-6-astra", "vendor/GPT-6-ASTRA-2026-10-21"])
def test_model_capabilities_resolve_names_and_snapshots(model):
    config = ModelConfig(api_key="test", model=model)
    assert "image/png" in config.input_capabilities.image_mime_types
    assert config.input_capabilities.pdf
    assert not config.input_capabilities.audio


def test_gateway_capability_overrides():
    config = ModelConfig.from_env({"ADA_API_KEY": "test", "ADA_MODEL": "custom-model",
                                   "ADA_MODEL_IMAGE_INPUT": "true", "ADA_MODEL_PDF_INPUT": "1"})
    assert config.input_capabilities.image_mime_types and config.input_capabilities.pdf
    disabled = ModelConfig.from_env({"ADA_API_KEY": "test", "ADA_MODEL": "gpt-6-astra",
                                     "ADA_MODEL_IMAGE_INPUT": "0", "ADA_MODEL_PDF_INPUT": "false"})
    assert not disabled.input_capabilities.image_mime_types and not disabled.input_capabilities.pdf
    with pytest.raises(ValueError, match="ADA_MODEL_IMAGE_INPUT"):
        ModelConfig.from_env({"ADA_API_KEY": "test", "ADA_MODEL_IMAGE_INPUT": "maybe"})


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

    assert config.model == "auto(free)"
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

    assert config.model == "auto(free)"
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
