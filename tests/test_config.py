from copy import deepcopy

import pytest
import yaml

from praxis_agent.config import load_config


def save(tmp_path, **updates):
    data = {"models": {"main": {
        "protocol": "responses", "base_url": "https://gateway.example/v1", "model": "custom-model",
        "context_window": 128000, "max_output_tokens": 8192,
    }}}
    data.update(updates)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_configuration_is_required_even_with_old_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ADA_API_KEY", "unused")
    with pytest.raises(ValueError, match="Cannot load configuration file"):
        load_config(tmp_path / "missing.yaml")


def test_single_model_defaults_and_no_name_based_capabilities(tmp_path):
    config = load_config(save(tmp_path))
    assert config.default_model == "main"
    model = config.resolve()
    assert model.model == "custom-model"
    assert not model.api_key
    assert not model.input_capabilities.image_mime_types
    assert not model.input_capabilities.pdf
    assert model.max_retries == 2
    assert model.timeout_seconds == 120
    assert config.resolve("main", compression=True) == model


def test_credentials_are_lazy_and_never_part_of_snapshot(tmp_path, monkeypatch):
    path = save(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["models"]["main"]["api_key_env"] = "TEST_PROVIDER_KEY"
    path.write_text(yaml.safe_dump(data))
    config = load_config(path)
    monkeypatch.delenv("TEST_PROVIDER_KEY", raising=False)
    with pytest.raises(ValueError, match="TEST_PROVIDER_KEY"):
        config.resolve()
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-value")
    assert config.resolve().api_key == "secret-value"
    assert "secret-value" not in repr(config.resolve())
    assert "secret-value" not in repr(config)
    assert config.models["main"].api_key == ""


@pytest.mark.parametrize("update", [
    {"models": {}}, {"max_retries": -1}, {"timeout_seconds": 0},
    {"subagents": {"max_concurrent": True}}, {"unknown": 1},
    {"default_model": "missing"}, {"compression": {"model": "missing"}},
    {"subagents": {"model": "missing"}}, {"context": {"safety_ratio": 2}},
])
def test_invalid_configuration_fails_before_model_creation(tmp_path, update):
    with pytest.raises(ValueError):
        load_config(save(tmp_path, **update))


@pytest.mark.parametrize("field,value", [
    ("protocol", "unknown"), ("context_window", 0), ("context_window", True),
    ("max_output_tokens", 128000), ("max_input_tokens", 200000),
    ("base_url", "file:///tmp/model"), ("capabilities", {"tool_calling": False}),
])
def test_invalid_model_fields(tmp_path, field, value):
    path = save(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["models"]["main"][field] = value
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_config(path)


def test_multiple_models_and_independent_compression_parameters(tmp_path):
    path = save(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["models"]["other"] = deepcopy(data["models"]["main"])
    data["models"]["other"].update(model="other", context_window=64000,
                                      parameters={"reasoning_effort": "high"})
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="default_model"):
        load_config(path)
    data.update(default_model="main", compression={"model": "other", "parameters": {"reasoning_effort": "low"}},
                subagents={"model": "other", "max_concurrent": 2})
    path.write_text(yaml.safe_dump(data))
    config = load_config(path)
    assert config.resolve("main", compression=True).model == "other"
    assert config.resolve("main", compression=True).parameters["reasoning_effort"] == "low"
    assert config.resolve("other").parameters["reasoning_effort"] == "high"
    assert config.resolve("main").model_limits.context_window_tokens == 128000
    assert config.max_concurrent == 2
    with pytest.raises(TypeError):
        config.models["main"] = config.models["other"]


def test_duplicate_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("models: {}\nmodels: {}\n")
    with pytest.raises(ValueError, match="unique"):
        load_config(path)
