"""File-only configuration. Credentials are resolved only for selected models."""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType

import yaml
from jsonschema import Draft202012Validator

from praxis_agent.context.capacity import ContextPolicy
from praxis_agent.context.model_metadata import ModelInputCapabilities, ModelLimits
from praxis_agent.sessions.database import data_directory


DEFAULT_MAX_PARALLEL_TOOLS = 4


def _object(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


_TEXT = {"type": "string", "pattern": r"\S"}
_POSITIVE = {"type": "integer", "minimum": 1}
_PARAMETERS = _object({
    "reasoning_effort": _TEXT,
    "temperature": {"type": "number", "minimum": 0, "maximum": 2},
})
_MODEL = _object({
    "protocol": {"enum": ["responses", "chat_completions"]},
    "base_url": {"type": "string", "pattern": r"^https?://[^\s/]+"},
    "model": _TEXT,
    "api_key_env": _TEXT,
    "context_window": _POSITIVE,
    "max_input_tokens": _POSITIVE,
    "max_output_tokens": _POSITIVE,
    "capabilities": _object({
        "image_input": {"type": "boolean"}, "pdf_input": {"type": "boolean"},
        "tool_calling": {"type": "boolean"},
    }),
    "parameters": _PARAMETERS,
}, ("protocol", "base_url", "model", "context_window", "max_output_tokens"))
_SCHEMA = _object({
    "default_model": _TEXT,
    "models": {"type": "object", "minProperties": 1, "propertyNames": _TEXT,
               "additionalProperties": _MODEL},
    "compression": _object({"model": _TEXT, "parameters": _PARAMETERS}),
    "subagents": _object({"model": _TEXT, "max_concurrent": _POSITIVE}),
    "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
    "max_retries": {"type": "integer", "minimum": 0},
    "max_parallel_tools": _POSITIVE,
    "tool_extensions": {"type": "array", "items": _object({
        "id": {"type": "string", "pattern": r"^[A-Za-z][A-Za-z0-9_-]{0,31}$"},
        "path": _TEXT,
        "enabled": {"type": "boolean"},
    }, ("id", "path"))},
    "hook_extensions": {"type": "array", "items": _object({
        "id": {"type": "string", "pattern": r"^[A-Za-z][A-Za-z0-9_-]{0,31}$"},
        "path": _TEXT,
        "enabled": {"type": "boolean"},
    }, ("id", "path"))},
    "observability": _object({
        "enabled": {"type": "boolean"},
        "provider": {"type": "string", "pattern": r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"},
        "options": {"type": "object"},
    }),
    "context": _object({f.name: {"type": "number" if "ratio" in f.name else "integer"}
                        for f in fields(ContextPolicy) if f.name != "output_reserve_tokens"}),
}, ("models",))


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model: str
    base_url: str
    protocol: str
    model_limits: ModelLimits
    api_key: str = field(default="", repr=False)
    input_capabilities: ModelInputCapabilities = field(default_factory=ModelInputCapabilities)
    tool_calling: bool = True
    parameters: Mapping = field(default_factory=dict)
    context_policy: ContextPolicy | None = None
    max_parallel_tools: int = DEFAULT_MAX_PARALLEL_TOOLS
    timeout_seconds: float = 120
    max_retries: int = 2
    model_key: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        if self.context_policy is None:
            object.__setattr__(self, "context_policy", ContextPolicy.for_model(self.model_limits))
        self.context_policy.effective_input_limit(self.model_limits)


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    provider: str | None = None
    options: Mapping = field(default_factory=dict, repr=False)
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class ToolExtensionConfig:
    id: str
    path: Path
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class PraxisConfig:
    path: Path
    default_model: str
    models: Mapping[str, ModelConfig]
    credential_envs: Mapping[str, str]
    compression_model: str | None = None
    compression_parameters: Mapping = field(default_factory=dict)
    subagent_model: str | None = None
    max_concurrent: int = 4
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    tool_extensions: tuple[ToolExtensionConfig, ...] = ()
    hook_extensions: tuple[ToolExtensionConfig, ...] = ()

    def resolve(self, name: str | None = None, *, compression=False) -> ModelConfig:
        name = (self.compression_model if compression else None) or name or self.default_model
        if name not in self.models:
            raise ValueError(f"Unknown model configuration: {name}")
        variable = self.credential_envs.get(name)
        key = os.environ.get(variable, "").strip() if variable else ""
        if variable and not key:
            raise ValueError(f"Credential environment variable {variable} is required for {name}")
        config = self.models[name]
        parameters = dict(config.parameters)
        if compression:
            parameters.update(self.compression_parameters)
        return replace(config, api_key=key, parameters=parameters, model_key=name)


class _ConfigLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError("Configuration keys must be unique strings")
        result[key] = loader.construct_object(value_node)
    return result


_ConfigLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_config(path: Path | str | None = None) -> PraxisConfig:
    path = Path(path or os.environ.get("PRAXIS_CONFIG") or data_directory() / "praxis.yaml").expanduser().resolve()
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_ConfigLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot load configuration file: {path}") from error
    errors = list(Draft202012Validator(_SCHEMA).iter_errors(data))
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "root"
        raise ValueError(f"Invalid configuration at {location}: {error.validator} validation failed")
    default = data.get("default_model")
    if default is None:
        if len(data["models"]) != 1:
            raise ValueError("default_model is required when multiple models are configured")
        default = next(iter(data["models"]))
    compression = data.get("compression", {})
    subagents = data.get("subagents", {})
    for name in (default, compression.get("model"), subagents.get("model")):
        if name is not None and name not in data["models"]:
            raise ValueError(f"Unknown model configuration: {name}")
    models, credentials = {}, {}
    for name, item in data["models"].items():
        limits = ModelLimits(item["context_window"], item.get("max_input_tokens"), item["max_output_tokens"])
        if limits.max_output_tokens >= limits.context_window_tokens:
            raise ValueError(f"{name}: max_output_tokens must be below context_window")
        if limits.max_input_tokens and limits.max_input_tokens > limits.context_window_tokens:
            raise ValueError(f"{name}: max_input_tokens exceeds context_window")
        capabilities = item.get("capabilities", {})
        if item["protocol"] == "chat_completions" and capabilities.get("pdf_input", False):
            raise ValueError(f"{name}: PDF input is only supported by the responses adapter")
        models[name] = ModelConfig(
            model=item["model"], base_url=item["base_url"], protocol=item["protocol"],
            model_limits=limits, parameters=item.get("parameters", {}),
            context_policy=ContextPolicy.for_model(limits, output_reserve_tokens=limits.max_output_tokens, **data.get("context", {})),
            input_capabilities=ModelInputCapabilities(
                ("image/png", "image/jpeg", "image/webp", "image/gif") if capabilities.get("image_input") else (),
                pdf=capabilities.get("pdf_input", False),
            ),
            tool_calling=capabilities.get("tool_calling", True),
            max_parallel_tools=data.get("max_parallel_tools", DEFAULT_MAX_PARALLEL_TOOLS),
            timeout_seconds=data.get("timeout_seconds", 120), max_retries=data.get("max_retries", 2),
        )
        if item.get("api_key_env"):
            credentials[name] = item["api_key_env"]
    for name in (default, subagents.get("model")):
        if name is not None and not models[name].tool_calling:
            raise ValueError(f"{name}: agents require tool_calling")
    observability = data.get("observability", {})
    def extensions(key):
        entries, identifiers = [], set()
        for item in data.get(key, []):
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", item["id"]):
                raise ValueError(f"Invalid configuration at {key}: invalid extension id")
            if item["id"] in identifiers:
                kind = "tool" if key == "tool_extensions" else "hook"
                raise ValueError(f"Duplicate {kind} extension id: {item['id']}")
            identifiers.add(item["id"])
            extension_path = Path(item["path"]).expanduser()
            if not extension_path.is_absolute():
                extension_path = path.parent / extension_path
            entries.append(ToolExtensionConfig(item["id"], extension_path.resolve(), item.get("enabled", True)))
        return tuple(entries)
    return PraxisConfig(path, default, MappingProxyType(models), MappingProxyType(credentials),
                        compression.get("model"), MappingProxyType(compression.get("parameters", {})),
                        subagents.get("model"), subagents.get("max_concurrent", 4),
                        ObservabilityConfig(observability.get("provider"), observability.get("options", {}),
                                            observability.get("enabled", True)), extensions("tool_extensions"),
                        extensions("hook_extensions"))
