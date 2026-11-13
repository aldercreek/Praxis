"""Explicit single-file extensions, loaded once and shared by service agents."""
from __future__ import annotations

import inspect
import json
import re
import typing
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

from pydantic import ConfigDict, TypeAdapter
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import to_jsonable_python
from typing_extensions import NotRequired, TypedDict

from praxis_agent.extension_loading import extension_modules, marked_functions
from praxis_agent.sdk import ToolMetadata, _METADATA_ATTRIBUTE
from praxis_agent.tools.registry import Tool, ToolRegistry, ToolResult

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")


class _InputSchema(GenerateJsonSchema):
    """Keep dataclass input boundaries in the pre-approval JSON validation."""

    def field_is_present(self, field):
        if field["type"] == "dataclass-field" and field.get("init") is False:
            return False
        return super().field_is_present(field)

    def dataclass_schema(self, schema):
        result = super().dataclass_schema(schema)
        result.setdefault("additionalProperties", False)
        return result


def _adapt(function, extension_id):
    location = f"extension {extension_id}, function {function.__name__}"
    metadata = getattr(function, _METADATA_ATTRIBUTE)
    if not isinstance(metadata, ToolMetadata):
        raise ValueError(f"{location}: invalid SDK tool metadata")
    if not isinstance(metadata.description, str) or not metadata.description.strip():
        raise ValueError(f"{location}: description must be a nonempty string")
    if any(type(getattr(metadata, key)) is not bool for key in ("requires_approval", "parallel_safe", "programmatic")):
        raise ValueError(f"{location}: tool policy flags must be booleans")
    name = function.__name__ if metadata.name is None else metadata.name
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"{location}: tool name must start with an ASCII letter and contain letters, digits, _ or -")
    name = f"{extension_id}__{name}"
    if len(name) > 64:
        raise ValueError(f"{location}: final tool name {name!r} exceeds 64 characters")
    if inspect.iscoroutinefunction(function) or inspect.isgeneratorfunction(function) or inspect.isasyncgenfunction(function):
        raise ValueError(f"{location}: only ordinary synchronous functions are supported")
    signature = inspect.signature(function)
    fields = {}
    for parameter in signature.parameters.values():
        field_path = f"{location}.{parameter.name}"
        if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            raise ValueError(f"{field_path}: positional-only, *args and **kwargs are not supported")
        if parameter.annotation is inspect.Parameter.empty:
            raise ValueError(f"{field_path}: input type annotation is required")
        try:
            # Resolve inputs separately so an optional return annotation imposes no constraint.
            hints = typing.get_type_hints(SimpleNamespace(__annotations__={parameter.name: parameter.annotation}),
                                          globalns=function.__globals__, include_extras=True)
        except Exception as error:
            raise ValueError(f"{field_path}: cannot resolve input annotation: {error}") from error
        annotation = hints[parameter.name]
        fields[parameter.name] = (annotation if parameter.default is inspect.Parameter.empty
                                  else NotRequired[annotation])

    # A TypedDict preserves omitted arguments for Python function defaults, without
    # BaseModel field-name restrictions or copying constructed nested objects.
    parameters = TypedDict(f"{name}_parameters", fields)
    parameters.__pydantic_config__ = ConfigDict(extra="forbid", allow_inf_nan=False)
    try:
        adapter = TypeAdapter(parameters)
        schema = adapter.json_schema(schema_generator=_InputSchema)
    except Exception as error:
        raise ValueError(f"{location}: cannot build input schema: {error}") from error

    def invoke(context, /, **arguments):
        if context.check_execution:
            context.check_execution()
        # JSON mode supports string representations of dates/UUIDs and arrays for
        # tuples/sets even in strict mode. Validators/constructors run after approval.
        converted = adapter.validate_json(json.dumps(arguments, allow_nan=False), strict=True)
        if context.check_execution:
            context.check_execution()
        result = function(**converted)
        if isinstance(result, str):
            return ToolResult(True, result)
        result = to_jsonable_python(result, inf_nan_mode="constants")
        return ToolResult(True, json.dumps(result, ensure_ascii=False, allow_nan=False),
                          structured_output=result if isinstance(result, dict) else None)

    return Tool(name, metadata.description, schema, invoke,
                requires_approval=metadata.requires_approval, parallel_safe=metadata.parallel_safe,
                strict=False, programmatic=metadata.programmatic,
                source=f"python:{extension_id}", display_name=f"{extension_id} · {name.split('__', 1)[1].replace('_', ' ')}")


class ToolExtensions:
    """A service-owned immutable catalog; registries and approvals remain per-agent.

    sys.modules retains successful imports for this process's lifetime. Updating
    code or configuration requires a process restart, including for SDK callers.
    """

    def __init__(self, configurations=()):
        pending, names = [], set()
        with extension_modules(configurations) as modules:
            for config, module in modules:
                for function in marked_functions(module, _METADATA_ATTRIBUTE):
                    candidate = _adapt(function, config.id)
                    if candidate.name in names:
                        raise ValueError(f"extension {config.id}: duplicate tool name {candidate.name}")
                    names.add(candidate.name)
                    pending.append(candidate)
        self.tools = tuple(pending)

    def register(self, registry: ToolRegistry):
        conflicts = set(registry.names) & {tool.name for tool in self.tools}
        if conflicts:
            raise ValueError(f"Extension tools conflict with registered tools: {', '.join(sorted(conflicts))}")
        for tool in self.tools:
            registry.register(replace(tool, parameters=deepcopy(tool.parameters)))
