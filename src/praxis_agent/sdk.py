"""Public SDK for explicitly configured, local Python tools."""
from dataclasses import dataclass
from typing import Callable, TypeVar

from praxis_agent.hook_types import HookDecision, HookEvent, HookOutput, HookResult

__all__ = ["tool", "hook", "HookDecision", "HookEvent", "HookOutput", "HookResult"]

_F = TypeVar("_F", bound=Callable)
_METADATA_ATTRIBUTE = "__praxis_tool__"


@dataclass(frozen=True, slots=True)
class ToolMetadata:
    description: str
    name: str | None = None
    requires_approval: bool = True
    parallel_safe: bool = False
    programmatic: bool = True


def tool(*, description: str, name: str | None = None,
         requires_approval: bool = True, parallel_safe: bool = False,
         programmatic: bool = True) -> Callable[[_F], _F]:
    """Mark a synchronous function as a tool; return the function unchanged.

    Praxis validates signatures and annotations when loading the extension.
    This decorator never executes the decorated function.
    """
    metadata = ToolMetadata(description, name, requires_approval, parallel_safe, programmatic)

    def decorate(function: _F) -> _F:
        setattr(function, _METADATA_ATTRIBUTE, metadata)
        return function

    return decorate


_HOOK_ATTRIBUTE = "__praxis_hook__"
HOOK_EVENTS = ("before_tool_call", "after_tool_call", "on_tool_error")


@dataclass(frozen=True, slots=True)
class HookMetadata:
    event: str
    tools: tuple[str, ...]


def hook(event: str, *, tools=None):
    """Subscribe a synchronous function to a tool event, in definition order."""
    if event not in HOOK_EVENTS:
        raise ValueError(f"Unknown hook event: {event}")
    patterns = ("*",) if tools is None else tuple(tools)
    if isinstance(tools, str) or not patterns or any(not isinstance(p, str) or not p for p in patterns):
        raise ValueError("Hook tools must be a nonempty list of tool-name patterns")
    def decorate(function):
        if hasattr(function, _HOOK_ATTRIBUTE):
            raise ValueError("Use a separate function for each hook subscription")
        setattr(function, _HOOK_ATTRIBUTE, HookMetadata(event, patterns))
        return function
    return decorate
