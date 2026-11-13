"""Public SDK for explicitly configured, local Python tools."""
from dataclasses import dataclass
from typing import Callable, TypeVar

__all__ = ["tool"]

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
