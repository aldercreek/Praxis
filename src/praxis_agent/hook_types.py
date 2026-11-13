"""Public hook payloads. These contain snapshots, never live runtime objects."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class HookDecision:
    allowed: bool = True
    reason: str | None = None
    arguments: dict | None = None
    requires_approval: bool = False

    @classmethod
    def allow(cls, *, arguments=None, requires_approval=False):
        return cls(arguments=arguments, requires_approval=requires_approval)

    @classmethod
    def deny(cls, reason):
        return cls(allowed=False, reason=reason)


@dataclass(frozen=True, slots=True)
class HookOutput:
    """Replace all model-visible result content; omitted channels are cleared."""
    output: str
    structured_output: dict | None = None
    model_content: list[dict] | None = None


@dataclass(frozen=True, slots=True)
class HookResult:
    success: bool
    output: str
    structured_output: dict | None = None
    model_content: list[dict] | None = None


@dataclass(frozen=True, slots=True)
class HookEvent:
    event: str
    tool_name: str
    description: str
    parameters: dict
    requires_approval: bool
    session_id: str | None
    agent_id: str
    turn_id: str
    call_id: str
    parent_call_id: str | None
    workspace: Path
    programmatic_call: bool
    original_arguments: dict
    arguments: dict
    approval_requested: bool = False
    approval_hooks: tuple[str, ...] = ()
    original_result: HookResult | None = None
    result: HookResult | None = None
    duration_ms: float | None = None
