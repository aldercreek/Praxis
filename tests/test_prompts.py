import json
from pathlib import Path

from praxis_agent.prompts import build_instructions
from praxis_agent.runtime import RuntimeContext


def test_build_instructions_includes_the_runtime_context() -> None:
    context = RuntimeContext(
        operating_system="ExampleOS",
        sys_platform="example-platform",
        workspace_root=Path("workspace").resolve(),
        available_shells=("cmd", "pwsh"),
        unavailable_shells=("bash", "sh"),
    )

    instructions = build_instructions(context)

    assert "Operating system: ExampleOS (example-platform)" in instructions
    encoded_workspace = json.dumps(str(context.workspace_root))
    assert f"Workspace root: {encoded_workspace}" in instructions
    assert "Available shells: cmd, pwsh" in instructions
    assert "Unavailable shells: bash, sh" in instructions
    assert "ADA_API_KEY" not in instructions
