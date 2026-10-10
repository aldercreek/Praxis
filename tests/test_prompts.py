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
    assert "read only the relevant line ranges" in instructions
    assert "Use find_files to locate files by name or glob pattern." in instructions
    assert "multiple tool calls are independent" in instructions
    assert "Use update_plan for non-trivial tasks" in instructions
    assert "at most one plan step in_progress" in instructions
    assert "Manage the working context proactively" in instructions
    assert "Use delete for material with no continuing value" in instructions
    assert "Use compress when the full text is no longer needed" in instructions
    assert "Context operations take effect in the next model request" in instructions
    assert "ADA_API_KEY" not in instructions
