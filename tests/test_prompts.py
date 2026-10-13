import json
from pathlib import Path

from praxis_agent.prompts import BASE_INSTRUCTIONS, build_instructions
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

    assert "操作系统：ExampleOS (example-platform)" in instructions
    encoded_workspace = json.dumps(str(context.workspace_root))
    assert f"工作目录：{encoded_workspace}" in instructions
    assert "可用 shell：cmd, pwsh" in instructions
    assert instructions.startswith(BASE_INSTRUCTIONS.rstrip() + "\n\n")
    assert "3. Available skills: none discovered.\n</skills>" in instructions
    assert "ADA_API_KEY" not in instructions
