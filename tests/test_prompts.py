import json
from pathlib import Path

from praxis_agent.prompts import BASE_INSTRUCTIONS, build_instructions
from praxis_agent.runtime import CommandShell, RuntimeContext


def test_build_instructions_includes_the_runtime_context() -> None:
    context = RuntimeContext(
        operating_system="ExampleOS",
        sys_platform="example-platform",
        workspace_root=Path("workspace").resolve(),
        available_shells=("cmd", "pwsh"),
        unavailable_shells=("bash", "sh"),
        command_shell=CommandShell("pwsh", "D:/PowerShell/pwsh.exe"),
        shell_version="7.5.0",
    )

    instructions = build_instructions(context)

    assert "Operating system: ExampleOS (example-platform)" in instructions
    encoded_workspace = json.dumps(str(context.workspace_root))
    assert f"Working directory: {encoded_workspace}" in instructions
    assert "Available shells: cmd, pwsh" in instructions
    assert "Shell used by command: pwsh 7.5.0 (D:/PowerShell/pwsh.exe)" in instructions
    assert "show the beginning and end" in instructions
    assert "show only the beginning" not in instructions
    assert instructions.startswith(BASE_INSTRUCTIONS.rstrip() + "\n\n")
    assert "3. Available skills: none discovered.\n</skills>" in instructions
    assert "ADA_API_KEY" not in instructions
