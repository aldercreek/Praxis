"""Run with the Praxis service interpreter; no network or real deployment."""
from pathlib import Path
from praxis_agent.config import load_config
from praxis_agent.hooks import HookManager
from praxis_agent.model import FunctionCall
from praxis_agent.planning import RunState
from praxis_agent.tool_execution import ToolCallExecutor
from praxis_agent.tool_extensions import ToolExtensions
from praxis_agent.tools import ToolContext, ToolRegistry

configuration = load_config(Path(__file__).with_name("praxis.yaml"))
registry = ToolRegistry()
ToolExtensions(configuration.tool_extensions).register(registry)


def approve(name, arguments):
    print("Demo approval:", name, arguments)
    assert arguments == {"environment": "prod"}
    return True


executor = ToolCallExecutor(registry, ToolContext(Path.cwd(), RunState("hook demo")),
                           lambda event: None, approve_tool_call=approve,
                           hooks=HookManager(configuration.hook_extensions))
result = executor.execute("demo", FunctionCall("query", "release__query", '{"environment": "production"}'))
assert result.success and result.structured_output["access_token"] == "***"
assert "demo-secret" not in result.output
print("Filtered result:", result.output)
