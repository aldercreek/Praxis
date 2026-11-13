"""Run with the Praxis Python interpreter; no live model or external API."""
import json
from pathlib import Path

from praxis_agent.config import load_config
from praxis_agent.model import FunctionCall
from praxis_agent.planning import RunState
from praxis_agent.tool_execution import ToolCallExecutor
from praxis_agent.tool_extensions import ToolExtensions
from praxis_agent.tools import ToolContext, ToolRegistry


def main():
    directory = Path(__file__).resolve().parent
    catalog = ToolExtensions(load_config(directory / "praxis.yaml").tool_extensions)
    registry = ToolRegistry()
    catalog.register(registry)
    print("Registered:", ", ".join(registry.names))

    def approve(name, arguments):
        print("Demo approval:", name, json.dumps(arguments, ensure_ascii=False))
        return True  # Explicitly approve only this offline example's simulated calls.

    executor = ToolCallExecutor(registry, ToolContext(directory, RunState("offline example")),
                                publish_event=lambda event: None, approve_tool_call=approve)
    for name, arguments in [
        ("release__greet", {"name": "Praxis"}),
        ("release__get_release_status", {"query": {"release_id": "123", "include_logs": True}}),
    ]:
        result = executor.execute("demo", FunctionCall(name, name, json.dumps(arguments)))
        if not result.success:
            raise RuntimeError(result.output)
        print(name, "=>", result.output)


if __name__ == "__main__":
    main()
