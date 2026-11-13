"""Offline demo: normalize inputs, require approval, and redact outputs."""
import json
from praxis_agent.sdk import tool, hook, HookDecision, HookOutput


@tool(description="Simulate a release query", requires_approval=False)
def query(environment: str):
    return {"environment": environment, "status": "ready", "access_token": "demo-secret"}


@hook("before_tool_call", tools=["release__query"])
def normalize(event):
    arguments = dict(event.arguments)
    if arguments.get("environment") == "production":
        arguments["environment"] = "prod"
    return HookDecision.allow(arguments=arguments, requires_approval=arguments.get("environment") == "prod")


@hook("after_tool_call", tools=["release__query"])
def redact(event):
    data = {**event.result.structured_output, "access_token": "***"}
    return HookOutput(json.dumps(data), structured_output=data)
