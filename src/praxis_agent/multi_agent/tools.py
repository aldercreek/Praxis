from __future__ import annotations

import json
from collections.abc import Callable

from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.tools import Tool, ToolContext, ToolRegistry


PARENT_INSTRUCTIONS = """<collaboration>
1. 分配工作
   需要并行推进或独立处理部分工作时，可创建子 agent。
   明确目标、范围、必要输入、验收条件和交付要求。

2. 协调与交付
   及时沟通影响任务的依赖、输入变化和阻塞。
   整合协作结果，交付结果、必要证据和未完成事项。
</collaboration>
"""

WORKER_INSTRUCTIONS = """<collaboration>
1. 接受分工
   完成主 agent 分配的工作，遵循约定的范围和验收条件。

2. 协调与交付
   及时向主 agent 沟通影响任务的依赖、输入变化和阻塞。
   交付结果、必要证据和未完成事项。
</collaboration>
"""


def _schema(properties: dict[str, object]) -> dict[str, object]:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def register_parent_tools(registry: ToolRegistry, runtime: AgentRuntime,
                          pump: Callable[[], None]) -> None:
    def spawn_agent(context: ToolContext, task: str, task_summary: str = "",
                    working_directory: str = "") -> str:
        return json.dumps(runtime.spawn(task, task_summary, working_directory=working_directory), ensure_ascii=False)

    def send_messages(context: ToolContext, messages: list[dict[str, object]]) -> str:
        return json.dumps(runtime.send_many(messages), ensure_ascii=False)

    def wait_agents(context: ToolContext, agent_ids: list[str], timeout: int) -> str:
        return json.dumps(runtime.wait_agents(agent_ids, timeout, pump), ensure_ascii=False)

    def close_agent(context: ToolContext, agent_id: str) -> str:
        return json.dumps(runtime.close_agent(agent_id), ensure_ascii=False)

    registry.register(Tool(
        name="spawn_agent", description="Delegate a bounded task to a worker. Returns its ID, workspace and registered command/file-write capabilities; tool availability does not bypass approvals. Deliveries arrive as messages with artifact paths.",
        parameters=_schema({"task": {"type": "string", "minLength": 1,
                                    "description": "Objective, available inputs, scope, acceptance criteria and deliverable. For independent review, name the unresolved question and evidence needed; review is not mandatory after successful checks."},
                            "task_summary": {"type": "string", "maxLength": 200,
                                             "description": "Short task scope for the runtime snapshot, not the complete task instructions."},
                            "working_directory": {"type": "string", "minLength": 1,
                                       "description": "Existing directory, absolute or relative to the parent. Required, non-empty and independently prepared. Confirm critical inputs/dependencies here before spawning. No files are copied or synchronized."}}),
        handler=spawn_agent,
    ))
    registry.register(Tool(
        name="send_messages",
        description="Batch messages to workers. Each entry succeeds or fails independently. Delivery does not mean processing is complete. Idle workers resume after the whole batch is enqueued.",
        parameters=_schema({"messages": {"type": "array", "minItems": 1, "maxItems": 100, "description": "Messages to existing workers; idle workers resume after the batch is enqueued.",
            "items": _schema({"agent_id": {"type": "string", "description": "Exact worker ID returned by spawn_agent."},
                              "content": {"type": "string", "minLength": 1,
                                          "description": "Finding, correction or focused reassignment. Reduce an overloaded worker's scope; pass focused findings and relevant file paths rather than large bodies."},
                              "task_summary": {"type": "string", "maxLength": 200,
                                               "description": "Updated short scope when reassigning work. Use an empty string for ordinary messages to retain the current label."}})}}),
        handler=send_messages,
    ))
    registry.register(Tool(
        name="wait_agents", description="Wait for unread messages, all selected runs to end, or timeout. Returns messages and refreshed worker snapshots. An ended run may need coordination rather than being finished work. Returned messages are consumed once.",
        parameters=_schema({"agent_ids": {"type": "array", "minItems": 1, "description": "Exact IDs of workers whose messages/run status to wait for.",
                                          "items": {"type": "string"}},
                            "timeout": {"type": "integer", "minimum": 0, "maximum": 60,
                                        "description": "Wait in seconds; normally use 60."}}),
        handler=wait_agents,
    ))
    registry.register(Tool(
        name="close_agent", description="Cancel/close a worker and release its private context after execution stops. A busy worker returns closing; wait for closed before handling unfinished files. The directory and files are not deleted.",
        parameters=_schema({"agent_id": {"type": "string", "description": "Exact worker ID returned by spawn_agent."}}), handler=close_agent,
    ))


def register_worker_tools(registry: ToolRegistry, runtime: AgentRuntime, agent_id: str) -> None:
    def send_message(context: ToolContext, content: str) -> str:
        return json.dumps(runtime.send_parent(agent_id, content), ensure_ascii=False)

    registry.register(Tool(
        name="send_message", description="Send an actionable finding, question or reassignment request. For overload explain completed work, remaining scope and a suggested split. Sending does not pause: end this run if waiting for coordination. Do not duplicate final deliveries.",
        parameters=_schema({"content": {"type": "string", "minLength": 1, "description": "Actionable finding or coordination request for the parent; sending alone does not pause execution."}}),
        handler=send_message,
    ))
