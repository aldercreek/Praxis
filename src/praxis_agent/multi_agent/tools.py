from __future__ import annotations

import json
from collections.abc import Callable

from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.tools import Tool, ToolContext, ToolRegistry


PARENT_INSTRUCTIONS = """Delegate when parallel progress or reduced context burden outweigh coordination costs.
Choose manageable objectives with clear expected outcomes, necessary inputs and dependencies.
Do not transfer an oversized task unchanged or concentrate all remaining work in one agent.
Adapt to actual progress: answer requests for help, reuse completed work, and adjust scope,
inputs or assignments when necessary. Remain responsible for overall integration and quality.

Use read_only workers for investigation and execute workers for changes. Prepare an independent
working directory with the required materials before starting an execute worker; Praxis binds
tools to that directory but does not copy or synchronize files. For Git work, include relevant
uncommitted inputs without losing the user's work. Directories isolate work, not command access.
Coordinate dependencies and updates through the provided communication tools; workers cannot
contact one another. Request concise deliveries with exact artifact paths and verification.
Inspect relevant results and integrate only needed changes after their producing runs have
ended. Handle conflicts explicitly, preserve unrelated work and ancestry needed for later
integration, and verify the combined outcome.

Before finalizing, let involved workers end their current runs, review outcomes and account for
unresolved requests. A run ending does not prove task completion. Worker histories and runtime
state do not survive closure/restart; retained deliveries and files support later continuation.
"""

WORKER_INSTRUCTIONS = """You are a worker assigned by the parent. Advance your assignment within its scope and report
findings or difficulties that materially affect completion. If blocked or overloaded, explain
completed work, the obstacle, remaining work and a useful adjustment. When new instructions are
needed, report and end this run awaiting coordination; reuse completed work on resumption.

Your history is private and temporary. Do not spawn workers or contact other workers. Follow
your execution mode. Your final response is automatically delivered to the parent: provide a
concise outcome, exact artifact/source paths, verification, uncertainty and unfinished work.
Do not label a blocked task complete or copy large bodies already available in files.
"""

READ_ONLY_INSTRUCTIONS = """Mode: read_only. Inspect and analyze; do not modify files or execute commands. Programmatic
execution does not grant writing or MCP access. Cite existing sources for large findings, or
ask the parent to arrange an execute worker when a new file is required.
"""

EXECUTE_INSTRUCTIONS = """Mode: execute. Edit files and run commands in your assigned working directory. Do not modify
the parent's or another worker's working files. Coordinate input changes and dependencies with
the parent; there is no automatic synchronization. Preserve unrelated changes and delivered
artifacts, or explicitly report replacement. For Git deliveries, include the branch and commit
when available. Praxis does not automatically integrate or delete your work on closure.
Commands are foreground jobs; descendant processes stop when a command finishes or is cancelled.
Idle/resume retains the working directory, conversation and saved execution sessions.
"""


def _schema(properties: dict[str, object]) -> dict[str, object]:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def register_parent_tools(registry: ToolRegistry, runtime: AgentRuntime,
                          pump: Callable[[], None]) -> None:
    def spawn_agent(context: ToolContext, task: str, task_summary: str = "",
                    mode: str = "read_only", working_directory: str = "") -> str:
        return json.dumps(runtime.spawn(task, task_summary, mode, working_directory), ensure_ascii=False)

    def send_messages(context: ToolContext, messages: list[dict[str, object]]) -> str:
        return json.dumps(runtime.send_many(messages), ensure_ascii=False)

    def wait_agents(context: ToolContext, agent_ids: list[str], timeout: int) -> str:
        return json.dumps(runtime.wait_agents(agent_ids, timeout, pump), ensure_ascii=False)

    def close_agent(context: ToolContext, agent_id: str) -> str:
        return json.dumps(runtime.close_agent(agent_id), ensure_ascii=False)

    registry.register(Tool(
        name="spawn_agent", description="Start a worker in an existing working_directory. Execute requires a directory prepared by the parent using commands; no files are copied or Git operations performed. Empty directory uses the parent workspace for read_only. Deliveries are ordinary messages or summary plus file paths.",
        parameters=_schema({"task": {"type": "string", "minLength": 1,
                                    "description": "A manageable objective, necessary inputs or file paths, scope, acceptance criteria and deliverable. Do not move an oversized task unchanged to a worker."},
                            "task_summary": {"type": "string", "maxLength": 200,
                                             "description": "Short task scope for the runtime snapshot, not the complete task instructions."},
                            "mode": {"type": "string", "enum": ["read_only", "execute"], "description": "read_only (default) for inspection; execute for edits/commands in a prepared directory."},
                            "working_directory": {"type": "string",
                                       "description": "Existing directory, absolute or relative to the parent workspace. Required for execute; empty inherits parent for read_only. Prepare an independent directory or worktree before spawning execute workers."}}),
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
