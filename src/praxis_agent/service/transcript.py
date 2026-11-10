from __future__ import annotations

import json

from .collaboration import collaboration_messages


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
            and part.get("type") in {"input_text", "output_text"}
        )
    return ""


def tool_result_status(result: dict, previous: str = "", tool_name: str = "") -> str:
    output = result.get("output", "")
    structured = result.get("structured_output") or (
        output if isinstance(output, dict) else {}
    )
    if tool_name == "run_code":
        code = structured
        if not code and isinstance(output, str):
            try:
                code = json.loads(output)
            except (ValueError, TypeError):
                code = {}
        if isinstance(code, dict) and code.get("error_kind") == "cancelled":
            return "cancelled"
    if structured.get("status") == "running":
        return "running"
    if structured.get("status") == "cancelled" or (
        isinstance(output, str) and output.startswith("ExecutionStopped:")
    ):
        return "cancelled"
    if (
        previous == "denied"
        or output == "tool execution was declined by the user; do not retry"
    ):
        return "denied"
    return "completed" if result.get("success") else "failed"


class Transcript:
    """A plain display snapshot, independent of Textual and model prompt internals."""

    def __init__(self):
        self.messages: list[dict] = []
        self.responses: dict[tuple, dict] = {}
        self.last_response: dict[tuple, dict] = {}
        self.tools: dict[tuple, dict] = {}
        self.agents: dict[tuple, dict] = {}

    def append(self, role, content, **extra):
        message = {
            "id": f"message-{len(self.messages) + 1}",
            "role": role,
            "content": content,
            **extra,
        }
        self.messages.append(message)
        return message

    def consume(self, event: str, turn_id: str, agent_id: str, data: dict):
        common = {"turn_id": turn_id, "agent_id": agent_id}
        execution_id = data.get("child_turn_id") or turn_id
        key = (agent_id, execution_id)
        if data.get("child_turn_id"):
            common["child_turn_id"] = data["child_turn_id"]
        if event == "TurnStarted":
            task = data["task"]
            envelope = collaboration_messages(task) if agent_id != "main" else None
            if envelope:
                for index, report in enumerate(envelope):
                    self.append(
                        "user", report["content"], **common, source="agent",
                        sender_id=report["sender_id"], recipient_id=report["recipient_id"],
                        status="completed", id=f"user:{turn_id}:{agent_id}:{execution_id}:report:{index}",
                    )
                return
            self.append(
                "user",
                task,
                **common,
                status="completed",
                source="user" if agent_id == "main" else "agent",
                id=f"user:{turn_id}:{agent_id}:{execution_id}",
            )
        elif event == "ModelRequestStarted":
            previous = self.last_response.get(key)
            if previous is not None and previous.get("status") == "streaming":
                previous["status"] = "failed"
        elif event in {"AssistantTextDelta", "ModelRequestCompleted"}:
            response_key = (*key, data["request_number"])
            text = (
                data.get("text", "")
                if event == "AssistantTextDelta"
                else data.get("output_text", "")
            )
            message = self.responses.get(response_key)
            if message is None and text:
                message = self.append(
                    "assistant",
                    "",
                    **common,
                    status="streaming",
                    id=f"assistant:{turn_id}:{agent_id}:{execution_id}:{data['request_number']}",
                    request_number=data["request_number"],
                )
                self.responses[response_key] = message
                self.last_response[key] = message
            if message is not None:
                if event == "AssistantTextDelta":
                    message["content"] += text
                else:
                    message["content"] = text or message["content"]
                    message["status"] = "completed"
        elif event == "AssistantTextCompleted":
            message = self.last_response.get(key)
            if message is None:
                message = self.append(
                    "assistant",
                    data["text"],
                    **common,
                    id=f"assistant:{turn_id}:{agent_id}:{execution_id}:final",
                )
                self.last_response[key] = message
            message.update(content=data["text"], status="completed")
        elif event in {
            "ToolCallPrepared",
            "ToolStarted",
            "ToolFinished",
            "ToolOutputDelta",
            "ApprovalRequested",
            "ApprovalResolved",
        }:
            tool_key = (agent_id, data["call_id"])
            message = self.tools.get(tool_key)
            if message is None:
                message = self.append(
                    "tool",
                    "",
                    **common,
                    call_id=data["call_id"],
                    tool_name=data.get("tool_name", "run_command"),
                    status="running",
                    id=f"tool:{turn_id}:{agent_id}:{data['call_id']}",
                    tool_finished=False,
                )
                self.tools[tool_key] = message
            if event == "ToolCallPrepared":
                message.update(
                    arguments=data["arguments"],
                    status="approval" if data["requires_approval"] else "pending",
                )
            elif event == "ToolStarted":
                message["status"] = "running"
            elif event == "ApprovalRequested":
                message["status"] = "approval"
            elif event == "ApprovalResolved":
                message["status"] = "pending" if data["approved"] else "denied"
            elif event == "ToolOutputDelta":
                message["output"] = (message.get("output", "") + data.get("text", ""))[
                    -120_000:
                ]
                if data.get("command"):
                    message["command"] = data["command"]
            else:
                result = data["result"]
                message.update(
                    content=result["output"],
                    status=tool_result_status(result, message["status"], message.get("tool_name", "")),
                    tool_finished=True,
                )
                if result.get("structured_output"):
                    message["structured_output"] = result["structured_output"]
                if result.get("file_change"):
                    message["file_change"] = result["file_change"]
        elif event == "ChildAgentStatus":
            agent_key = (turn_id, agent_id)
            message = self.agents.get(agent_key)
            if message is None:
                message = self.append("system", "Subtask", **common, id=f"agent:{turn_id}:{agent_id}")
                self.agents[agent_key] = message
            message.update(status=data.get("status", ""), content=data.get("task_summary") or message["content"])
        elif event == "ChildAgentMessage":
            self.append("assistant", data.get("content", ""), **common, source="agent", status="completed")
        elif event == "MessagesReceived":
            for item in data.get("items", []):
                if isinstance(item, dict) and item.get("role") != "developer":
                    text = content_text(item.get("content"))
                    envelope = collaboration_messages(text) if data.get("source") != "user" else None
                    if envelope:
                        for report in envelope:
                            sender = report["sender_id"]
                            if agent_id == "main" and report["recipient_id"] == "parent":
                                if not any(m.get("agent_id") == sender and m.get("turn_id") == turn_id and m["content"] == report["content"] for m in self.messages):
                                    self.append("assistant", report["content"], turn_id=turn_id, agent_id=sender, source="agent", status="completed",
                                                sender_id=sender, recipient_id=report["recipient_id"])
                            elif agent_id != "main":
                                self.append("user", report["content"], **common, source="agent", status="completed",
                                            sender_id=sender, recipient_id=report["recipient_id"])
                        continue
                    if text:
                        self.append(
                            "user" if data.get("source") == "user" else "system",
                            text,
                            **common,
                            source=data.get("source"),
                            status="completed",
                        )
        elif event in {"TurnCompleted", "TurnFailed", "TurnCancelled", "TurnSettled"}:
            terminal_status = (
                data.get("status", "completed")
                if event == "TurnSettled"
                else {
                    "TurnCompleted": "completed",
                    "TurnFailed": "failed",
                    "TurnCancelled": "cancelled",
                }[event]
            )
            for message in self.messages:
                matches = message.get("turn_id") == turn_id and (
                    agent_id == "main"
                    or (
                        message.get("agent_id") == agent_id
                        and message.get("child_turn_id", turn_id) == execution_id
                    )
                )
                unfinished = message.get("status") in {
                    "pending",
                    "running",
                    "streaming",
                    "approval",
                } or (
                    message.get("status") == "denied"
                    and not message.get("tool_finished")
                )
                if matches and unfinished and not message.get("tool_finished"):
                    message["status"] = (
                        "completed"
                        if terminal_status == "completed"
                        else terminal_status
                    )
            if event == "TurnFailed":
                self.append(
                    "system",
                    data.get("message", "Execution failed"),
                    **common,
                    status="failed",
                )
