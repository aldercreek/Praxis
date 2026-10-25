from __future__ import annotations

import json

from praxis_agent.mcp_results import decode_tool_output


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


def tool_result_status(result: dict, previous: str = "") -> str:
    output = result.get("output", "")
    structured = result.get("structured_output") or (
        output if isinstance(output, dict) else {}
    )
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

    def __init__(self, history=()):
        self.messages: list[dict] = []
        self.responses: dict[tuple, dict] = {}
        self.last_response: dict[tuple, dict] = {}
        self.tools: dict[tuple, dict] = {}
        for item in history:
            self._restore(item)

    def append(self, role, content, **extra):
        message = {
            "id": f"message-{len(self.messages) + 1}",
            "role": role,
            "content": content,
            **extra,
        }
        self.messages.append(message)
        return message

    def _restore(self, item):
        if not isinstance(item, dict):
            return
        role, kind = item.get("role"), item.get("type")
        if role in {"user", "assistant", "system"}:
            text = content_text(item.get("content"))
            if text:
                self.append(role, text, agent_id="main", status="completed")
        elif kind == "function_call":
            try:
                arguments = json.loads(item.get("arguments", "{}"))
            except (ValueError, TypeError):
                arguments = {"raw": item.get("arguments", "")}
            self.tools[("main", item["call_id"])] = self.append(
                "tool",
                "",
                tool_name=item.get("name", ""),
                call_id=item["call_id"],
                agent_id="main",
                arguments=arguments,
                status="completed",
                tool_finished=False,
            )
        elif kind == "function_call_output":
            message = self.tools.get(("main", item.get("call_id")))
            if message is not None:
                try:
                    result = decode_tool_output(item.get("output", "{}"))
                except (ValueError, TypeError):
                    result = {"output": item.get("output", ""), "success": False}
                output = (
                    result.get("output", "") if isinstance(result, dict) else result
                )
                message["content"] = (
                    output
                    if isinstance(output, str)
                    else json.dumps(output, ensure_ascii=False)
                )
                message["status"] = (
                    tool_result_status(result) if isinstance(result, dict) else "failed"
                )
                message["tool_finished"] = True

    def consume(self, event: str, turn_id: str, agent_id: str, data: dict):
        common = {"turn_id": turn_id, "agent_id": agent_id}
        execution_id = data.get("child_turn_id") or turn_id
        key = (agent_id, execution_id)
        if data.get("child_turn_id"):
            common["child_turn_id"] = data["child_turn_id"]
        if event == "TurnStarted":
            self.append(
                "user",
                data["task"],
                **common,
                status="completed",
                id=f"user:{turn_id}:{agent_id}:{execution_id}",
            )
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
                    status=tool_result_status(result, message["status"]),
                    tool_finished=True,
                )
                if result.get("structured_output"):
                    message["structured_output"] = result["structured_output"]
                if result.get("file_change"):
                    message["file_change"] = result["file_change"]
        elif event == "MessagesReceived":
            for item in data.get("items", []):
                if isinstance(item, dict) and item.get("role") != "developer":
                    text = content_text(item.get("content"))
                    if text:
                        self.append(
                            "user" if data.get("source") == "user" else "system",
                            text,
                            **common,
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
