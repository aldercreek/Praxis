import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from praxis_agent.agent import Agent, AgentError
from praxis_agent.events import (
    AgentEvent,
    ApprovalRequested,
    ApprovalResolved,
    AssistantTextDelta,
    AssistantTextCompleted,
    EventSink,
    ModelRequestCompleted,
    ModelRequestStarted,
    ToolCallPrepared,
    ToolFinished,
    ToolStarted,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
)
from praxis_agent.model import FunctionCall, ModelResponse, TextDeltaHandler
from praxis_agent.tools import create_default_registry


class FakeModelClient:
    """返回预设响应的模型客户端，并记录每次请求的输入和工具定义。

    该替身使测试可以精确断言 Agent 发送给模型的上下文，而无须发起网络请求。
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.inputs: list[list[object]] = []
        self.tools: list[list[dict[str, object]]] = []

    def create_response(
        self,
        input_items: Sequence[object],
        tools: Sequence[dict[str, object]],
        *,
        on_text_delta: TextDeltaHandler | None = None,
    ) -> ModelResponse:
        self.inputs.append(list(input_items))
        self.tools.append(list(tools))
        response = next(self._responses)
        # 模拟流式输出：模型有文本时，将完整文本交给 Agent 注册的回调。
        if response.output_text and on_text_delta is not None:
            on_text_delta(response.output_text)
        return response


class RecordingEventSink(EventSink):
    """记录 Agent 发布的事件，供断言事件顺序使用。"""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


def _agent(
    tmp_path: Path,
    responses: list[ModelResponse],
    *,
    max_turns: int = 10,
    event_sink: EventSink | None = None,
) -> tuple[Agent, FakeModelClient]:
    """构造使用临时工具工作区和预设模型响应的 Agent。"""
    model_client = FakeModelClient(responses)
    agent = Agent(
        model_client=model_client,
        tool_registry=create_default_registry(tmp_path),
        max_turns=max_turns,
        event_sink=event_sink,
    )
    return agent, model_client


# 基础对话与会话历史。
def test_agent_returns_the_models_final_text(tmp_path: Path) -> None:
    event_sink = RecordingEventSink()
    agent, model_client = _agent(
        tmp_path,
        [ModelResponse((), "Repository inspected.", ())],
        event_sink=event_sink,
    )

    result = agent.run("inspect repository")

    assert result == "Repository inspected."
    assert model_client.inputs == [
        [{"role": "user", "content": "inspect repository"}]
    ]
    assert len(model_client.tools[0]) == 6
    assert [type(event) for event in event_sink.events] == [
        TurnStarted,
        ModelRequestStarted,
        AssistantTextDelta,
        ModelRequestCompleted,
        AssistantTextCompleted,
        TurnCompleted,
    ]
    assert len({event.turn_id for event in event_sink.events}) == 1
    assert event_sink.events[0].task == "inspect repository"
    assert event_sink.events[1].request_number == 1
    assert event_sink.events[1].input_items == (
        {"role": "user", "content": "inspect repository"},
    )
    assert event_sink.events[2].text == "Repository inspected."
    assert event_sink.events[3].response is not None
    assert event_sink.events[3].response.output_text == (
        "Repository inspected."
    )
    assert event_sink.events[4].text == "Repository inspected."


def test_agent_keeps_history_between_tasks(tmp_path: Path) -> None:
    first_output_item = {
        "type": "message",
        "role": "assistant",
        "content": "First response",
    }
    agent, model_client = _agent(
        tmp_path,
        [
            ModelResponse((first_output_item,), "First response", ()),
            ModelResponse((), "Second response", ()),
        ],
    )

    agent.run("first task")
    result = agent.run("follow-up task")

    assert result == "Second response"
    assert model_client.inputs[1] == [
        {"role": "user", "content": "first task"},
        first_output_item,
        {"role": "user", "content": "follow-up task"},
    ]


def test_agent_can_clear_history(tmp_path: Path) -> None:
    agent, model_client = _agent(
        tmp_path,
        [
            ModelResponse((), "First response", ()),
            ModelResponse((), "Fresh response", ()),
        ],
    )

    agent.run("first task")
    agent.clear_history()
    result = agent.run("fresh task")

    assert result == "Fresh response"
    assert model_client.inputs[1] == [
        {"role": "user", "content": "fresh task"}
    ]


def test_agent_rejects_an_empty_task(tmp_path: Path) -> None:
    agent, _ = _agent(tmp_path, [])

    with pytest.raises(ValueError, match="task must not be empty"):
        agent.run("   ")


# 工具调用：成功执行、参数错误和用户拒绝审批。
def test_agent_executes_a_tool_and_returns_its_result_to_model(
    tmp_path: Path,
) -> None:
    event_sink = RecordingEventSink()
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    function_call_item = {
        "type": "function_call",
        "call_id": "call-1",
        "name": "read_file",
        "arguments": '{"relative_path":"notes.txt"}',
    }
    agent, model_client = _agent(
        tmp_path,
        [
            ModelResponse(
                output_items=(function_call_item,),
                output_text="",
                function_calls=(
                    FunctionCall(
                        call_id="call-1",
                        name="read_file",
                        arguments='{\"relative_path\":\"notes.txt\"}',
                    ),
                ),
            ),
            ModelResponse((), "The file contains hello.", ()),
        ],
        event_sink=event_sink,
    )

    result = agent.run("read notes.txt")

    assert result == "The file contains hello."
    second_input = model_client.inputs[1]
    assert second_input[1] == function_call_item
    tool_output = second_input[2]
    assert tool_output["type"] == "function_call_output"
    assert tool_output["call_id"] == "call-1"
    assert json.loads(tool_output["output"]) == {
        "success": True,
        "output": "[lines 1-1 of 1]\nhello\n",
    }
    assert [type(event) for event in event_sink.events] == [
        TurnStarted,
        ModelRequestStarted,
        ModelRequestCompleted,
        ToolCallPrepared,
        ToolStarted,
        ToolFinished,
        ModelRequestStarted,
        AssistantTextDelta,
        ModelRequestCompleted,
        AssistantTextCompleted,
        TurnCompleted,
    ]
    prepared = event_sink.events[3]
    assert prepared.call_id == "call-1"
    assert prepared.tool_name == "read_file"
    assert prepared.arguments == {"relative_path": "notes.txt"}
    assert prepared.requires_approval is False


def test_agent_can_execute_a_write_tool(tmp_path: Path) -> None:
    agent, model_client = _agent(
        tmp_path,
        [
            ModelResponse(
                output_items=(),
                output_text="",
                function_calls=(
                    FunctionCall(
                        call_id="call-1",
                        name="create_file",
                        arguments=(
                            '{"relative_path":"created.txt",'
                            '"content":"created by Praxis\\n"}'
                        ),
                    ),
                ),
            ),
            ModelResponse((), "File created.", ()),
        ],
    )

    result = agent.run("create created.txt")

    assert result == "File created."
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == (
        "created by Praxis\n"
    )
    tool_output = model_client.inputs[1][-1]
    assert json.loads(tool_output["output"])["success"] is True


def test_agent_returns_invalid_arguments_to_the_model(tmp_path: Path) -> None:
    agent, model_client = _agent(
        tmp_path,
        [
            ModelResponse(
                output_items=(),
                output_text="",
                function_calls=(
                    FunctionCall("call-1", "read_file", "not-json"),
                ),
            ),
            ModelResponse((), "I could not read the file.", ()),
        ],
    )

    result = agent.run("read a file")

    assert result == "I could not read the file."
    tool_output = model_client.inputs[1][-1]
    decoded_output = json.loads(tool_output["output"])
    assert decoded_output["success"] is False
    assert decoded_output["output"].startswith("invalid JSON arguments:")


def test_agent_asks_for_approval_and_respects_a_decline(
    tmp_path: Path,
) -> None:
    event_sink = RecordingEventSink()
    approval_requests: list[str] = []

    def decline_command(name: str, arguments: object) -> bool:
        del arguments
        approval_requests.append(name)
        return False

    agent, model_client = _agent(
        tmp_path,
        [
            ModelResponse(
                output_items=(),
                output_text="",
                function_calls=(
                    FunctionCall(
                        "call-1",
                        "run_command",
                        '{"argv":["not-a-real-command"],'
                        '"relative_path":".","timeout_seconds":10}',
                    ),
                ),
            ),
            ModelResponse((), "Command was not run.", ()),
        ],
        event_sink=event_sink,
    )
    agent.approve_tool_call = decline_command

    result = agent.run("run a command")

    assert result == "Command was not run."
    assert approval_requests == ["run_command"]
    assert [type(event) for event in event_sink.events] == [
        TurnStarted,
        ModelRequestStarted,
        ModelRequestCompleted,
        ToolCallPrepared,
        ApprovalRequested,
        ApprovalResolved,
        ToolFinished,
        ModelRequestStarted,
        AssistantTextDelta,
        ModelRequestCompleted,
        AssistantTextCompleted,
        TurnCompleted,
    ]
    approval = event_sink.events[5]
    assert approval.approved is False
    tool_finished = event_sink.events[6]
    assert tool_finished.result.success is False
    tool_output = model_client.inputs[1][-1]
    decoded_output = json.loads(tool_output["output"])
    assert decoded_output["success"] is False
    assert "declined by the user" in decoded_output["output"]


# 防止模型持续请求工具导致无限循环。
def test_agent_stops_after_the_maximum_number_of_turns(
    tmp_path: Path,
) -> None:
    event_sink = RecordingEventSink()
    repeated_call = ModelResponse(
        output_items=(),
        output_text="",
        function_calls=(FunctionCall("call-1", "missing", "{}"),),
    )
    agent, _ = _agent(
        tmp_path,
        [repeated_call],
        max_turns=1,
        event_sink=event_sink,
    )

    with pytest.raises(AgentError, match="maximum of 1 model turns"):
        agent.run("keep calling tools")

    assert isinstance(event_sink.events[-1], TurnFailed)
    assert event_sink.events[-1].error_type == "AgentError"
