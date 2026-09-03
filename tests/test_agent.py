import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from praxis_agent.agent import Agent, AgentError
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.tools import create_default_registry


class FakeModelClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.inputs: list[list[object]] = []
        self.tools: list[list[dict[str, object]]] = []

    def create_response(
        self,
        input_items: Sequence[object],
        tools: Sequence[dict[str, object]],
    ) -> ModelResponse:
        self.inputs.append(list(input_items))
        self.tools.append(list(tools))
        return next(self._responses)


def _agent(
    tmp_path: Path,
    responses: list[ModelResponse],
    *,
    max_turns: int = 10,
) -> tuple[Agent, FakeModelClient]:
    model_client = FakeModelClient(responses)
    agent = Agent(
        model_client=model_client,
        tool_registry=create_default_registry(tmp_path),
        max_turns=max_turns,
    )
    return agent, model_client


def test_agent_returns_the_models_final_text(tmp_path: Path) -> None:
    agent, model_client = _agent(
        tmp_path,
        [ModelResponse((), "Repository inspected.", ())],
    )

    result = agent.run("inspect repository")

    assert result == "Repository inspected."
    assert model_client.inputs == [
        [{"role": "user", "content": "inspect repository"}]
    ]
    assert len(model_client.tools[0]) == 6


def test_agent_rejects_an_empty_task(tmp_path: Path) -> None:
    agent, _ = _agent(tmp_path, [])

    with pytest.raises(ValueError, match="task must not be empty"):
        agent.run("   ")


def test_agent_executes_a_tool_and_returns_its_result_to_model(
    tmp_path: Path,
) -> None:
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
        "output": "hello\n",
    }


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
    tool_calls: list[str] = []
    tool_results: list[tuple[str, bool]] = []
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
    )
    agent.on_tool_call = lambda name, arguments: tool_calls.append(name)
    agent.on_tool_result = lambda name, result: tool_results.append(
        (name, result.success)
    )
    agent.approve_tool_call = decline_command

    result = agent.run("run a command")

    assert result == "Command was not run."
    assert tool_calls == ["run_command"]
    assert approval_requests == ["run_command"]
    assert tool_results == [("run_command", False)]
    tool_output = model_client.inputs[1][-1]
    decoded_output = json.loads(tool_output["output"])
    assert decoded_output["success"] is False
    assert "declined by the user" in decoded_output["output"]


def test_agent_stops_after_the_maximum_number_of_turns(
    tmp_path: Path,
) -> None:
    repeated_call = ModelResponse(
        output_items=(),
        output_text="",
        function_calls=(FunctionCall("call-1", "missing", "{}"),),
    )
    agent, _ = _agent(tmp_path, [repeated_call], max_turns=1)

    with pytest.raises(AgentError, match="maximum of 1 model turns"):
        agent.run("keep calling tools")
