from collections.abc import Iterator

from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from praxis_agent.config import ModelConfig
from praxis_agent.model import OpenAIModelClient


class FakeResponses:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    def create(
        self,
        **arguments: object,
    ) -> Iterator[ResponseStreamEvent]:
        self.arguments = arguments
        function_call = ResponseFunctionToolCall(
            type="function_call",
            call_id="call-1",
            name="read_file",
            arguments='{"relative_path":"README.md"}',
        )
        message = ResponseOutputMessage(
            id="message-1",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text="Reading README",
                    type="output_text",
                )
            ],
            role="assistant",
            status="completed",
            type="message",
        )
        response = Response(
            id="response-1",
            created_at=0,
            model="example-model",
            object="response",
            output=[message, function_call],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            usage=ResponseUsage(
                input_tokens=12,
                input_tokens_details=InputTokensDetails(
                    cached_tokens=0,
                    cache_write_tokens=0,
                ),
                output_tokens=3,
                output_tokens_details=OutputTokensDetails(
                    reasoning_tokens=0,
                ),
                total_tokens=15,
            ),
        )
        return iter(
            [
                ResponseTextDeltaEvent(
                    content_index=0,
                    type="response.output_text.delta",
                    delta="Reading ",
                    item_id="message-1",
                    logprobs=[],
                    output_index=0,
                    sequence_number=0,
                ),
                ResponseTextDeltaEvent(
                    content_index=0,
                    type="response.output_text.delta",
                    delta="README",
                    item_id="message-1",
                    logprobs=[],
                    output_index=0,
                    sequence_number=1,
                ),
                ResponseCompletedEvent(
                    type="response.completed",
                    response=response,
                    sequence_number=2,
                ),
            ]
        )


class FakeOpenAI:
    def __init__(self, **options: object) -> None:
        self.options = options
        self.responses = FakeResponses()


def test_openai_model_client_builds_and_parses_a_response(monkeypatch) -> None:
    sdk_client = FakeOpenAI()
    monkeypatch.setattr(
        "praxis_agent.model.OpenAI",
        lambda **options: _capture_options(sdk_client, options),
    )
    client = OpenAIModelClient(
        ModelConfig(
            api_key="secret-key",
            model="example-model",
            base_url="https://example.com/v1",
        ),
        instructions="custom instructions",
    )

    text_deltas: list[str] = []
    response = client.create_response(
        [{"role": "user", "content": "read README"}],
        [{"type": "function", "name": "read_file"}],
        on_text_delta=text_deltas.append,
    )

    assert sdk_client.options == {
        "api_key": "secret-key",
        "base_url": "https://example.com/v1",
    }
    assert sdk_client.responses.arguments["model"] == "example-model"
    assert (
        sdk_client.responses.arguments["instructions"]
        == "custom instructions"
    )
    assert sdk_client.responses.arguments["stream"] is True
    assert text_deltas == ["Reading ", "README"]
    assert response.output_text == "Reading README"
    assert response.function_calls[0].name == "read_file"
    assert response.function_calls[0].call_id == "call-1"
    assert response.response_id == "response-1"
    assert response.model == "example-model"
    assert response.usage is not None
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 3
    assert response.usage.total_tokens == 15


def _capture_options(
    client: FakeOpenAI,
    options: dict[str, object],
) -> FakeOpenAI:
    client.options = options
    return client
