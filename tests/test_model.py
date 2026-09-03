from types import SimpleNamespace

from praxis_agent.config import ModelConfig
from praxis_agent.model import OpenAIModelClient


class FakeResponses:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    def create(self, **arguments: object) -> object:
        self.arguments = arguments
        function_call = SimpleNamespace(
            type="function_call",
            call_id="call-1",
            name="read_file",
            arguments='{"relative_path":"README.md"}',
        )
        return SimpleNamespace(output=[function_call], output_text="")


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

    response = client.create_response(
        [{"role": "user", "content": "read README"}],
        [{"type": "function", "name": "read_file"}],
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
    assert response.function_calls[0].name == "read_file"
    assert response.function_calls[0].call_id == "call-1"


def _capture_options(
    client: FakeOpenAI,
    options: dict[str, object],
) -> FakeOpenAI:
    client.options = options
    return client
