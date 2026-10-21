import asyncio
from collections.abc import Iterator
from threading import Event, Thread

import pytest
import httpx2
from openai import AsyncOpenAI
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
from praxis_agent.context.model_catalog import MODEL_INPUT_CATALOG
from praxis_agent.execution import ExecutionControl, ExecutionStopped, Run
from praxis_agent.model import ModelClientError, OpenAIModelClient


@pytest.mark.parametrize("model", list(MODEL_INPUT_CATALOG) + ["auto(free)", "custom-model"])
def test_mcp_media_requests_for_all_model_configs(monkeypatch, model):
    import json
    from mcp.types import CallToolResult, ImageContent, AudioContent, TextContent, EmbeddedResource, BlobResourceContents
    from praxis_agent.agent import _build_function_call_output
    from praxis_agent.mcp_results import mcp_tool_result
    from praxis_agent.model import FunctionCall

    received = []
    body = _successful_stream_body()
    def handle(request):
        received.append(json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    monkeypatch.setattr("praxis_agent.model.AsyncOpenAI", lambda **options: AsyncOpenAI(
        **options, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle))))
    config = ModelConfig(api_key="test", model=model, base_url="https://example.test/v1")
    client = OpenAIModelClient(config)
    reply = CallToolResult(content=[
        TextContent(type="text", text="Report completed"),
        ImageContent(type="image", mime_type="image/png", data="iVBORw0KGgo="),
        AudioContent(type="audio", mime_type="audio/wav", data="YXVkaW8="),
        EmbeddedResource(type="resource", resource=BlobResourceContents(
            uri="file:///report.pdf", mime_type="application/pdf", blob="JVBERi0xLjQK")),
    ], structured_content={"count": 2})
    call = FunctionCall("c", "mcp_report", "{}")
    output = _build_function_call_output(call, mcp_tool_result(reply))
    try:
        client.create_response([
            {"type": "function_call", "call_id": "c", "name": "mcp_report", "arguments": "{}"}, output,
        ], [{"type": "function", "name": "mcp_report", "parameters": {"type": "object", "properties": {}}}])
    finally:
        client.close()
    request = received[0]
    assert request["model"] == model
    blocks = request["input"][-1]["output"]
    assert any(b["type"] == "input_image" for b in blocks) is bool(config.input_capabilities.image_mime_types)
    assert any(b["type"] == "input_file" for b in blocks) is config.input_capabilities.pdf
    text = "\n".join(b["text"] for b in blocks if b["type"] == "input_text")
    assert "Report completed" in text and '"count": 2' in text
    assert "interface does not support audio" in text
    assert "YXVkaW8=" not in json.dumps(request)


class FakeResponses:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    async def create(
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
        return FakeStream(
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

    async def close(self):
        pass


class FakeStream:
    def __init__(self, events):
        self.events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.events)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        pass


def test_openai_model_client_builds_and_parses_a_response(monkeypatch) -> None:
    sdk_client = FakeOpenAI()
    monkeypatch.setattr(
        "praxis_agent.model.AsyncOpenAI",
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
    client.close()

    assert sdk_client.options == {
        "api_key": "secret-key",
        "base_url": "https://example.com/v1",
        "max_retries": 5,
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


def _model_with_http_transport(monkeypatch, handler):
    monkeypatch.setattr(
        "praxis_agent.model.AsyncOpenAI",
        lambda **options: AsyncOpenAI(
            **options, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        ),
    )
    return OpenAIModelClient(
        ModelConfig(api_key="test", model="test", base_url="https://example.test/v1"),
        execution_control=ExecutionControl(),
    )


def _successful_stream_body():
    events = asyncio.run(FakeResponses().create())
    return "".join("data: " + event.model_dump_json() + "\n\n" for event in events.events).encode()


@pytest.mark.parametrize("failure", ["connection", "timeout", 408, 409, 429, 500, 503])
def test_model_retries_five_times_with_exponential_backoff(monkeypatch, failure):
    body = _successful_stream_body()
    calls, delays = [], []

    def handle(request):
        calls.append(request)
        if len(calls) <= 5:
            if failure == "connection":
                raise httpx2.ConnectError("connection failed", request=request)
            if failure == "timeout":
                raise httpx2.ReadTimeout("timed out before headers", request=request)
            return httpx2.Response(failure, json={"error": {"message": "try later"}})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    async def record_delay(seconds):
        delays.append(seconds)

    monkeypatch.setattr("openai._base_client.random", lambda: 0.0)
    monkeypatch.setattr("openai._base_client.anyio.sleep", record_delay)
    client = _model_with_http_transport(monkeypatch, handle)
    try:
        response = Run().execute(lambda: client.create_response([], []))
        assert response.output_text == "Reading README"
        assert len(calls) == 6
        assert delays == [0.5, 1.0, 2.0, 4.0, 8.0]
    finally:
        client.close()


@pytest.mark.parametrize("status, attempts", [(503, 6), (400, 1), (401, 1), (403, 1)])
def test_model_bounds_retries_and_rejects_non_retryable_errors(monkeypatch, status, attempts):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx2.Response(status, json={"error": {"message": "request failed"}})

    async def skip_delay(seconds):
        pass

    monkeypatch.setattr("openai._base_client.anyio.sleep", skip_delay)
    client = _model_with_http_transport(monkeypatch, handle)
    try:
        with pytest.raises(ModelClientError, match="model request failed"):
            Run().execute(lambda: client.create_response([], []))
        assert len(calls) == attempts
    finally:
        client.close()


def test_cancel_during_retry_after_stops_retry_and_keeps_client_reusable(monkeypatch):
    body = _successful_stream_body()
    entered = Event()
    calls, delays = [], []

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx2.Response(429, headers={"retry-after": "120"},
                                  json={"error": {"message": "try later"}})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    async def wait_for_retry(seconds):
        delays.append(seconds)
        entered.set()
        await asyncio.sleep(seconds)

    monkeypatch.setattr("openai._base_client.anyio.sleep", wait_for_retry)
    client = _model_with_http_transport(monkeypatch, handle)
    scope = Run()

    def execute():
        try:
            scope.execute(lambda: client.create_response([], []))
        except ExecutionStopped:
            pass

    thread = Thread(target=execute, daemon=True)
    thread.start()
    try:
        assert entered.wait(3)
        scope.cancel()
        with pytest.raises(ExecutionStopped):
            scope.finished.result(timeout=3)
        thread.join(3)
        assert not thread.is_alive()
        assert len(calls) == 1
        assert delays == [120.0]
        assert Run().execute(lambda: client.create_response([], [])).output_text == "Reading README"
        assert len(calls) == 2
    finally:
        scope.cancel()
        thread.join(3)
        client.close()


def test_stream_failure_after_text_is_not_retried(monkeypatch):
    calls, text = [], []
    event = next(asyncio.run(FakeResponses().create()).events)

    class BrokenStream(httpx2.AsyncByteStream):
        async def __aiter__(self):
            yield ("data: " + event.model_dump_json() + "\n\n").encode()
            raise httpx2.ReadError("stream disconnected", request=calls[-1])

    def handle(request):
        calls.append(request)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=BrokenStream())

    client = _model_with_http_transport(monkeypatch, handle)
    try:
        with pytest.raises(httpx2.ReadError, match="stream disconnected"):
            Run().execute(lambda: client.create_response([], [], on_text_delta=text.append))
        assert len(calls) == 1
        assert text == ["Reading "]
    finally:
        client.close()


@pytest.mark.parametrize("phase", ["headers", "stream"])
def test_cancel_network_wait_acknowledges_cleanup_before_future(monkeypatch, phase):
    entered, cleaning, release = Event(), Event(), Event()

    async def blocked():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await asyncio.to_thread(release.wait, 5)

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await blocked()

        async def close(self):
            pass

    class Responses:
        async def create(self, **kwargs):
            if phase == "headers":
                await blocked()
            return Stream()

    sdk = FakeOpenAI()
    sdk.responses = Responses()
    monkeypatch.setattr("praxis_agent.model.AsyncOpenAI", lambda **kwargs: sdk)
    client = OpenAIModelClient(ModelConfig(api_key="test", model="test"))
    scope = Run()
    def execute():
        try:
            scope.execute(lambda: client.create_response([], []))
        except ExecutionStopped:
            pass
    thread = Thread(target=execute, daemon=True)
    thread.start()
    try:
        assert entered.wait(3)
        scope.cancel()
        assert cleaning.wait(3)
        assert not scope.finished.done()
        assert not scope.handles[0].future.done()
        scope.cancel()
        release.set()
        with pytest.raises(ExecutionStopped):
            scope.finished.result(timeout=3)
        thread.join(3)
        assert not thread.is_alive()
        sdk.responses = FakeResponses()
        assert Run().execute(lambda: client.create_response([], [])).output_text == "Reading README"
    finally:
        release.set()
        thread.join(3)
        client.close()


def test_cancel_real_http_request_before_headers_and_reuse_client():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    entered, release = Event(), Event()
    events = asyncio.run(FakeResponses().create())
    body = "".join("data: " + event.model_dump_json() + "\n\n" for event in events.events).encode()

    class Handler(BaseHTTPRequestHandler):
        calls = 0

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            Handler.calls += 1
            if Handler.calls == 1:
                entered.set()
                release.wait(5)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving = Thread(target=server.serve_forever, daemon=True)
    serving.start()
    client = OpenAIModelClient(ModelConfig(
        api_key="test", model="test", base_url=f"http://127.0.0.1:{server.server_port}/v1",
    ))
    scope = Run()
    def execute():
        try:
            scope.execute(lambda: client.create_response([], []))
        except ExecutionStopped:
            pass
    thread = Thread(target=execute, daemon=True)
    thread.start()
    try:
        assert entered.wait(5)
        scope.cancel()
        with pytest.raises(ExecutionStopped):
            scope.finished.result(timeout=3)
        assert not release.is_set()  # Cancellation did not depend on a server response.
        assert Run().execute(lambda: client.create_response([], [])).output_text == "Reading README"
    finally:
        release.set()
        thread.join(3)
        client.close()
        server.shutdown()
        server.server_close()
        serving.join(3)
