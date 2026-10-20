import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Protocol

from openai import AsyncOpenAI, OpenAIError
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseTextDeltaEvent,
)

from praxis_agent.config import ModelConfig
from praxis_agent.deepseek import DeepSeekResponseNormalizer
from praxis_agent.execution import await_cancellable, current_handle, run_operation
from praxis_agent.multi_agent.models import ExecutionControl
from praxis_agent.prompts import BASE_INSTRUCTIONS

TextDeltaHandler = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class FunctionCall:
    """表示模型请求执行的一次函数工具调用。"""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """保存模型响应中由服务端统计的 Token 用量。"""

    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """保存 Agent 循环需要使用的一次模型响应。"""

    output_items: tuple[object, ...]
    output_text: str
    function_calls: tuple[FunctionCall, ...]
    response_id: str | None = None
    model: str | None = None
    usage: TokenUsage | None = None


class ModelClient(Protocol):
    """定义 Agent 所依赖的最小模型客户端接口。"""

    def create_response(
        self,
        input_items: Sequence[object],
        tools: Sequence[dict[str, object]],
        *,
        on_text_delta: TextDeltaHandler | None = None,
        max_output_tokens: int | None = None,
    ) -> ModelResponse:
        """根据对话输入和工具定义生成下一次模型响应。"""
        ...


class ModelClientError(RuntimeError):
    """表示模型请求或响应处理失败。"""


class OpenAIModelClient:
    """使用 OpenAI Python SDK 调用 Responses API。"""

    def __init__(
        self,
        config: ModelConfig,
        instructions: str = BASE_INSTRUCTIONS,
        *,
        reasoning_effort: str | None = None,
        execution_control: ExecutionControl | None = None,
    ) -> None:
        # SDK 提供有限重试、指数退避和抖动；await_cancellable 也会取消退避等待。
        client_options: dict[str, object] = {"api_key": config.api_key, "max_retries": 5}
        if execution_control is not None:
            client_options["timeout"] = 60.0
        if config.base_url is not None:
            client_options["base_url"] = config.base_url

        self._client = AsyncOpenAI(**client_options)
        self._loop = None
        self._thread = None
        self._loop_lock = Lock()
        self._closed = False
        self._model = config.model
        self._instructions = instructions
        self._execution_control = execution_control
        # None 沿用服务端默认值；摘要客户端显式传入 "none" 关闭 thinking。
        self._reasoning_effort = reasoning_effort
        self._response_normalizer = (
            DeepSeekResponseNormalizer() if "deepseek" in config.model.lower() else None
        )

    def create_response(
        self,
        input_items: Sequence[object],
        tools: Sequence[dict[str, object]],
        *,
        on_text_delta: TextDeltaHandler | None = None,
        max_output_tokens: int | None = None,
    ) -> ModelResponse:
        """流式调用模型，并提取最终文本与全部函数调用。"""
        control = self._execution_control
        if control is not None:
            control.check()
        def request(handle):
            self._ensure_loop()
            async def consume():
                return await await_cancellable(self._request(
                    input_items, tools, on_text_delta, max_output_tokens,
                ), handle)
            try:
                return asyncio.run_coroutine_threadsafe(consume(), self._loop).result()
            except OpenAIError as error:
                raise ModelClientError(f"model request failed: {error}") from error

        handle = current_handle()
        response = request(handle) if handle is not None else run_operation("model request", request)
        if self._response_normalizer is not None:
            response = self._response_normalizer.normalize(response)
        return _build_model_response(response)

    async def _request(self, input_items, tools, on_text_delta, max_output_tokens):
        options = {} if max_output_tokens is None else {"max_output_tokens": max_output_tokens}
        if self._reasoning_effort is not None:
            options["reasoning"] = {"effort": self._reasoning_effort}
        stream = await self._client.responses.create(
            model=self._model,
            instructions=self._instructions,
            input=[_response_input(item) for item in input_items],
            tools=list(tools),
            stream=True,
            **options,
        )
        completed = None
        try:
            async for event in stream:
                if isinstance(event, ResponseTextDeltaEvent):
                    if event.delta and on_text_delta is not None:
                        on_text_delta(event.delta)
                elif isinstance(event, ResponseCompletedEvent):
                    completed = event.response
                elif isinstance(event, ResponseIncompleteEvent):
                    raise ModelClientError(f"model response incomplete: {event.response.incomplete_details}")
                elif isinstance(event, (ResponseErrorEvent, ResponseFailedEvent)):
                    raise ModelClientError(f"model request failed: {_stream_error_message(event)}")
        finally:
            await stream.close()
        if completed is None:
            raise ModelClientError("model stream ended without a completed response")
        return completed

    def _ensure_loop(self):
        with self._loop_lock:
            if self._closed:
                raise ModelClientError("model client is closed")
            if self._loop is not None:
                return
            ready = Event()
            def serve():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                ready.set()
                try:
                    loop.run_forever()
                finally:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.run_until_complete(loop.shutdown_default_executor())
                    loop.close()
            self._thread = Thread(target=serve, name="praxis-model", daemon=True)
            self._thread.start()
            ready.wait()

    def close(self) -> None:
        with self._loop_lock:
            if self._closed:
                return
            self._closed = True
        if self._loop is None:
            asyncio.run(self._client.close())
        else:
            try:
                asyncio.run_coroutine_threadsafe(self._client.close(), self._loop).result()
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join()


def _response_input(item: object) -> dict[str, object]:
    """回传响应时省略状态和空可选字段，兼容网关较窄的输入协议。

    保留 reasoning 的 id、正文及 encrypted_content，也保留消息 phase。
    只清理协议层；不解析或修改工具参数、结果里的业务 JSON。
    """
    payload = dict(item) if isinstance(item, Mapping) else item.model_dump(mode="json")
    payload = {key: value for key, value in payload.items()
               if key != "status" and value is not None}
    for field in ("content", "summary"):
        if isinstance(payload.get(field), list):
            payload[field] = [
                {key: value for key, value in block.items() if value is not None}
                if isinstance(block, Mapping) else block
                for block in payload[field]
            ]
    return payload


def _stream_error_message(
    event: ResponseErrorEvent | ResponseFailedEvent,
) -> str:
    """从普通错误事件或失败响应事件中提取错误信息。"""
    if isinstance(event, ResponseErrorEvent):
        return event.message

    if event.response.error is None:
        return "unknown streaming error"
    return event.response.error.message


def _build_model_response(response: Response) -> ModelResponse:
    """把 SDK 的完整响应转换成 Agent 使用的稳定结构。"""
    output_items = tuple(response.output)
    function_calls = tuple(
        FunctionCall(
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments,
        )
        for item in output_items
        if isinstance(item, ResponseFunctionToolCall)
    )
    output_text = response.output_text or ""
    usage = response.usage
    return ModelResponse(
        output_items=output_items,
        output_text=output_text.strip(),
        function_calls=function_calls,
        response_id=response.id,
        model=response.model,
        usage=(
            None
            if usage is None
            else TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            )
        ),
    )
