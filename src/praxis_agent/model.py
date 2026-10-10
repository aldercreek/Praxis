from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI, OpenAIError
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseFunctionToolCall,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)

from praxis_agent.config import ModelConfig
from praxis_agent.deepseek import DeepSeekResponseNormalizer
from praxis_agent.prompts import BASE_INSTRUCTIONS
from praxis_agent.multi_agent.models import ExecutionControl


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
        client_options: dict[str, object] = {"api_key": config.api_key}
        if execution_control is not None:
            # Worker cancellation must not wait through the SDK's long default retry cycle.
            client_options.update(timeout=60.0, max_retries=0)
        if config.base_url is not None:
            client_options["base_url"] = config.base_url

        self._client = OpenAI(**client_options)
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
        try:
            options = {} if max_output_tokens is None else {"max_output_tokens": max_output_tokens}
            if self._reasoning_effort is not None:
                options["reasoning"] = {"effort": self._reasoning_effort}
            stream = self._client.responses.create(
                model=self._model,
                instructions=self._instructions,
                input=[_response_input(item) for item in input_items],
                tools=list(tools),
                stream=True,
                **options,
            )
            try:
                response = _consume_stream(
                    stream, on_text_delta,
                    check_execution=control.check if control is not None else None,
                )
            finally:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
        except OpenAIError as error:
            raise ModelClientError(f"model request failed: {error}") from error

        if self._response_normalizer is not None:
            response = self._response_normalizer.normalize(response)
        return _build_model_response(response)

    def close(self) -> None:
        self._client.close()


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


def _consume_stream(
    stream: Iterable[ResponseStreamEvent],
    on_text_delta: TextDeltaHandler | None,
    *,
    check_execution: Callable[[], None] | None = None,
) -> Response:
    """消费 SDK 事件流，并返回完成事件携带的完整响应。"""
    completed_response: Response | None = None

    for event in stream:
        if check_execution is not None:
            check_execution()
        if isinstance(event, ResponseTextDeltaEvent):
            if event.delta and on_text_delta is not None:
                on_text_delta(event.delta)
        elif isinstance(event, ResponseCompletedEvent):
            completed_response = event.response
        elif isinstance(event, ResponseIncompleteEvent):
            raise ModelClientError(f"model response incomplete: {event.response.incomplete_details}")
        elif isinstance(event, (ResponseErrorEvent, ResponseFailedEvent)):
            message = _stream_error_message(event)
            raise ModelClientError(f"model request failed: {message}")

    if completed_response is None:
        raise ModelClientError(
            "model stream ended without a completed response"
        )
    return completed_response


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
