from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI, OpenAIError

from praxis_agent.config import ModelConfig


_DEFAULT_INSTRUCTIONS = """You are Praxis, a coding agent working in a local workspace.
Use the provided tools to inspect and modify files when needed.
All tool paths must be relative to the workspace root.
After modifying code, run relevant checks when they are available.
Do not retry a command that the user declined.
Do not claim that you changed or inspected something unless a tool result confirms it.
When the task is complete, give a concise summary of the result.
"""


@dataclass(frozen=True, slots=True)
class FunctionCall:
    """表示模型请求执行的一次函数工具调用。"""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """保存 Agent 循环需要使用的一次模型响应。"""

    output_items: tuple[object, ...]
    output_text: str
    function_calls: tuple[FunctionCall, ...]


class ModelClient(Protocol):
    """定义 Agent 所依赖的最小模型客户端接口。"""

    def create_response(
        self,
        input_items: Sequence[object],
        tools: Sequence[dict[str, object]],
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
        instructions: str = _DEFAULT_INSTRUCTIONS,
    ) -> None:
        client_options: dict[str, object] = {"api_key": config.api_key}
        if config.base_url is not None:
            client_options["base_url"] = config.base_url

        self._client = OpenAI(**client_options)
        self._model = config.model
        self._instructions = instructions

    def create_response(
        self,
        input_items: Sequence[object],
        tools: Sequence[dict[str, object]],
    ) -> ModelResponse:
        """调用模型，并提取文本与全部函数调用。"""
        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=self._instructions,
                input=list(input_items),
                tools=list(tools),
            )
        except OpenAIError as error:
            raise ModelClientError(f"model request failed: {error}") from error

        output_items = tuple(response.output)
        function_calls = tuple(
            FunctionCall(
                call_id=item.call_id,
                name=item.name,
                arguments=item.arguments,
            )
            for item in output_items
            if getattr(item, "type", None) == "function_call"
        )
        return ModelResponse(
            output_items=output_items,
            output_text=(response.output_text or "").strip(),
            function_calls=function_calls,
        )
