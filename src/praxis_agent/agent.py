import json
from dataclasses import dataclass
from pathlib import Path

from praxis_agent.config import ModelConfig
from praxis_agent.model import (
    FunctionCall,
    ModelClient,
    OpenAIModelClient,
)
from praxis_agent.tools import ToolRegistry, create_default_registry


class AgentError(RuntimeError):
    """表示 Agent 无法正常完成当前任务。"""


@dataclass(slots=True)
class Agent:
    model_client: ModelClient
    tool_registry: ToolRegistry
    max_turns: int = 30
    name: str = "Praxis"

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")

    def run(self, task: str) -> str:
        """运行模型与工具之间的循环，直到模型返回最终文本。"""
        normalized_task = task.strip()
        if not normalized_task:
            raise ValueError("task must not be empty")

        input_items: list[object] = [
            {"role": "user", "content": normalized_task}
        ]
        tool_definitions = self.tool_registry.definitions()

        for _ in range(self.max_turns):
            response = self.model_client.create_response(
                input_items,
                tool_definitions,
            )
            # 官方协议要求把模型本轮输出原样加入下一轮输入。
            input_items.extend(response.output_items)

            if not response.function_calls:
                if response.output_text:
                    return response.output_text
                raise AgentError("model returned no text and no function calls")

            # 一次响应可能包含多个工具调用，需要逐个执行并回传结果。
            for function_call in response.function_calls:
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": function_call.call_id,
                        "output": self._execute_function_call(function_call),
                    }
                )

        raise AgentError(
            f"agent exceeded the maximum of {self.max_turns} model turns"
        )

    def _execute_function_call(self, function_call: FunctionCall) -> str:
        """解析模型参数、运行工具，并返回 JSON 格式结果。"""
        try:
            arguments = json.loads(function_call.arguments)
        except json.JSONDecodeError as error:
            return _encode_tool_result(
                False,
                f"invalid JSON arguments: {error.msg}",
            )

        if not isinstance(arguments, dict):
            return _encode_tool_result(
                False,
                "tool arguments must be a JSON object",
            )

        result = self.tool_registry.execute(function_call.name, arguments)
        return _encode_tool_result(result.success, result.output)


def _encode_tool_result(success: bool, output: str) -> str:
    """把统一工具结果编码成模型能够读取的 JSON 字符串。"""
    return json.dumps(
        {"success": success, "output": output},
        ensure_ascii=False,
    )


def create_default_agent(workspace: Path) -> Agent:
    """使用环境配置和内置文件工具创建默认 Agent。"""
    config = ModelConfig.from_env()
    return Agent(
        model_client=OpenAIModelClient(config),
        tool_registry=create_default_registry(workspace),
    )
