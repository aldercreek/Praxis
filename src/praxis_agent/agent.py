import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from praxis_agent.config import ModelConfig
from praxis_agent.events import (
    AgentEvent,
    AssistantTextDelta,
    AssistantTextCompleted,
    EventSink,
    ModelRequestCompleted,
    ModelRequestStarted,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
)
from praxis_agent.model import (
    FunctionCall,
    ModelClient,
    OpenAIModelClient,
)
from praxis_agent.prompts import build_instructions
from praxis_agent.runtime import collect_runtime_context
from praxis_agent.tool_execution import (
    ToolApprovalHandler,
    ToolCallExecutor,
)
from praxis_agent.tools import ToolRegistry, ToolResult, create_default_registry


class AgentError(RuntimeError):
    """表示 Agent 无法正常完成当前任务。"""


@dataclass(slots=True)
class Agent:
    model_client: ModelClient
    tool_registry: ToolRegistry
    max_turns: int = 30
    name: str = "Praxis"
    event_sink: EventSink | None = None
    approve_tool_call: ToolApprovalHandler | None = None
    _input_items: list[object] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")

    def run(self, task: str) -> str:
        """运行一轮任务，并保留内容供同一会话的后续任务使用。"""
        normalized_task = task.strip()
        if not normalized_task:
            raise ValueError("task must not be empty")

        turn_id = uuid4().hex
        self._emit(TurnStarted(turn_id, normalized_task))

        try:
            output_text = self._run_model_loop(
                turn_id,
                normalized_task,
            )
            self._emit(AssistantTextCompleted(turn_id, output_text))
            self._emit(TurnCompleted(turn_id))
            return output_text
        except Exception as error:
            self._emit(
                TurnFailed(
                    turn_id,
                    type(error).__name__,
                    str(error),
                )
            )
            raise

    def clear_history(self) -> None:
        """清空当前进程内保存的对话历史。"""
        self._input_items.clear()

    def _run_model_loop(
        self,
        turn_id: str,
        task: str,
    ) -> str:
        """持续请求模型并执行工具，直到模型生成最终文本。"""
        self._input_items.append({"role": "user", "content": task})
        tool_definitions = self.tool_registry.definitions()
        tool_executor = ToolCallExecutor(
            tool_registry=self.tool_registry,
            publish_event=self._emit,
            approve_tool_call=self.approve_tool_call,
        )

        for request_number in range(1, self.max_turns + 1):
            self._emit(
                ModelRequestStarted(
                    turn_id,
                    request_number,
                    tuple(self._input_items),
                )
            )
            response = self.model_client.create_response(
                self._input_items,
                tool_definitions,
                on_text_delta=lambda text: self._emit(
                    AssistantTextDelta(
                        turn_id,
                        request_number,
                        text,
                    )
                ),
            )
            self._emit(
                ModelRequestCompleted(
                    turn_id,
                    request_number,
                    response,
                )
            )
            # 官方协议要求把模型本轮输出原样加入下一轮输入。
            self._input_items.extend(response.output_items)

            if not response.function_calls:
                if response.output_text:
                    return response.output_text
                raise AgentError(
                    "model returned no text and no function calls"
                )

            # 一次响应可能包含多个工具调用，需要逐个执行并回传结果。
            for function_call in response.function_calls:
                result = tool_executor.execute(turn_id, function_call)
                self._input_items.append(
                    _build_function_call_output(function_call, result)
                )

        raise AgentError(
            f"agent exceeded the maximum of {self.max_turns} model turns"
        )

    def _emit(self, event: AgentEvent) -> None:
        """把事件发送给外部消费者；未配置消费者时保持静默。"""
        if self.event_sink is not None:
            self.event_sink.emit(event)


def _build_function_call_output(
    function_call: FunctionCall,
    result: ToolResult,
) -> dict[str, object]:
    """把工具结果转换成模型接口要求的输入项。"""
    return {
        "type": "function_call_output",
        "call_id": function_call.call_id,
        "output": json.dumps(
            {
                "success": result.success,
                "output": result.output,
            },
            ensure_ascii=False,
        ),
    }


def create_default_agent(
    workspace: Path,
    *,
    event_sink: EventSink | None = None,
    approve_tool_call: ToolApprovalHandler | None = None,
) -> Agent:
    """使用环境配置和内置文件工具创建默认 Agent。"""
    config = ModelConfig.from_env()
    runtime_context = collect_runtime_context(workspace)
    return Agent(
        model_client=OpenAIModelClient(
            config,
            instructions=build_instructions(runtime_context),
        ),
        tool_registry=create_default_registry(runtime_context.workspace_root),
        event_sink=event_sink,
        approve_tool_call=approve_tool_call,
    )
