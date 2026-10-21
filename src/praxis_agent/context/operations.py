from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json


def item_payload(item: object) -> dict[str, object]:
    """读取字典或 SDK 消息；返回副本，避免修改完整历史。"""
    if isinstance(item, Mapping):
        return dict(item)
    return item.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class ContextOperation:
    """一条可持久化的视图操作：把目标消息替换成一条新消息。"""

    operation_id: str
    action: str
    message_ids: tuple[str, ...] = ()
    instructions: str = ""
    replacement: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ContextOperation:
        values = dict(data)
        ids = values.get("message_ids", [])
        if not isinstance(ids, list) or any(
            not isinstance(mid, str) or not mid for mid in ids
        ):
            raise ValueError("invalid context operation message_ids")
        values["message_ids"] = tuple(ids)
        replacement = values.get("replacement")
        if replacement is not None:
            if not isinstance(replacement, Mapping):
                raise ValueError("invalid replacement message")
            if not isinstance(replacement.get("message_id"), str) or not isinstance(
                replacement.get("payload"), Mapping
            ):
                raise ValueError("replacement requires message_id and payload")
            values["replacement"] = dict(replacement)
        operation = cls(**values)
        if operation.action not in {"delete", "compress"}:
            raise ValueError("invalid context operation action")
        if operation.replacement is None or not operation.message_ids:
            raise ValueError("context operation requires targets and a replacement")
        return operation


def render_message(
    message: Mapping[str, object], operation_id: str | None = None,
) -> dict[str, object]:
    """临时把 ID 写入正文，发送给模型时不携带额外协议字段。"""
    payload = dict(message["payload"])
    label = f"[message_id={message['message_id']}"
    if operation_id is not None:
        label += f", operation_id={operation_id}"
    label += "]\n"
    if payload.get("type") == "function_call_output":
        output = payload.get("output")
        if isinstance(output, str):
            payload["output"] = label + output
        elif isinstance(output, list):
            payload["output"] = [{"type": "input_text", "text": label}, *output]
    elif payload.get("role") in {"user", "assistant"}:
        content = payload.get("content")
        if isinstance(content, str):
            payload["content"] = label + content
        elif isinstance(content, list):
            blocks = [dict(block) for block in content]
            for block in blocks:
                if isinstance(block.get("text"), str):
                    block["text"] = label + block["text"]
                    break
            else:
                kind = "output_text" if payload["role"] == "assistant" else "input_text"
                block = {"type": kind, "text": label}
                if kind == "output_text":
                    block["annotations"] = []
                blocks.insert(0, block)
            payload["content"] = blocks
        # 修改过文本的消息不再引用服务端旧输出对象的 ID。
        payload.pop("id", None)
    return payload


def is_editable(payload: Mapping[str, object]) -> bool:
    if payload.get("type") == "function_call_output":
        return isinstance(payload.get("output"), (str, list))
    if payload.get("role") not in {"user", "assistant"}:
        return False
    content = payload.get("content")
    return isinstance(content, str) or (
        isinstance(content, list) and bool(content)
        and all(block.get("type") in {"input_text", "output_text"} for block in content)
    )


def compression_reasoning_ids(
    messages: Sequence[dict[str, object]], selected_ids: set[str],
) -> set[str]:
    """整组助手输出被压缩时，附带移除其 reasoning；部分工具结果压缩时保留。"""
    selected_calls = {
        message["payload"].get("call_id") for message in messages
        if message["message_id"] in selected_ids
        and message["payload"].get("type") == "function_call_output"
    }
    reasoning_ids: set[str] = set()
    for index, message in enumerate(messages):
        if message["payload"].get("type") != "reasoning":
            continue
        outputs = []
        for following in messages[index + 1:]:
            payload = following["payload"]
            if payload.get("type") == "function_call" or payload.get("role") == "assistant":
                outputs.append(following)
            else:
                break
        if outputs and all(
            output["message_id"] in selected_ids
            or (output["payload"].get("type") == "function_call"
                and output["payload"].get("call_id") in selected_calls)
            for output in outputs
        ):
            reasoning_ids.add(message["message_id"])
    return reasoning_ids


@dataclass(slots=True)
class ContextOperations:
    """上下文操作层：重放操作生成视图，永远不覆盖原始历史。"""

    records: list[ContextOperation] = field(default_factory=list)
    persist: Callable[[ContextOperation], None] | None = None

    @property
    def next_id(self) -> str:
        return f"op{len(self.records) + 1}"

    def replay(self, operations: Sequence[ContextOperation]) -> None:
        self.records = list(operations)

    def record(self, operation: ContextOperation) -> None:
        if operation.operation_id != self.next_id:
            raise ValueError("context operations changed; retry with the current view")
        if self.persist is not None:
            self.persist(operation)
        self.records.append(operation)

    def clear(self) -> None:
        self.records.clear()

    def project(
        self, history: Sequence[object],
    ) -> list[dict[str, object]]:
        payloads = [item_payload(item) for item in history]
        # sequence 从 1 开始且完整历史只追加，因此这些 ID 不随压缩改变。
        messages = [
            {"message_id": f"m{index}", "payload": payload}
            for index, payload in enumerate(payloads, 1)
        ]
        for operation in self.records:
            targets = set(operation.message_ids)
            positions = [i for i, message in enumerate(messages) if message["message_id"] in targets]
            if len(positions) != len(targets) or not positions:
                raise ValueError(f"context operation has missing targets: {operation.operation_id}")
            if operation.replacement is None:
                raise ValueError("context operation has no replacement message")
            # 工具结果被替换时，一起从视图移除对应调用，避免留下未配对调用。
            # 这里只改变模型视图，完整历史中的调用与结果仍然保留。
            removed_calls = {
                messages[index]["payload"].get("call_id") for index in positions
                if messages[index]["payload"].get("type") == "function_call_output"
            }
            projected = []
            pending_calls: set[str] = set()
            replacement_pending = False
            for index, message in enumerate(messages):
                is_paired_call = (
                    message["payload"].get("type") == "function_call"
                    and message["payload"].get("call_id") in removed_calls
                )
                if message["message_id"] not in targets and not is_paired_call:
                    projected.append(message)
                    payload = message["payload"]
                    if payload.get("type") == "function_call":
                        pending_calls.add(payload["call_id"])
                    elif payload.get("type") == "function_call_output":
                        pending_calls.discard(payload["call_id"])
                # 非连续选择也只替换目标，保留中间未选择的消息。
                if index == positions[-1]:
                    replacement_pending = True
                # 部分工具结果被替换时，摘要必须等其余调用结果全部返回后再插入。
                # 否则普通 assistant 消息会切断网关要求连续的工具调用/结果序列。
                if replacement_pending and not pending_calls:
                    projected.append(operation.replacement)
                    replacement_pending = False
            if replacement_pending:
                raise ValueError("cannot insert context replacement before pending tool results")
            messages = projected
        # Successful direct summaries live in their replacement messages. Retain the
        # protocol call/result pair, but show a reference instead of a second full copy.
        replacements = {
            op.operation_id: op.replacement["message_id"]
            for op in self.records if op.action == "compress" and op.replacement is not None
        }
        applied_calls = {}
        for payload in payloads:
            if payload.get("type") != "function_call_output":
                continue
            try:
                result = json.loads(payload.get("output", ""))
                if not isinstance(result, dict) or result.get("success") is not True:
                    continue
                receipt = json.loads(result.get("output", ""))
                mid = replacements.get(receipt.get("operation_id"))
            except (ValueError, TypeError, AttributeError):
                continue
            if mid is not None:
                applied_calls[payload.get("call_id")] = mid
        for index, message in enumerate(messages):
            payload = message["payload"]
            mid = applied_calls.get(payload.get("call_id"))
            if payload.get("type") != "function_call" or payload.get("name") != "compress" or mid is None:
                continue
            try:
                arguments = json.loads(payload.get("arguments", ""))
            except (ValueError, TypeError):
                continue
            if isinstance(arguments, dict) and "summary" in arguments:
                arguments["summary"] = f"[Submitted summary saved as {mid}; original arguments remain in history.]"
                messages[index] = {**message, "payload": {
                    **payload, "arguments": json.dumps(arguments, ensure_ascii=False),
                }}
        return messages
