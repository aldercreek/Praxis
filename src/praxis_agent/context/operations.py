from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field


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
        return isinstance(payload.get("output"), str)
    if payload.get("role") not in {"user", "assistant"}:
        return False
    content = payload.get("content")
    return isinstance(content, str) or (
        isinstance(content, list) and bool(content)
        and all(block.get("type") in {"input_text", "output_text"} for block in content)
    )


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
            for index, message in enumerate(messages):
                is_paired_call = (
                    message["payload"].get("type") == "function_call"
                    and message["payload"].get("call_id") in removed_calls
                )
                if message["message_id"] not in targets and not is_paired_call:
                    projected.append(message)
                # 非连续选择也只替换目标，保留中间未选择的消息。
                if index == positions[-1]:
                    projected.append(operation.replacement)
            messages = projected
        return messages
