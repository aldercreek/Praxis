"""工具结果的展示兜底；原文和上下文操作记录均不改动。"""

import json
from collections.abc import Sequence


def _excerpt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head_chars = (limit + 1) // 2
    tail_chars = limit // 2
    tail = text[-tail_chars:] if tail_chars else ""
    return text[:head_chars] + f"\n\n[... {len(text) - limit} characters omitted ...]\n\n" + tail


def _preview(message: dict[str, object], limit: int) -> dict[str, object]:
    payload = message["payload"]
    output = payload.get("output")
    if payload.get("type") != "function_call_output" or not isinstance(output, str):
        return message
    status = ""
    text = output
    try:
        result = json.loads(output)
    except (ValueError, TypeError):
        result = None
    if isinstance(result, dict):
        if isinstance(result.get("success"), bool):
            status = f"success={str(result['success']).lower()}; "
        body = result.get("output")
        if isinstance(body, dict) and "status" in body:
            stdout, stderr = body["stdout"], body["stderr"]
            if len(stdout) + len(stderr) <= limit:
                return message
            # Split the log budget between streams, reusing space left by a short stream.
            stdout_limit = min(len(stdout), max(limit // 2, limit - len(stderr)))
            preview = {**body, "stdout": _excerpt(stdout, stdout_limit),
                       "stderr": _excerpt(stderr, limit - stdout_limit)}
            notice = f"[Tool result preview: log beginnings and endings; full original result is saved as {message['message_id']}.]\n\n"
            return {**message, "payload": {**payload, "output": notice + json.dumps(
                {**result, "output": preview}, ensure_ascii=False)}}
        if isinstance(body, str):
            # 按解码后的正文限长，避免 JSON 转义把已限长的文件结果再次截断。
            text = body

    if len(text) <= limit:
        return message

    message_id = message["message_id"]
    head_chars = (limit + 1) // 2
    tail_chars = limit // 2
    notice = (
        f"[Tool result preview: {status}shown first {head_chars} and last {tail_chars} characters "
        f"of {len(text)}. Full original result is saved as {message_id}.]\n\n"
    )
    return {**message, "payload": {**payload, "output": notice + _excerpt(text, limit)}}


def preview_tool_outputs(
    messages: Sequence[dict[str, object]], max_chars: int,
) -> list[dict[str, object]]:
    """展示长结果的首尾，保留执行状态、call_id 和完整历史的消息 ID。"""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    return [_preview(message, max_chars) for message in messages]
