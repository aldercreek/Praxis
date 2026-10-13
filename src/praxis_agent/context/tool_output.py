"""工具结果的展示兜底；原文和上下文操作记录均不改动。"""

import json
from collections.abc import Sequence


def _preview(message: dict[str, object], limit: int) -> dict[str, object]:
    payload = message["payload"]
    output = payload.get("output")
    if payload.get("type") != "function_call_output" or not isinstance(output, str):
        return message
    # 即使正文只能展示极少字符，也保留执行状态和命令退出码。
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
        if isinstance(body, str) and body.startswith("exit_code: "):
            status += body.split("\n", 1)[0] + "; "
        if isinstance(body, str):
            # 按解码后的正文限长，避免 JSON 转义把已限长的文件结果再次截断。
            # 文件的行号头不占正文预算，但最多豁免 512 字符。
            header_chars = body.find("\n") + 1 if body.startswith("[lines ") else 0
            header_chars = min(header_chars, 512)
            if len(body) <= limit + header_chars:
                return message
            encoded = json.dumps(body, ensure_ascii=False)
            start = output.find(encoded)
            if start >= 0:
                text = body

    if len(text) <= limit:
        return message

    message_id = message["message_id"]
    notice = (
        f"[Tool result preview: {status}shown first {limit} characters "
        f"of {len(text)}. Full original result is saved as {message_id}.]\n\n"
    )
    return {**message, "payload": {**payload, "output": notice + text[:limit]}}


def preview_tool_outputs(
    messages: Sequence[dict[str, object]], max_chars: int,
) -> list[dict[str, object]]:
    """单条结果只展示固定开头，保留 call_id 和原始消息 ID。"""
    return [_preview(message, max_chars) for message in messages]
