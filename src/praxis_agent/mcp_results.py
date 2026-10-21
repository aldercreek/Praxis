"""Keep MCP data intact for code, and prepare content for model consumption."""

import json

from praxis_agent.tools.registry import ToolResult


def _text(value: str) -> dict:
    return {"type": "input_text", "text": value}


def _binary(data: str, mime: str) -> dict:
    if mime.startswith("image/"):
        return {"type": "input_image", "image_url": f"data:{mime};base64,{data}", "detail": "auto"}
    if mime == "application/pdf":
        return {"type": "input_file", "filename": "mcp-resource.pdf",
                "file_data": f"data:{mime};base64,{data}"}
    if mime.startswith("audio/"):
        return _text(f"[Audio: {mime}; not provided to the model: the current Responses interface does not support audio input.]")
    return _text(f"[Binary resource: {mime}; this content type is not supported for model input.]")


def _content_blocks(result) -> list[dict]:
    blocks = []
    for block in result.content:
        if block.type == "text":
            blocks.append(_text(block.text))
        elif block.type in {"image", "audio"}:
            if block.data:
                blocks.append(_binary(block.data, block.mime_type))
            else:
                blocks.append(_text(f"[{block.type}: {block.mime_type}; no data returned.]"))
        elif block.type == "resource_link":
            blocks.append(_text(f"[Resource link: {block.name}; {block.uri}]"))
            if block.description:
                blocks.append(_text(block.description))
        elif block.type == "resource":
            resource = block.resource
            blocks.append(_text(f"[Resource: {resource.uri}]"))
            if hasattr(resource, "text"):
                blocks.append(_text(resource.text))
            else:
                blocks.append(_binary(resource.blob, resource.mime_type or "application/octet-stream"))
        else:
            blocks.append(_text(f"[MCP content type {block.type!r} is not supported for model input.]"))
    if result.structured_content is not None:
        blocks.append(_text("Structured content:\n" + json.dumps(result.structured_content, ensure_ascii=False)))
    return blocks or [_text("[MCP tool returned no content.]")]


def content_summary(blocks: list[dict]) -> str:
    """A readable projection that never dumps binary data into UI/history text."""
    parts = []
    for block in blocks:
        if block.get("type") == "input_text":
            parts.append(block["text"])
        elif block.get("type") == "input_image":
            url = block.get("image_url", "")
            mime = url[5:].split(";", 1)[0] if url.startswith("data:") else "image"
            parts.append(f"[Image: {mime}]")
        elif block.get("type") == "input_file":
            parts.append(f"[File: {block.get('filename', 'resource')}]")
    return "\n".join(parts)


def mcp_tool_result(result, *, programmatic: bool = False) -> ToolResult:
    data = result.model_dump(mode="json", by_alias=True, exclude_none=True)
    if programmatic and not result.is_error:
        # No display conversion or text serialization on the successful code path.
        return ToolResult(True, "", structured_output=data)
    blocks = _content_blocks(result)
    if result.is_error:
        blocks.insert(0, _text("MCP tool reported an execution error."))
    return ToolResult(not result.is_error, content_summary(blocks), structured_output=data,
                      model_content=None if programmatic else blocks)


def decode_tool_output(output) -> dict:
    """Read both legacy string outputs and persisted multimodal tool outputs."""
    if isinstance(output, str):
        return json.loads(output)
    status = json.loads(output[0]["text"])
    return {**status, "output": content_summary(output[1:])}
