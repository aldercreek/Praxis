"""Project persisted media onto the selected model and the Responses wire format."""

import base64
import binascii

from praxis_agent.context.model_catalog import ModelInputCapabilities


# Responses accepts images/files in function outputs, but has no audio input part.
_RESPONSE_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


def _notice(kind: str, reason: str) -> dict:
    return {"type": "input_text", "text": f"[{kind} not provided to the model: {reason}.]"}


def _valid_data_url(value: str) -> tuple[str, bool]:
    if not value.startswith("data:") or ";base64," not in value:
        return "", False
    mime, encoded = value[5:].split(";base64,", 1)
    try:
        return mime, bool(encoded) and bool(base64.b64decode(encoded, validate=True))
    except (ValueError, binascii.Error):
        return mime, False


def adapt_media_blocks(blocks: list[dict], capabilities: ModelInputCapabilities) -> list[dict]:
    adapted = []
    for block in blocks:
        kind = block.get("type")
        if kind == "input_image":
            if not capabilities.image_mime_types:
                adapted.append(_notice("Image", "image input is disabled or unconfirmed for this model"))
                continue
            url = block.get("image_url") or ""
            mime, valid = _valid_data_url(url)
            if url.startswith("data:"):
                if not valid:
                    adapted.append(_notice("Image", "invalid or empty base64 data"))
                    continue
                if mime not in capabilities.image_mime_types or mime not in _RESPONSE_IMAGE_TYPES:
                    adapted.append(_notice("Image", f"format {mime} is not supported by this model/interface"))
                    continue
            # MCP resource links never become remote image URLs automatically.
        elif kind == "input_file":
            data = block.get("file_data") or ""
            mime, valid = _valid_data_url(data)
            if not capabilities.pdf:
                adapted.append(_notice("File", "PDF input is disabled or unconfirmed for this model"))
                continue
            if not valid or mime != "application/pdf":
                adapted.append(_notice("File", "only valid base64 PDF data is supported here"))
                continue
        adapted.append(dict(block))
    return adapted
