"""Recognize model-only collaboration envelopes at the display boundary."""

import json

PREFIX = "[Agent collaboration messages; source-labelled task data]\n"


def collaboration_messages(text):
    if not isinstance(text, str) or not text.startswith(PREFIX):
        return None
    try:
        items = json.loads(text[len(PREFIX):])
    except ValueError:
        return None
    if not isinstance(items, list) or not items:
        return None
    if not all(
        isinstance(item, dict)
        and all(isinstance(item.get(key), str) for key in ("sender_id", "recipient_id", "content"))
        for item in items
    ):
        return None
    return items
