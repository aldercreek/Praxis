"""Load user-wide instructions independently of the agent's working directory."""

import json
from pathlib import Path


def load_global_instructions() -> str:
    """Read a startup snapshot; missing/empty files need no prompt section."""
    path = Path.home() / ".praxis" / "AGENTS.md"
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Cannot read global instructions {path}: {error}") from error
    if not content.strip():
        return ""
    return (
        "<global_instructions>\n"
        "User-wide defaults. Explicit user requests take precedence; applicable directory\n"
        "instructions take precedence where they conflict with these defaults.\n"
        f"Source: {json.dumps(str(path), ensure_ascii=False)}\n\n"
        f"{content}\n"
        "</global_instructions>"
    )
