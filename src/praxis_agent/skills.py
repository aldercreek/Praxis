"""Discover skill metadata and load versioned body pages by name."""

from __future__ import annotations

import json
import logging
import hashlib
from io import StringIO
from typing import TextIO
from dataclasses import dataclass
from pathlib import Path

import yaml


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    path: Path


def _read_metadata(path: Path) -> Skill:
    # Stop at the closing delimiter: discovering a skill must not load its body.
    with path.open(encoding="utf-8-sig") as stream:
        return _read_header(stream, path)


def _read_header(stream: TextIO, path: Path) -> Skill:
    if stream.readline().strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter")
    lines = []
    for line in stream:
        if line.strip() == "---":
            break
        lines.append(line)
    else:
        raise ValueError("YAML frontmatter has no closing delimiter")
    metadata = yaml.safe_load("".join(lines))
    if not isinstance(metadata, dict):
        raise ValueError("YAML frontmatter must be a mapping")
    for field, limit in (("name", 64), ("description", 1024)):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError(f"{field} must be a non-empty string of at most {limit} characters")
    name = metadata["name"].strip()
    if name != path.parent.name:
        raise ValueError("skill name must match its directory name")
    return Skill(name, " ".join(metadata["description"].split()), path.resolve())


def discover_skills(project_root: Path, *, user_root: Path | None = None) -> tuple[Skill, ...]:
    """Project entries override valid user entries by name; bad entries only warn."""
    roots = (
        user_root if user_root is not None else Path.home() / ".agents" / "skills",
        project_root / ".agents" / "skills",
    )
    skills: dict[str, Skill] = {}
    for root in roots:
        try:
            directories = sorted(root.iterdir(), key=lambda entry: entry.name)
        except FileNotFoundError:
            continue
        except OSError as error:
            logger.warning("Cannot scan skills directory %s: %s", root, error)
            continue
        for directory in directories:
            if not directory.is_dir():
                continue
            path = directory / "SKILL.md"
            try:
                skill = _read_metadata(path)
            except (OSError, ValueError, yaml.YAMLError) as error:
                logger.warning("Skipping skill %s: %s", path, error)
                continue
            skills[skill.name] = skill
    return tuple(skills[name] for name in sorted(skills))


def format_skill_catalog(skills: tuple[Skill, ...]) -> str:
    """Paths are returned by load_skill rather than duplicated in the catalog."""
    if not skills:
        return "Available skills: none discovered."
    entries = [json.dumps({"name": skill.name, "description": skill.description},
                          ensure_ascii=False) for skill in skills]
    return "Available skills (use load_skill by name):\n" + "\n".join(entries)


def create_skill_tool(skills: tuple[Skill, ...], *, max_chars: int):
    from praxis_agent.tools.registry import Tool, ToolContext

    catalog = {skill.name: skill for skill in skills}

    def load_skill(context: ToolContext, name: str, offset: int = 0, version: str = "") -> str:
        if name not in catalog:
            raise ValueError(f"Unknown or unavailable skill: {name}. Use a name from the skill catalog.")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if offset and not version:
            raise ValueError("Continuation requires the version returned by load_skill; otherwise restart at offset=0.")
        path = catalog[name].path
        raw = path.read_bytes()
        current_version = hashlib.sha256(raw).hexdigest()
        if version and version != current_version:
            raise ValueError("Skill file changed. Discard earlier pages and restart with offset=0, version=''.")
        stream = StringIO(raw.decode("utf-8-sig"))
        skill = _read_header(stream, path)
        if skill.name != name:
            raise ValueError("Skill name changed; restart the session to refresh the catalog.")
        body = stream.read()
        if not body.strip():
            raise ValueError("Skill body is empty")
        if offset >= len(body):
            raise ValueError(f"offset is past the body (length {len(body)}); use the returned next_offset.")

        def render(length: int) -> str:
            end = offset + length
            return json.dumps({
                "name": name, "path": str(path), "resource_directory": str(path.parent),
                "version": current_version, "offset": offset, "total_chars": len(body),
                "eof": end == len(body), "next_offset": None if end == len(body) else end,
                "body": body[offset:end],
            }, ensure_ascii=False)

        # Budget the complete serialized result, including paths and cursor metadata.
        # The normal tool-output preview can then retain every page without truncation.
        low, high = 0, min(len(body) - offset, max_chars)
        while low < high:
            middle = (low + high + 1) // 2
            if len(render(middle)) <= max_chars:
                low = middle
            else:
                high = middle - 1
        if low == 0:
            raise ValueError("Tool output budget cannot fit skill metadata and body; increase max_tool_output_chars.")
        return render(low)

    return Tool(
        name="load_skill",
        description=("Load a catalog skill's Markdown body by name, with its source and resource directory. "
                     "First call: offset=0, version=''. If eof=false, continue with the returned next_offset "
                     "and version using this same tool until eof=true. Offsets count body characters, not lines. "
                     "If the file changes, restart. Read referenced files with read_file; do not load all resources."),
        parameters={"type": "object", "properties": {
            "name": {"type": "string", "description": "Exact name from the skill catalog."},
            "offset": {"type": "integer", "minimum": 0, "description": "0 initially; returned next_offset for continuation."},
            "version": {"type": "string", "description": "Empty initially; copy the returned version for continuation."},
        }, "required": ["name", "offset", "version"], "additionalProperties": False},
        handler=load_skill, parallel_safe=True,
    )
