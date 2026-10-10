"""SQLite FTS5/BM25 and name-based MCP tool discovery."""

from __future__ import annotations

import base64
import json
import logging
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import jieba


jieba.setLogLevel(logging.WARNING)


def tokens(text: str) -> list[str]:
    # Keep original identifiers as well as their underscore/camel-case components.
    def expand(match):
        original = match.group()
        split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", original).replace("_", " ")
        return original if split == original else original + " " + split
    expanded = re.sub(r"[A-Za-z][A-Za-z0-9_]*", expand, text)
    return [word.lower() for word in jieba.cut_for_search(expanded)
            if any(char.isalnum() for char in word)]


def parameter_text(schema: dict) -> list[str]:
    result = []
    for name, value in schema.get("properties", {}).items():
        if isinstance(value, dict):
            result.append(f"{name}: {str(value.get('description', ''))[:300]}")
            result.extend(parameter_text(value))
    if isinstance(schema.get("items"), dict):
        result.extend(parameter_text(schema["items"]))
    for kind in ("anyOf", "oneOf", "allOf"):
        for value in schema.get(kind, []):
            if isinstance(value, dict):
                result.extend(parameter_text(value))
    return result


class ToolSearchIndex:
    def __init__(self, path: Path, search_config: dict):
        self.path = path
        self.default_limit = int(search_config.get("limit", 5))
        if not 1 <= self.default_limit <= 20:
            raise ValueError("Invalid MCP search configuration")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def rebuild(self, records: list[dict]):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            # The catalog is disposable; replace its schema and contents atomically.
            db.execute("BEGIN IMMEDIATE")
            db.execute("DROP TABLE IF EXISTS tool_fts")
            db.execute("DROP TABLE IF EXISTS tools")
            db.execute("CREATE TABLE tools (id TEXT PRIMARY KEY, server TEXT, original TEXT, alias TEXT UNIQUE, description TEXT, schema_json TEXT, metadata_json TEXT, search_text TEXT, approval INTEGER)")
            db.execute("CREATE VIRTUAL TABLE tool_fts USING fts5(tool_id UNINDEXED, terms)")
            for record in records:
                db.execute("INSERT INTO tools VALUES (?,?,?,?,?,?,?,?,?)", (
                    record["id"], record["server"], record["original"], record["alias"], record["description"],
                    json.dumps(record["schema"], ensure_ascii=False), json.dumps(record["metadata"], ensure_ascii=False),
                    record["search_text"], record["approval"]))
                db.execute("INSERT INTO tool_fts VALUES (?,?)", (record["id"], " ".join(tokens(record["search_text"]))))

    def get(self, tool_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM tools WHERE id=?", (tool_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown tool identifier")
        return dict(row)

    def list_tools(self, server: str, cursor: str = "", limit: int = 10):
        if not 1 <= limit <= 20:
            raise ValueError("Invalid page limit")
        after = ""
        if cursor:
            try:
                value = json.loads(base64.b64decode(cursor, validate=True))
                if value["index"] != str(self.path) or value["server"] != server:
                    raise ValueError()
                after = value["after"]
                if not isinstance(after, str) or self.get(after)["server"] != server:
                    raise ValueError()
            except (ValueError, KeyError, TypeError):
                raise ValueError("Invalid cursor for this service/catalog") from None
        with self.connect() as db:
            rows = db.execute("SELECT * FROM tools WHERE server=? AND id>? ORDER BY id LIMIT ?",
                              (server, after, limit + 1)).fetchall()
        records = [dict(row) for row in rows[:limit]]
        next_cursor = ""
        if len(rows) > limit:
            next_cursor = base64.b64encode(json.dumps({"index": str(self.path), "server": server,
                "after": records[-1]["id"]}).encode()).decode()
        return records, next_cursor

    def search(self, query: str, mode="keyword", server="", limit=None) -> dict:
        query = query.strip()
        limit = self.default_limit if limit is None else limit
        if not query or len(query) > 2000:
            raise ValueError("query must contain 1-2000 characters")
        if mode not in {"keyword", "name"} or not 1 <= limit <= 20:
            raise ValueError("Invalid search mode or limit")
        with self.connect() as db:
            filters, args = ("WHERE server=?", [server]) if server else ("", [])
            if server and not db.execute("SELECT 1 FROM tools WHERE server=? LIMIT 1", (server,)).fetchone():
                return {"matches": [], "mode": mode, "notice": "No available tools for this server"}
            if mode == "name":
                rows = db.execute(f"SELECT id,original FROM tools {filters}", args).fetchall()
                q = query.casefold()
                def rank(row):
                    identity, original = row["id"].casefold(), row["original"].casefold()
                    if q == identity: return 0
                    if q == original: return 1
                    if original.startswith(q) or identity.startswith(q): return 2
                    if q in original or q in identity: return 3
                    return 4
                ranked = sorted((row for row in rows if rank(row) < 4), key=lambda row: (rank(row), row["id"]))
                return {"matches": [row["id"] for row in ranked[:limit]], "mode": mode}

            terms = list(dict.fromkeys(tokens(query)))
            expression = " OR ".join('"' + word.replace('"', '""') + '"' for word in terms)
            keyword = []
            if expression:
                keyword = [row["tool_id"] for row in db.execute(
                    "SELECT tool_fts.tool_id FROM tool_fts JOIN tools ON tools.id=tool_fts.tool_id "
                    "WHERE tool_fts MATCH ?" + (" AND tools.server=?" if server else "") +
                    " ORDER BY bm25(tool_fts), tool_fts.tool_id LIMIT ?", [expression, *args, limit])]
            return {"matches": keyword, "mode": mode}
