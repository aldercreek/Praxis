"""Shared MCP connections, started in the background and borrowed by agents."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from threading import Event, RLock, Thread
from uuid import uuid4

from praxis_agent.execution import await_cancellable, run_operation
from praxis_agent.mcp_results import mcp_tool_result
from praxis_agent.tool_search import ToolSearchIndex, parameter_text
from praxis_agent.tools.registry import Tool, ToolContext, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class MCPConnections:
    MAX_CONNECTING = 8
    MAX_ATTEMPTS = 3
    RETRY_DELAYS = (1, 2)

    def __init__(self, path: Path | None = None):
        self.path = path or Path.home() / ".praxis" / "mcp.json"
        self.config = json.loads(self.path.read_text(encoding="utf-8-sig")) if self.path.exists() else {}
        self.servers = self.config.get("servers", self.config.get("mcp_servers", {}))
        if not isinstance(self.servers, dict):
            raise ValueError("MCP servers must be an object")
        self.status: dict[str, dict] = {
            name: {"state": "disabled" if isinstance(config, dict) and not config.get("enabled", True) else "queued"}
            for name, config in self.servers.items()
        }
        self.state_lock = RLock()
        self.clients = {}
        self.definitions = {}
        self.locks = {}
        self.ready = Event()
        self.thread = None
        self.loop = None
        self.stopping = None
        self.closed = False
        self.index = None
        self.server_descriptions = {}
        self.aliases = set()

    def start(self):
        if self.closed:
            raise RuntimeError("MCP connections are closed")
        if self.thread is not None:
            return
        if not self.servers:
            self.ready.set()
            return
        # Register discovery tools before any agent can borrow this manager.
        self._build_index()
        self.thread = Thread(target=self._run, name="praxis-mcp", daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.state_lock:
            return {"servers": [{"name": name, **state} for name, state in self.status.items()]}

    def _state(self, name, state, **details):
        with self.state_lock:
            self.status[name] = {"state": state, **details}

    def _run(self):
        asyncio.run(self._serve())

    async def _serve(self):
        self.loop = asyncio.get_running_loop()
        self.stopping = asyncio.Event()
        if self.closed:
            self.ready.set()
            return
        semaphore = asyncio.Semaphore(self.MAX_CONNECTING)
        initialized = [asyncio.Event() for _ in self.servers]
        tasks = [asyncio.create_task(self._server(name, config, done, semaphore))
                 for (name, config), done in zip(self.servers.items(), initialized)]
        try:
            # Shutdown must also interrupt connections still starting or retrying.
            async def mark_ready():
                await asyncio.gather(*(done.wait() for done in initialized))
                self.ready.set()
            readiness = asyncio.create_task(mark_ready())
            await self.stopping.wait()
        finally:
            readiness.cancel()
            for task in tasks:
                task.cancel()
            await asyncio.gather(readiness, *tasks, return_exceptions=True)
            self.ready.set()

    async def _server(self, name, config, initialized, semaphore):
        connected = False
        try:
            if self.status[name]["state"] == "disabled":
                return
            if not isinstance(config, dict):
                self._state(name, "failed", error="InvalidConfiguration")
                return
            for key, default in (("startup_timeout_sec", 30), ("tool_timeout_sec", 60)):
                value = config.get(key, default)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 3600:
                    self._state(name, "failed", error="InvalidConfiguration")
                    return
            for attempt in range(self.MAX_ATTEMPTS):
                acquired = False
                try:
                    await semaphore.acquire()
                    acquired = True
                    self._state(name, "connecting")
                    # Each failed attempt releases its own transport before retrying.
                    async with AsyncExitStack() as stack:
                        async with asyncio.timeout(config.get("startup_timeout_sec", 30)):
                            client, definitions = await self._connect(name, config, stack)
                            self.definitions[name] = definitions
                            self._build_index()
                            with self.state_lock:
                                self.clients[name] = client
                                self.server_descriptions[name] = str(config.get("description") or client.instructions or name)[:400]
                                self.locks[name] = asyncio.Lock()
                                self._state(name, "connected")
                            connected = True
                        semaphore.release()
                        acquired = False
                        initialized.set()
                        await self.stopping.wait()
                        return
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    # Exception text can contain credentials from URLs or server responses.
                    state = "disconnected" if connected else "failed" if attempt + 1 == self.MAX_ATTEMPTS else "retrying"
                    self._state(name, state, error=type(error).__name__)
                    logger.warning("MCP server %s unavailable (%s)", name, type(error).__name__)
                    if connected or state == "failed":
                        return
                finally:
                    if acquired:
                        semaphore.release()
                    self.definitions.pop(name, None)
                await asyncio.sleep(self.RETRY_DELAYS[attempt])
        finally:
            with self.state_lock:
                self.clients.pop(name, None)
                if connected or self.closed and self.status[name]["state"] in {"queued", "connecting", "retrying"}:
                    self._state(name, "disconnected")
            initialized.set()

    async def _connect(self, name, config, stack):
        import httpx2
        from mcp import Client
        from mcp.client.stdio import StdioServerParameters
        from mcp.client.streamable_http import streamable_http_client

        async def receive(message):
            # SDK transport errors can arrive separately from the pending call.
            if isinstance(message, Exception) and self.status[name]["state"] == "connected":
                self._state(name, "disconnected", error=type(message).__name__)

        if "url" in config:
            headers = dict(config.get("http_headers", {}))
            for key, variable in config.get("env_http_headers", {}).items():
                headers[key] = os.environ[variable]
            if config.get("bearer_token_env_var"):
                headers["Authorization"] = "Bearer " + os.environ[config["bearer_token_env_var"]]
            http = await stack.enter_async_context(httpx2.AsyncClient(headers=headers, timeout=30, follow_redirects=False))
            transport = streamable_http_client(config["url"], http_client=http)
        else:
            env = {key: os.environ[key] for key in config.get("env_vars", []) if key in os.environ}
            env.update(config.get("env", {}))
            transport = StdioServerParameters(command=config["command"], args=config.get("args", []),
                                              env=env, cwd=config.get("cwd", str(self.path.parent)))
        client = await stack.enter_async_context(Client(transport, cache=None, message_handler=receive,
                                                     read_timeout_seconds=config.get("tool_timeout_sec", 60)))
        definitions = []
        cursor = None
        seen = set()
        while True:
            page = await client.list_tools(cursor=cursor)
            definitions.extend(page.tools)
            cursor = page.next_cursor
            if not cursor:
                break
            if cursor in seen:
                raise ValueError("Repeated MCP tools pagination cursor")
            seen.add(cursor)
        return client, definitions

    def _build_index(self):
        records = []
        aliases = set(self.aliases)
        for server, definitions in self.definitions.items():
            config = self.servers[server]
            for definition in definitions:
                original = definition.name
                if original in config.get("disabled_tools", []):
                    continue
                if "enabled_tools" in config and original not in config["enabled_tools"]:
                    continue
                tool_config = config.get("tools", {}).get(original, {})
                if tool_config.get("enabled") is False:
                    continue
                identity = f"{server}/{original}"
                raw_name = f"mcp_{server}_{original}"
                alias = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name)
                # Separators inside either name can otherwise collide depending
                # on which service finishes connecting first (a_b/c vs a/b_c).
                if len(alias) > 64 or alias != raw_name or "_" in server or "_" in original or alias in aliases:
                    alias = alias[:51] + "_" + hashlib.sha256(identity.encode()).hexdigest()[:12]
                if alias in aliases:
                    raise ValueError("Duplicate MCP tool identity")
                aliases.add(alias)
                description = definition.description or original
                text = f"Server: {server}\nTool: {original}\nDescription: {description}\nParameters:\n"
                text += "\n".join(parameter_text(definition.input_schema))
                records.append({"id": identity, "server": server, "original": original, "alias": alias,
                    "description": description, "schema": definition.input_schema,
                    "metadata": definition.model_dump(mode="json", by_alias=True, exclude_none=True),
                    "search_text": text, "approval": tool_config.get("approval_mode") != "approve"})
        if self.index is None:
            path = self.path.parent / "indexes" / f"{uuid4().hex}.sqlite"
            self.index = ToolSearchIndex(path, self.config.get("search", {}))
            self.index.rebuild([])
        self.index.add(records)
        self.aliases = aliases
        self.definitions.clear()

    def catalog(self):
        if self.index is None:
            return ""
        with self.state_lock:
            entries = [{"server": name, "description": description}
                       for name, description in sorted(self.server_descriptions.items())
                       if self.status.get(name, {}).get("state") == "connected"]
        if not entries:
            return ""
        return """<mcp>
1. Available services
   """ + json.dumps(entries, ensure_ascii=False) + """

2. Finding tools
   search_tools finds candidates; list_tools browses a specific service. Neither activates tools.
   Search using objects, actions, and key constraints. Use server to narrow the scope when the service is known.
   Adjust keywords or browse the service if the results are unsuitable.

3. Activation and execution
   activate_tools activates tools. Omitting arguments only activates them;
   the next request includes their full definitions.
   Providing arguments, even an empty object, executes immediately. Do this only when the arguments are known.
   Batch only independent calls. Verify the outcome after a failure or timeout before retrying.

4. Tool management
   Each agent manages its own activated tools.
   Use deactivate_tools to deactivate tools that are no longer needed.

5. Programmatic calls
   Call MCP management tools directly.
   In run_code, call activated tools using the name from the candidate results.
   Successful MCP calls in run_code always return the complete response dictionary: content, optional structuredContent, isError.
   Read dictionary fields directly. Raw media blocks are preserved; do not print large encoded payloads.
   Parse text blocks according to their actual format and follow the tool interface for remote pagination.
</mcp>"""

    def register(self, registry: ToolRegistry, *, check_execution=None):
        if self.index is None:
            return lambda: None

        active_aliases = set()

        def check(context):
            if context.check_execution:
                context.check_execution()
            if check_execution:
                check_execution()

        def available(server):
            if server not in self.status:
                raise ValueError("Unknown MCP server: " + server)
            if server not in self.clients or self.status[server]["state"] == "disconnected":
                raise ValueError("MCP server unavailable: " + server)

        def candidate(record):
            return {"tool_id": record["id"], "name": record["alias"],
                    "description": record["description"][:600],
                    "active": record["alias"] in registry.names}

        def load(record):
            def invoke(context: ToolContext, **arguments):
                return self.call(record["server"], record["original"], arguments,
                                 check_execution=lambda: check(context), parallel=context.parallel_tool_call,
                                 programmatic=context.programmatic_call)
            if record["alias"] in registry.names:
                return "already_active"
            registry.register(Tool(name=record["alias"], description=record["description"],
                parameters=json.loads(record["schema_json"]), handler=invoke, strict=False,
                requires_approval=bool(record["approval"]), parallel_safe=False, programmatic=True))
            active_aliases.add(record["alias"])
            return "activated"

        def search_tools(context: ToolContext, query: str, mode: str = "keyword", server: str = "", limit: int | None = None):
            check(context)
            if server:
                available(server)
            result = self.index.search(query, mode, server, limit)
            records = [self.index.get(identity) for identity in result["matches"]]
            result["matches"] = [candidate(record) for record in records if record["server"] in self.clients
                                 and self.status[record["server"]]["state"] != "disconnected"]
            return json.dumps(result, ensure_ascii=False)

        def list_tools(context: ToolContext, server: str, cursor: str = "", limit: int = 10):
            check(context)
            available(server)
            records, next_cursor = self.index.list_tools(server, cursor, limit)
            return json.dumps({"tools": [candidate(record) for record in records],
                               "next_cursor": next_cursor}, ensure_ascii=False)

        def activate_tools(context: ToolContext, tools: list[dict]):
            results = []
            attachments = []
            payload = {"results": results}
            for item in tools:
                try:
                    check(context)
                except RuntimeError:
                    payload["stopped"] = "Agent cancelled or closed; remaining items not processed"
                    break
                entry = {"tool_id": item["tool_id"], "activation": "failed", "execution": "not_requested"}
                results.append(entry)
                try:
                    record = self.index.get(item["tool_id"])
                    available(record["server"])
                    entry["activation"] = load(record)
                    if "arguments" in item:
                        if context.invoke_tool is None:
                            entry.update(execution="failed", output="Tool executor unavailable; not executed")
                        else:
                            result = context.invoke_tool(record["alias"], item["arguments"])
                            entry.update(execution="success" if result.success else "failed", output=result.output)
                            media = [block for block in result.model_content or []
                                     if block["type"] != "input_text"]
                            if media:
                                attachments.extend([
                                    {"type": "input_text", "text": f"Attachments from {record['id']}:"},
                                    *media,
                                ])
                except ValueError as error:
                    entry["error"] = str(error)
            output = json.dumps(payload, ensure_ascii=False)
            if attachments:
                return ToolResult(True, output, model_content=[
                    {"type": "input_text", "text": output}, *attachments,
                ])
            return output

        def deactivate_tools(context: ToolContext, tool_ids: list[str]):
            results = []
            for identity in tool_ids:
                check(context)
                try:
                    record = self.index.get(identity)
                    state = "deactivated" if record["alias"] in registry.names else "already_inactive"
                    registry.unregister(record["alias"])
                    active_aliases.discard(record["alias"])
                    results.append({"tool_id": identity, "status": state})
                except ValueError as error:
                    results.append({"tool_id": identity, "status": "failed", "error": str(error)})
            return json.dumps({"results": results}, ensure_ascii=False)

        def register(name, description, properties, required, handler):
            registry.register(Tool(name=name, description=description,
                parameters={"type": "object", "properties": properties, "required": required,
                            "additionalProperties": False}, handler=handler, strict=False))

        register("search_tools", "Search MCP candidates without activating or returning schemas. Use several specific keywords covering object, action and constraints, with useful synonyms or English terms when appropriate. Keyword (default) uses BM25; name matches identifiers/prefixes/substrings.", {
            "query": {"type": "string", "minLength": 1, "maxLength": 2000, "description": "Specific keywords for the object/action/constraints, or a tool name in name mode."},
            "mode": {"type": "string", "enum": ["keyword", "name"], "description": "keyword (default): BM25; name: identifier, prefix or substring matching."},
            "server": {"type": "string", "description": "Exact service name from the catalog; omit or use empty string to search all available services."},
            "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": 20, "description": "Candidate count; omit/null uses configured default. Candidates are not activated."}}, ["query"], search_tools)
        register("list_tools", "Browse a service's tools in stable pages without activating or returning schemas. Continue with next_cursor; empty means end.", {
            "server": {"type": "string", "minLength": 1, "description": "Exact service name from the catalog."}, "cursor": {"type": "string", "description": "Empty/omitted initially; copy next_cursor unchanged for the same service."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Page size, default 10; no activation."}}, ["server"], list_tools)
        register("activate_tools", "Activate MCP tools for this agent. Omit arguments to activate only. Supplying arguments (including {}) executes once through normal validation and approval. Use only known parameters, not guesses. Execution is not idempotent; no automatic retry. Batch independent items only.", {
            "tools": {"type": "array", "minItems": 1, "description": "Independent activation requests, processed sequentially; each may optionally execute once.", "items": {
                "type": "object", "properties": {"tool_id": {"type": "string", "minLength": 1, "description": "Exact server/tool identifier returned by search_tools or list_tools."},
                "arguments": {"type": "object", "description": "Omit to activate only. An object, including {}, executes once using already-known tool parameters; never guess."}}, "required": ["tool_id"],
                "additionalProperties": False}}}, ["tools"], activate_tools)
        register("deactivate_tools", "Remove specified MCP definitions from this agent's future requests. History and running calls remain; other agents are unaffected. Reactivate when needed.", {
            "tool_ids": {"type": "array", "minItems": 1, "description": "Exact server/tool identifiers to deactivate in this agent only.", "items": {"type": "string", "minLength": 1}}},
            ["tool_ids"], deactivate_tools)

        def release():
            for alias in tuple(active_aliases):
                registry.unregister(alias)
            active_aliases.clear()

        return release

    def call(self, server, tool, arguments, *, check_execution=None, parallel=False, programmatic=False):
        if self.closed or self.loop is None:
            return ToolResult(False, "MCP connections are closed")
        if check_execution:
            check_execution()
        def invoke(handle):
            async def owned_call():
                return await await_cancellable(self._call(
                    server, tool, arguments, parallel=parallel, programmatic=programmatic), handle)
            # Never cancel this proxy: its completion must acknowledge coroutine cleanup.
            return asyncio.run_coroutine_threadsafe(owned_call(), self.loop).result()
        return run_operation(f"MCP {server}/{tool}", invoke)

    async def _call(self, server, tool, arguments, *, parallel=False, programmatic=False):
        from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
        from httpx2 import TransportError
        from mcp.shared.exceptions import MCPError
        from mcp_types import CONNECTION_CLOSED

        try:
            if server not in self.clients:
                return ToolResult(False, "MCP server disconnected; no automatic retry was attempted")
            async with asyncio.timeout(self.servers[server].get("tool_timeout_sec", 60)):
                # Explicit parallel batches opt out of the ordinary per-server queue.
                async with nullcontext() if parallel else self.locks[server]:
                    if server not in self.clients:
                        return ToolResult(False, "MCP server disconnected; no automatic retry was attempted")
                    result = await self.clients[server].call_tool(tool, arguments)
            self._state(server, "connected")
            return mcp_tool_result(result, programmatic=programmatic)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if isinstance(error, (ConnectionError, BrokenResourceError, ClosedResourceError, EndOfStream, TransportError)) or (
                isinstance(error, MCPError) and error.code == CONNECTION_CLOSED
            ):
                self._state(server, "disconnected", error=type(error).__name__)
            return ToolResult(False, f"MCP call failed ({type(error).__name__}); remote completion may be unknown. No automatic retry.")

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.loop and self.stopping and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.stopping.set)
        if self.thread:
            self.thread.join(timeout=15)
            if self.thread.is_alive():
                logger.warning("MCP shutdown did not finish within 15 seconds")
