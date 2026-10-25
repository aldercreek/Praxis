import asyncio
import json
import socket
import sys
from collections import Counter
from threading import Event, Thread
from types import SimpleNamespace

import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp_types import Tool
from support.service import answer, wait_until

from praxis_agent.agent import Agent
from praxis_agent.mcp import MCPConnections
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.planning import RunState
from praxis_agent.tools import ToolContext, create_default_registry


def config_file(tmp_path, servers):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": servers}), encoding="utf-8")
    return path


class ControlledConnections(MCPConnections):
    RETRY_DELAYS = (0.01, 0.02)

    def __init__(self, path):
        super().__init__(path)
        self.attempts = Counter()
        self.cleaned = Counter()
        self.connecting = self.peak = 0
        self.release = Event()

    async def _connect(self, name, config, stack):
        name = config["command"]
        # A failed attempt must have disposed its transport before retrying.
        assert self.attempts[name] == self.cleaned[name]
        self.attempts[name] += 1
        self.connecting += 1
        self.peak = max(self.peak, self.connecting)

        async def cleanup():
            self.cleaned[name] += 1
        stack.push_async_callback(cleanup)
        try:
            if config.get("blocked"):
                while not self.release.is_set():
                    await asyncio.sleep(0.005)
            if self.attempts[name] <= config.get("failures", 0):
                raise ConnectionError("private-token-must-not-leak")
            await asyncio.sleep(0)
            return SimpleNamespace(instructions=name), [Tool(name="lookup", description=name, input_schema={"type": "object"})]
        finally:
            self.connecting -= 1


def test_start_is_nonblocking_and_limits_connection_attempts_to_eight(tmp_path):
    servers = {f"server{i}": {"command": f"server{i}", "blocked": True} for i in range(12)}
    servers["disabled"] = {"enabled": False, "command": "disabled"}
    mcp = ControlledConnections(config_file(tmp_path, servers))
    try:
        mcp.start()
        thread = mcp.thread
        mcp.start()
        assert mcp.thread is thread
        wait_until(lambda: mcp.connecting == 8)
        assert not mcp.ready.is_set()
        assert sum(state["state"] == "queued" for state in mcp.status.values()) == 4
        mcp.release.set()
        assert mcp.ready.wait(5)
        assert mcp.peak == 8
        assert len(mcp.clients) == 12
        assert "disabled" not in mcp.attempts
        assert mcp.snapshot()["servers"][-1] == {"name": "disabled", "state": "disabled"}
    finally:
        mcp.close()
    assert not mcp.thread.is_alive()
    assert mcp.cleaned == mcp.attempts


def test_retries_are_bounded_and_failures_do_not_block_other_services(tmp_path):
    mcp = ControlledConnections(config_file(tmp_path, {
        "eventual": {"command": "eventual", "failures": 2},
        "broken": {"command": "broken", "failures": 100},
        "fast": {"command": "fast"},
        "invalid": {"startup_timeout_sec": 0},
    }))
    try:
        mcp.start()
        assert mcp.ready.wait(5)
        assert mcp.attempts == {"eventual": 3, "broken": 3, "fast": 1}
        assert mcp.status["eventual"]["state"] == "connected"
        assert mcp.status["broken"]["state"] == "failed"
        assert mcp.status["invalid"]["state"] == "failed"
        assert mcp.cleaned == {"eventual": 2, "broken": 3}
        assert "private-token" not in json.dumps(mcp.snapshot())
        assert "fast" in mcp.catalog() and "eventual" in mcp.catalog()
    finally:
        mcp.close()
    assert mcp.cleaned == mcp.attempts


@pytest.mark.parametrize("state", ["connecting", "retrying"])
def test_shutdown_cancels_startup_and_backoff_without_waiting_for_timeouts(tmp_path, state):
    mcp = ControlledConnections(config_file(tmp_path, {
        "test": {"command": "test", "blocked": state == "connecting", "failures": 100}
    }))
    mcp.RETRY_DELAYS = (30, 30)
    mcp.start()
    try:
        wait_until(lambda: mcp.status["test"]["state"] == state)
    finally:
        mcp.close()
    assert not mcp.thread.is_alive()
    assert mcp.cleaned == mcp.attempts == {"test": 1}
    assert mcp.status["test"]["state"] == "disconnected"


def test_late_connections_are_visible_and_agent_activation_stays_independent(tmp_path):
    mcp = ControlledConnections(config_file(tmp_path, {"late": {"command": "late", "blocked": True}}))
    mcp.start()
    first, second = create_default_registry(), create_default_registry()
    release = mcp.register(first)
    release_second = mcp.register(second)
    context = ToolContext(tmp_path, RunState("test"))
    try:
        assert mcp.catalog() == ""
        assert json.loads(first.execute("search_tools", {"query": "lookup"}, context=context).output)["matches"] == []
        mcp.release.set()
        assert mcp.ready.wait(5)
        for registry in (first, second):
            result = registry.execute("list_tools", {"server": "late"}, context=context)
            assert json.loads(result.output)["tools"][0]["tool_id"] == "late/lookup"
        first.execute("activate_tools", {"tools": [{"tool_id": "late/lookup"}]}, context=context)
        assert "mcp_late_lookup" in first.names
        assert "mcp_late_lookup" not in second.names
        release()
        assert "mcp_late_lookup" not in first.names
        assert mcp.status["late"]["state"] == "connected"
    finally:
        release_second()
        mcp.close()


def test_catalog_is_refreshed_between_model_requests(tmp_path):
    catalog = "first service"
    seen = []

    class Model:
        def create_response(self, items, tools, **kwargs):
            nonlocal catalog
            seen.append(str(items))
            if len(seen) == 1:
                catalog = "second service now connected"
                return ModelResponse((), "", (FunctionCall("call", "list_directory", "{}"),))
            return answer()

    agent = Agent(Model(), create_default_registry(), tmp_path, mcp_catalog=lambda: catalog)
    try:
        agent.run("test")
        assert "first service" in seen[0]
        assert "second service now connected" in seen[1]
        assert "second service now connected" not in seen[0]
    finally:
        agent.close()


def test_tool_aliases_do_not_depend_on_connection_completion_order(tmp_path):
    path = config_file(tmp_path, {"a_b": {}, "a": {}})
    aliases = []
    for names in (("a_b", "a"), ("a", "a_b")):
        mcp = MCPConnections(path)
        for name in names:
            tool = "c" if name == "a_b" else "b_c"
            mcp.definitions[name] = [Tool(name=tool, input_schema={"type": "object"})]
            mcp._build_index()
        aliases.append({identity: mcp.index.get(identity)["alias"] for identity in ("a_b/c", "a/b_c")})
        mcp.close()
    assert aliases[0] == aliases[1]
    assert len(set(aliases[0].values())) == 2


def test_real_stdio_connection_is_reused_and_closed(tmp_path):
    script = tmp_path / "server.py"
    script.write_text('''from mcp.server.mcpserver import MCPServer
server = MCPServer("fixture")
@server.tool()
def lookup(value: str) -> str:
    return value
server.run()
''', encoding="utf-8")
    mcp = MCPConnections(config_file(tmp_path, {"stdio": {"command": sys.executable, "args": [str(script)]}}))
    try:
        mcp.start()
        assert mcp.ready.wait(15)
        assert mcp.status["stdio"]["state"] == "connected", mcp.snapshot()
        client = mcp.clients["stdio"]
        for text in ("first", "second"):
            result = mcp.call("stdio", "lookup", {"value": text})
            assert result.success and text in result.output
            assert mcp.clients["stdio"] is client
    finally:
        mcp.close()
    assert not mcp.thread.is_alive()
    assert mcp.clients == {}


def test_real_http_connection_registers_and_reports_a_later_disconnect(tmp_path):
    server = MCPServer("http fixture")

    @server.tool()
    def lookup(value: str) -> str:
        return value

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    http = uvicorn.Server(uvicorn.Config(server.streamable_http_app(), log_level="error", timeout_graceful_shutdown=1))
    thread = Thread(target=lambda: http.run(sockets=[listener]), daemon=True)
    thread.start()
    wait_until(lambda: http.started)
    mcp = MCPConnections(config_file(tmp_path, {"http": {"url": f"http://127.0.0.1:{port}/mcp", "tool_timeout_sec": 5}}))
    try:
        mcp.start()
        assert mcp.ready.wait(10)
        assert mcp.status["http"]["state"] == "connected", mcp.snapshot()
        client = mcp.clients["http"]
        assert mcp.call("http", "lookup", {"value": "first"}).success
        assert mcp.call("http", "lookup", {"value": "second"}).success
        assert mcp.clients["http"] is client
        http.should_exit = True
        thread.join(5)
        assert not thread.is_alive()
        listener.close()
        result = mcp.call("http", "lookup", {"value": "disconnected"})
        assert not result.success
        wait_until(lambda: mcp.status["http"]["state"] == "disconnected", timeout=8)
    finally:
        mcp.close()
        http.should_exit = True
        thread.join(5)
        listener.close()


def test_shared_factory_agents_do_not_own_the_manager(tmp_path, monkeypatch):
    from praxis_agent import factory
    from praxis_agent.service.manager import SessionManager
    from praxis_agent.service.store import ServiceStore
    from praxis_agent.sessions import SQLiteSessionStore

    monkeypatch.setenv("ADA_API_KEY", "test")
    monkeypatch.setattr(factory, "discover_skills", lambda workspace: [])
    monkeypatch.setattr(factory, "OpenAIModelClient", lambda *args, **kwargs: SimpleNamespace(close=lambda: None))
    mcp = ControlledConnections(config_file(tmp_path, {"shared": {"command": "shared"}}))
    mcp.start()
    assert mcp.ready.wait(5)
    store = ServiceStore(SQLiteSessionStore(tmp_path / "state.sqlite3"))
    manager = SessionManager(store, mcp=mcp)
    agents = []
    try:
        for _ in range(2):
            agents.append(manager.factory(tmp_path, session_store=store.sessions))
        assert agents[0].mcp_catalog.__self__ is agents[1].mcp_catalog.__self__ is mcp
        agents[0].close()
        assert not mcp.closed
        assert "shared" in agents[1].mcp_catalog()
        assert mcp.attempts == {"shared": 1}
    finally:
        for agent in agents:
            agent.close()
        manager.close()
    assert mcp.closed and not mcp.thread.is_alive()
