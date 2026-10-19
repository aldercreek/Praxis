"""Explicit parallel calls through Monty and the real tool executor."""
import asyncio
import json
import threading
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import pytest
from mcp import Client
from mcp.server import MCPServer

from praxis_agent.events import ToolFinished, ToolStarted
from praxis_agent.execution import Run, current_handle, current_run
from praxis_agent.mcp import MCPConnections
from praxis_agent.model import FunctionCall
from praxis_agent.planning import RunState
from praxis_agent.tool_execution import ToolCallExecutor
from praxis_agent.tools import Tool, ToolContext, ToolResult, create_default_registry
from praxis_agent.tools.code import CodeConfig, CodeRuntime


@pytest.fixture
def setup(tmp_path):
    runtime = CodeRuntime(output_dir=tmp_path / "logs")
    registry = create_default_registry()
    registry.register(runtime.tool())
    events = []
    executor = ToolCallExecutor(registry, ToolContext(tmp_path, RunState("parallel test")),
                                publish_event=events.append)

    def execute(code):
        return executor.execute("turn", FunctionCall("code", "run_code", json.dumps({
            "code": code,
        })))

    def run(code):
        result = execute(code)
        assert result.success, result.output
        return json.loads(result.output)

    try:
        yield SimpleNamespace(runtime=runtime, registry=registry, executor=executor,
                              events=events, execute=execute, run=run)
    finally:
        runtime.close()


def register(setup, name, handler, **options):
    setup.registry.register(Tool(name, name, {
        "type": "object", "properties": {"index": {"type": "integer"}},
        "required": ["index"], "additionalProperties": False,
    }, handler, **{"programmatic": True, **options}))


@pytest.mark.parametrize("workers", [2, 16])
def test_mixed_dynamic_batch_runs_concurrently_with_ordered_results(setup, workers):
    setup.runtime.config = replace(setup.runtime.config, max_parallel_tools=workers)
    barrier = threading.Barrier(workers)
    lock = threading.Lock()
    active = peak = 0
    completed = []
    last_finished = threading.Event()

    def query(context, index):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            assert current_run().name == "run_code"
            if index < workers:
                barrier.wait(timeout=5)
            # Force completion order to differ from input order.
            if index == 0:
                assert last_finished.wait(5)
            with lock:
                completed.append(index)
            if index == workers + 2:
                last_finished.set()
            return ToolResult(True, "original", structured_output={"index": index})
        finally:
            with lock:
                active -= 1

    register(setup, "query_a", query)
    register(setup, "query_b", query)
    payload = setup.run(f'''
requests = []
for i in range({workers + 3}):
    requests.append({{"name": "query_a" if i % 2 else "query_b", "arguments": {{"index": i}}}})
call_parallel_tools(requests)
''')
    assert peak == workers
    assert completed[0] != 0
    assert payload["result"] == [
        {"success": True, "output": {"index": i}} for i in range(workers + 3)
    ]
    finished = [e for e in setup.events if isinstance(e, ToolFinished) and e.tool_name.startswith("query_")]
    assert len(finished) == workers + 3
    assert len({e.call_id for e in finished}) == workers + 3
    assert len(payload["calls"]) == workers + 3
    assert setup.run("call_parallel_tools([])")["result"] == []


def test_partial_errors_keep_permissions_validation_and_approval(setup):
    invoked, approvals = [], []

    def query(context, index):
        invoked.append(index)
        if index == 3:
            raise ValueError("bad record")
        return ToolResult(True, "text result")

    def approve(name, args):
        approvals.append(args["index"])
        return args["index"] != 1

    setup.executor.approve_tool_call = approve
    register(setup, "query", query, requires_approval=True)
    register(setup, "forbidden", query, programmatic=False)
    requests = [
        {"name": "query", "arguments": {"index": 0}},
        {"name": "query", "arguments": {"index": 1}},
        {"name": "query", "arguments": {"index": "wrong type"}},
        {"name": "query", "arguments": {"index": 3}},
        {"name": "forbidden", "arguments": {"index": 4}},
        {"name": "run_code", "arguments": {}},
        {"name": "missing", "arguments": {}},
    ]
    results = setup.run(f"call_parallel_tools({requests!r})")["result"]
    assert results[0] == {"success": True, "output": "text result"}
    assert all(not item["success"] for item in results[1:])
    assert "declined" in results[1]["error"]
    assert "bad record" in results[3]["error"]
    assert all("unavailable" in item["error"] for item in results[4:])
    assert sorted(invoked) == [0, 3]
    assert approvals == [0, 1, 3]


@pytest.mark.parametrize("bad_value", ["{1, 2}", "b'bytes'", "{(1, 2): 'value'}", "{b'key': 'value'}"])
@pytest.mark.parametrize("valid_neighbors", [False, True])
def test_unencodable_arguments_fail_only_their_item(setup, bad_value, valid_neighbors):
    invoked, approved = [], []

    def query(context, index):
        invoked.append(index)
        return ToolResult(True, "original", structured_output={"index": index})

    setup.executor.approve_tool_call = lambda name, args: approved.append(args["index"]) or True
    register(setup, "query", query, requires_approval=True)
    items = [f'{{"name": "query", "arguments": {{"index": {bad_value}}}}}']
    if valid_neighbors:
        items.insert(0, '{"name": "query", "arguments": {"index": 0}}')
        items.append('{"name": "query", "arguments": {"index": 2}}')
    payload = setup.run("call_parallel_tools([" + ",".join(items) + "])")
    results = payload["result"]
    failed = results[1 if valid_neighbors else 0]
    assert failed["success"] is False
    assert "cannot be encoded as JSON" in failed["error"]
    if valid_neighbors:
        assert results[0] == {"success": True, "output": {"index": 0}}
        assert results[2] == {"success": True, "output": {"index": 2}}
    assert sorted(invoked) == ([0, 2] if valid_neighbors else [])
    assert approved == ([0, 2] if valid_neighbors else [])
    assert [r["success"] for r in payload["calls"]] == [r["success"] for r in results]
    finished = [e for e in setup.events if isinstance(e, ToolFinished) and e.tool_name == "query"]
    started = [e for e in setup.events if isinstance(e, ToolStarted) and e.tool_name == "query"]
    assert len(finished) == len(items)
    assert len({e.call_id for e in finished}) == len(items)
    assert len(started) == len(invoked)


def test_malformed_batch_does_not_start_earlier_valid_items(setup):
    invoked = []
    register(setup, "query", lambda context, index: invoked.append(index) or "done")
    result = setup.execute('call_parallel_tools([{ "name": "query", "arguments": {"index": 0}}, {"name": "query"}])')
    assert not result.success
    assert invoked == []


def test_parallel_cancellation_drains_scheduler_and_does_not_dispatch_queue(setup):
    setup.runtime.config = replace(setup.runtime.config, max_parallel_tools=2)
    assert setup.run("saved: int = 1")["worker_created"]
    entered = threading.Barrier(3)
    release = threading.Event()
    cancelled = [threading.Event(), threading.Event()]
    invoked = []

    def waiting(context, index):
        invoked.append(index)
        handle = current_handle()
        unsubscribe = handle.on_cancel(cancelled[index].set)
        try:
            entered.wait(5)
            assert cancelled[index].wait(5)
            assert release.wait(5)
            handle.check()
        finally:
            unsubscribe()

    register(setup, "waiting", waiting)
    root = Run("turn")
    done = Future()

    def worker():
        try:
            done.set_result(root.execute(lambda: setup.execute('''
saved = 2
call_parallel_tools([{"name": "waiting", "arguments": {"index": i}} for i in range(5)])
''')))
        except BaseException as error:
            done.set_exception(error)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        try:
            entered.wait(5)
        except threading.BrokenBarrierError:
            if done.done():
                pytest.fail(done.result().output)
            raise
        root.handles[0].cancel()
        assert all(event.wait(2) for event in cancelled)
        assert not done.done()
        release.set()
        result = done.result(timeout=5)
        payload = json.loads(result.output)
        assert not result.success
        assert payload["error_kind"] == "cancelled"
        assert sorted(invoked) == [0, 1]
        assert len(payload["calls"]) == 5
        finished = [e for e in setup.events if isinstance(e, ToolFinished) and e.tool_name == "waiting"]
        assert len(finished) == 5
        assert payload["state_lost"] is True
        assert setup.run("'saved' in locals()")["result"] is False
    finally:
        release.set()
        root.cancel()
        thread.join(10)
    assert not thread.is_alive()


def test_activated_mcp_tools_overlap_on_same_connection(setup, tmp_path):
    connections = MCPConnections(tmp_path / "mcp.json")
    connections.servers = {"test": {"tools": {"lookup": {"approval_mode": "approve"}}}}
    connections.status = {"test": {"state": "ready"}}
    connections.loop = setup.runtime._ensure_loop()
    entered = 0
    both = asyncio.Event()
    server = MCPServer("parallel test")

    @server.tool(structured_output=True)
    async def lookup(index: int) -> dict[str, int]:
        nonlocal entered
        entered += 1
        if entered == 2:
            both.set()
        await asyncio.wait_for(both.wait(), timeout=3)
        return {"index": index}

    ready = Future()
    stop = asyncio.Event()

    async def serve():
        try:
            # Exercise the real JSON-RPC request routing, not only a mock client.
            async with Client(server, mode="legacy") as client:
                connections.clients = {"test": client}
                connections.locks = {"test": asyncio.Lock()}
                connections.definitions = {"test": (await client.list_tools()).tools}
                ready.set_result(None)
                await stop.wait()
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
            raise

    serving = asyncio.run_coroutine_threadsafe(serve(), connections.loop)
    release = lambda: None
    try:
        ready.result(timeout=5)
        connections._build_index()
        release = connections.register(setup.registry)
        activated = setup.executor.execute("turn", FunctionCall("activate", "activate_tools", json.dumps({
            "tools": [{"tool_id": "test/lookup"}],
        })))
        assert activated.success
        results = setup.run('call_parallel_tools([{"name": "mcp_test_lookup", "arguments": {"index": i}} for i in range(2)])')["result"]
        assert results == [{"success": True, "output": {"index": i}} for i in range(2)]
        assert entered == 2
    finally:
        release()
        connections.loop.call_soon_threadsafe(stop.set)
        serving.result(timeout=5)


def test_parallel_config(monkeypatch):
    monkeypatch.delenv("PRAXIS_CODE_MAX_PARALLEL_TOOLS", raising=False)
    assert CodeConfig.from_env().max_parallel_tools == 16
    monkeypatch.setenv("PRAXIS_CODE_MAX_PARALLEL_TOOLS", "32")
    assert CodeConfig.from_env().max_parallel_tools == 32
    for value in ("0", "-1", "2.5", "invalid"):
        monkeypatch.setenv("PRAXIS_CODE_MAX_PARALLEL_TOOLS", value)
        with pytest.raises(ValueError):
            CodeConfig.from_env()
