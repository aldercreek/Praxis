"""Opt-in real DeepSeek evaluation of native Python execution.

Example: .venv/Scripts/python.exe -u tests/manual/run_complex_code_tasks.py
         --case all --output build/complex-code-20261110 --jobs 2
Results, immutable inputs, prompts and streaming traces are retained per attempt.
This is an evaluation harness, not part of the normal pytest suite.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from complex_code_cases import CASES, dump, prepare, verify

REPO = Path(__file__).resolve().parents[2]


def alive(pid):
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def hashes(paths, base):
    return {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def run_case(args):
    sys.path.insert(0, str(REPO / "src"))
    os.environ["ADA_MODEL"] = args.model
    from praxis_agent.events import ToolCallPrepared, ToolFinished, ToolStarted
    from praxis_agent.execution import ExecutionStopped, Run, current_handle
    from praxis_agent.factory import create_default_agent
    from praxis_agent.mcp import MCPConnections
    from praxis_agent.tools import Tool, ToolResult

    destination = args.output.resolve() / args.case
    destination.mkdir(parents=True, exist_ok=False)
    workspace = destination / "workspace"
    workspace.mkdir()
    prompts, expected = prepare(args.case, workspace, destination)
    dump(destination / "fixture-hashes.json", hashes(workspace.rglob("*"), workspace))
    sources = list((REPO / "src/praxis_agent").rglob("*.py"))
    source_hashes = hashes(sources, REPO)
    dump(destination / "source-hashes.json", source_hashes)
    if args.prepare_only:
        print(json.dumps({"case": args.case, "prepared": True}), flush=True)
        return

    config = {}
    if args.case == "mcp_paging":
        config = {"servers": {"benchmark": {"command": sys.executable, "args": [str(destination / "mcp_server.py")],
                  "description": "Local paginated order and exchange rate benchmark. Read-only fixture data.",
                  "tools": {name: {"approval_mode": "approve"} for name in ("orders_page", "rate")}}}}
    dump(destination / "mcp.json", config)
    mcp = MCPConnections(destination / "mcp.json")
    mcp.start()
    assert mcp.ready.wait(40), "MCP startup timed out"
    if config:
        assert mcp.status["benchmark"]["state"] == "connected", mcp.snapshot()

    trace, model_calls, answers, failures, turns = [], [], [], [], []
    started = time.monotonic()
    trace_lock = threading.Lock()
    tool_started, tracked_pids = {}, set()
    state = dict(turn=0, scope=None, injected_cancel=False, host_cancelled=False, runtime=None)
    trace_file = (destination / "trace.jsonl").open("w", encoding="utf-8", buffering=1)

    def emit(item):
        item = {"elapsed": round(time.monotonic() - started, 4), "turn": state["turn"], **item}
        with trace_lock:
            trace.append(item)
            trace_file.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    def sample_workers():
        runtime = state["runtime"]
        if runtime:
            for worker in list(runtime._workers.values()):
                if worker.session and worker.session.worker_pid:
                    tracked_pids.add(worker.session.worker_pid)

    def observe(event):
        sample_workers()
        if isinstance(event, ToolCallPrepared):
            emit({"type": "request", "id": event.call_id, "tool": event.tool_name, "arguments": dict(event.arguments)})
        elif isinstance(event, ToolStarted):
            tool_started[event.call_id] = time.monotonic()
        elif isinstance(event, ToolFinished):
            emit({"type": "result", "id": event.call_id, "tool": event.tool_name, "success": event.result.success,
                  "output": event.result.output, "seconds": round(time.monotonic() - tool_started.get(event.call_id, time.monotonic()), 4)})
            print(json.dumps({"case": args.case, "turn": state["turn"], "tool": event.tool_name, "success": event.result.success}), flush=True)

    agent = create_default_agent(workspace, session_database_path=destination / "session.sqlite",
                                 event_sink=SimpleNamespace(emit=observe), bypass_approvals=True,
                                 enable_multi_agent=False, mcp=mcp)
    state["runtime"] = next(c.__self__ for c in agent.close_callbacks
                            if getattr(getattr(c, "__self__", None), "__class__", type(None)).__name__ == "CodeRuntime")
    underlying = agent.model_client

    class RecordingModel:
        def __getattr__(self, name):
            return getattr(underlying, name)

        def create_response(self, *positional, **keywords):
            if len(model_calls) >= args.max_requests:
                raise RuntimeError("BENCHMARK_REQUEST_LIMIT")
            call = {"turn": state["turn"], "request": len(model_calls) + 1}
            model_calls.append(call)
            before = time.monotonic()
            try:
                response = underlying.create_response(*positional, **keywords)
                call.update(model=response.model, input_tokens=response.usage.input_tokens if response.usage else None,
                            output_tokens=response.usage.output_tokens if response.usage else None,
                            tokens=response.usage.total_tokens if response.usage else None)
                return response
            except BaseException as error:
                call["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                call["seconds"] = round(time.monotonic() - before, 4)
                emit({"type": "model", **call})

    agent.model_client = RecordingModel()
    if args.case == "cancel_resume":
        def checkpoint(context, label):
            sample_workers()
            entered = threading.Event()
            unsubscribe = current_handle().on_cancel(entered.set)
            scope = state["scope"]
            def cancel():
                state["injected_cancel"] = True
                emit({"type": "fault_injection", "kind": "cancel", "label": label})
                scope.cancel()
            timer = threading.Timer(.15, cancel)
            timer.start()
            try:
                observed = entered.wait(15)
                state["host_cancelled"] = observed
                emit({"type": "host_cleanup", "cancelled": observed})
                return ToolResult(observed, "host cancellation cleanup finished")
            finally:
                timer.cancel()
                unsubscribe()
        agent.tool_registry.register(Tool("benchmark_checkpoint", "Wait at checkpoint until external cancellation (benchmark only).",
            {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"], "additionalProperties": False},
            checkpoint, programmatic=True))

    try:
        for index, prompt in enumerate(prompts):
            state["turn"] = index
            scope = Run("complex benchmark turn")
            state["scope"] = scope
            turn_started = time.monotonic()
            deadline_fired = threading.Event()
            def deadline():
                deadline_fired.set()
                scope.cancel()
            timer = threading.Timer(args.timeout, deadline)
            timer.start()
            status = "completed"
            try:
                answers.append(scope.execute(lambda: agent.run(prompt +
                    "\n请实际完成任务并自查，优先使用run_code。仅操作当前任务目录和当前会话依赖环境；"
                    "不要读取父目录、密钥或其他项目，不创建子agent，不需要向我确认；不创建Markdown报告。")))
            except Exception as error:
                intentional = (isinstance(error, ExecutionStopped) and args.case == "cancel_resume" and index == 0
                               and state["injected_cancel"] and not deadline_fired.is_set())
                status = "expected_cancel" if intentional else "failed"
                record = {"turn": index, "error": f"{type(error).__name__}: {error}", "expected": intentional, "deadline": deadline_fired.is_set()}
                failures.append(record)
                emit({"type": "turn_exception", **record})
                if not intentional:
                    break
            finally:
                timer.cancel()
                turns.append({"turn": index, "status": status, "seconds": round(time.monotonic() - turn_started, 3)})
                sample_workers()
    finally:
        agent.close()
        mcp.close()
        trace_file.close()

    code_events = [t for t in trace if t["type"] == "result" and t["tool"] == "run_code"]
    code_results = []
    for event in code_events:
        try:
            code_results.append(json.loads(event["output"]))
        except (ValueError, TypeError):
            code_results.append({"unparsed_output": event["output"]})
    evidence = dict(trace=trace, code_results=code_results, alive=alive,
                    host_cancelled=state["host_cancelled"], injected_cancel=state["injected_cancel"])
    try:
        checks, details = verify(args.case, workspace, expected, evidence)
    except Exception as error:
        checks, details = {"verification_completed": False}, {"verification_error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    checks["kernel_cleanup"] = all(not alive(pid) for pid in tracked_pids)
    checks["product_source_unchanged"] = source_hashes == hashes(sources, REPO)
    results = [t for t in trace if t["type"] == "result"]
    issues = []
    request_by_id = {t["id"]: t for t in trace if t["type"] == "request"}
    injected_mcp_seen = False
    for result in results:
        if result["success"]:
            continue
        output = result["output"]
        request = request_by_id.get(result["id"], {})
        # MCP masks server exception messages. Identify our injected EUR failure
        # by its recorded tool request instead of relying solely on error text.
        mcp_injected = (not injected_mcp_seen and args.case == "mcp_paging" and result["tool"].startswith("mcp_benchmark_rate")
                        and request.get("arguments", {}).get("currency") == "EUR")
        injected_mcp_seen = injected_mcp_seen or mcp_injected
        cancelled_code = False
        if result["tool"] == "run_code":
            try:
                cancelled_code = json.loads(output).get("error_kind") == "cancelled"
            except (ValueError, TypeError):
                pass
        injected = ("INJECTED_AFTER_MUTATION" in output or "INJECTED_TRANSIENT" in output or mcp_injected or
                    (args.case == "cancel_resume" and result["turn"] == 0 and state["injected_cancel"] and cancelled_code))
        issues.append({"kind": "expected_injection" if injected else "tool_error_needs_review", "tool": result["tool"],
                       "call_id": result["id"], "turn": result["turn"], "evidence": output})
    issues.extend({"kind": "acceptance_failure", "check": k, "details": details.get(k)} for k, passed in checks.items() if not passed)
    if "activation_query_overhead" in details:
        issues.append({"kind": "efficiency_warning", "details": details["activation_query_overhead"]})
    report = dict(case=args.case, requested_model=args.model, session_id=agent.session_id,
                  completed=all(checks.values()) and not any(not f["expected"] for f in failures),
                  seconds=round(time.monotonic() - started, 3), checks=checks, verification_details=details,
                  run_code_calls=len(code_events), run_code_errors=sum(not c["success"] for c in code_events),
                  unexpected_tool_errors=sum(i["kind"] == "tool_error_needs_review" for i in issues),
                  all_tool_calls=len(results), command_calls=sum(t["tool"] == "run_command" for t in results),
                  kernel_pids=sorted(tracked_pids), model_calls=model_calls, turns=turns, failures=failures,
                  issues=issues, answers=answers, code_results=code_results)
    dump(destination / "result.json", report)
    dump(destination / "issues.json", issues)
    print(json.dumps({k: report[k] for k in ("case", "completed", "seconds", "checks", "run_code_calls", "run_code_errors", "unexpected_tool_errors")}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[*CASES, "all"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument("--max-requests", type=int, default=40)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.case != "all":
        run_case(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    def run_child(case):
        with (args.output / f"{case}.console.log").open("w", encoding="utf-8") as log:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--case", case, "--output", str(args.output.resolve()),
                       "--model", args.model, "--timeout", str(args.timeout), "--max-requests", str(args.max_requests)]
            if args.prepare_only:
                command.append("--prepare-only")
            result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        report_path = args.output / case / "result.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"case": case, "process_exit": result.returncode}
        print(json.dumps({k: report.get(k) for k in ("case", "completed", "seconds", "run_code_errors", "unexpected_tool_errors", "process_exit")}), flush=True)
        return report
    reports = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_child, case) for case in CASES]
        for future in as_completed(futures):
            reports.append(future.result())
            dump(args.output / "summary.json", sorted(reports, key=lambda r: CASES.index(r["case"])))


if __name__ == "__main__":
    main()
