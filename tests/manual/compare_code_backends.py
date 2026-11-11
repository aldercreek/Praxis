"""Opt-in, real-model task benchmark; never run by the ordinary pytest suite.

Run this same script with --source pointing at an archived baseline or this
checkout. Artifacts and JSON traces are kept under --output. Requires a Praxis configuration file and its configured credentials.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from types import SimpleNamespace
import zipfile


def prepare(case, root):
    if case == "csv_state":
        (root / "orders.csv").write_text("category,amount\nbooks,120\nfood,30\nbooks,80\nfood,70\n", encoding="utf-8")
        return [
            "处理当前目录 orders.csv：优先使用 run_code 和 call_tool 读取并按 category 汇总 amount。"
            "把汇总字典保存在 Python 变量 totals 中，并用 write_file 工具写 summary.json（顶层对象为汇总字典）。"
            "说明结果，保留变量供下一轮使用。",
            "继续上一轮：直接使用保留的 totals 变量，把 books 加 25，新增 travel=50，写 updated.json。"
            "不要重新读取输入或者手写之前的汇总值。",
        ]
    if case == "project_library":
        (root / "invoice_math.py").write_text(
            "from decimal import Decimal, ROUND_HALF_UP\n"
            "def net(price, count, discount):\n"
            "    return str((Decimal(price)*count*(1-Decimal(discount))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))\n",
            encoding="utf-8")
        (root / "invoice.json").write_text(json.dumps({"price": "19.95", "count": 7, "discount": "0.12"}), encoding="utf-8")
        return ["检查 invoice.json，使用已安装的 jsonschema 校验 price 和 discount 为字符串、count 为正整数，"
                "并实际 import 当前项目 invoice_math.py 的 net 函数来计算金额，不能复制实现。"
                "优先 run_code，必要时可使用其他已有工具。把 {net: 金额字符串, validated: true} 写入 invoice-result.json。"]
    if case == "sqlite_zip":
        db = sqlite3.connect(root / "sales.sqlite")
        db.execute("create table sales(region text, amount integer)")
        db.executemany("insert into sales values (?, ?)", [("east", 10), ("west", 7), ("east", 15)])
        db.commit()
        db.close()
        with zipfile.ZipFile(root / "input.zip", "w") as archive:
            archive.write(root / "sales.sqlite", "sales.sqlite")
        (root / "sales.sqlite").unlink()
        return ["解压 input.zip，实际读取其中的 SQLite 数据库 sales.sqlite 的 sales 表，按 region 汇总 amount。"
                "优先使用 run_code，必要时可使用其他已有工具。生成 database-result.json，顶层对象为地区到金额的映射。"]
    (root / "orders.json").write_text(json.dumps([
        {"currency": "CNY", "amount": 100}, {"currency": "USD", "amount": 20}, {"currency": "EUR", "amount": 30},
    ]), encoding="utf-8")
    return ["读取 orders.json，使用已连接的 benchmark MCP 服务器的 rate 工具获取每个币种转 CNY 的汇率。"
            "先发现并激活工具，然后在 run_code 内通过 call_parallel_tools 并行调用这三个汇率查询。"
            "根据真实返回值计算总 CNY 金额，用 call_tool 的 write_file 写 fx-result.json，格式 {total_cny: 数值}。"]


def verify(case, root):
    expected = {
        "csv_state": {"summary.json": {"books": 200, "food": 100},
                      "updated.json": {"books": 225, "food": 100, "travel": 50}},
        "project_library": {"invoice-result.json": {"net": "122.89", "validated": True}},
        "sqlite_zip": {"database-result.json": {"east": 25, "west": 7}},
        "mcp_parallel": {"fx-result.json": {"total_cny": 480}},
    }[case]
    return {name: path.exists() and json.loads(path.read_text(encoding="utf-8-sig")) == value
            for name, value in expected.items() for path in [root / name]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--case", choices=["csv_state", "project_library", "sqlite_zip", "mcp_parallel"], required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve() / "src"))

    from praxis_agent.events import ToolCallPrepared, ToolFinished
    from praxis_agent.execution import Run
    from praxis_agent.factory import create_default_agent
    from praxis_agent.mcp import MCPConnections

    destination = args.output.resolve() / args.backend / args.case
    destination.mkdir(parents=True, exist_ok=False)
    workspace = destination / "workspace"
    workspace.mkdir()
    prompts = prepare(args.case, workspace)
    config_path = destination / "mcp.json"
    server_path = destination / "mcp_server.py"
    server_path.write_text(
        "from mcp.server import MCPServer\n"
        "server = MCPServer('benchmark')\n"
        "@server.tool(structured_output=True)\n"
        "def rate(currency: str) -> dict[str, str | int]:\n"
        "    return {'currency': currency, 'cny_rate': {'CNY': 1, 'USD': 7, 'EUR': 8}[currency]}\n"
        "server.run()\n", encoding="utf-8")
    config_path.write_text(json.dumps({"servers": {"benchmark": {
        "command": sys.executable, "args": [str(server_path)],
        "description": "Local benchmark currency conversion rates; no external network.",
        "tools": {"rate": {"approval_mode": "approve"}},
    }}} if args.case == "mcp_parallel" else {}), encoding="utf-8")
    mcp = MCPConnections(config_path)
    mcp.start()
    assert mcp.ready.wait(40), "MCP startup timed out"
    if args.case == "mcp_parallel":
        assert mcp.status["benchmark"]["state"] == "connected", mcp.snapshot()
    trace, model_calls, answers, failures = [], [], [], []

    def observe(event):
        if isinstance(event, ToolCallPrepared):
            trace.append({"type": "request", "id": event.call_id, "tool": event.tool_name, "arguments": event.arguments})
        elif isinstance(event, ToolFinished):
            trace.append({"type": "result", "id": event.call_id, "tool": event.tool_name,
                          "success": event.result.success, "output": event.result.output})
            print(json.dumps({"tool": event.tool_name, "success": event.result.success}), flush=True)

    agent = create_default_agent(workspace, model=args.model, session_database_path=destination / "session.sqlite",
                                 event_sink=SimpleNamespace(emit=observe), bypass_approvals=True, enable_multi_agent=False, mcp=mcp)
    underlying = agent.model_client

    class RecordingModel:
        def __getattr__(self, name):
            return getattr(underlying, name)

        def create_response(self, *positional, **keywords):
            start = time.monotonic()
            response = underlying.create_response(*positional, **keywords)
            model_calls.append({"seconds": time.monotonic() - start, "model": response.model,
                                "tokens": response.usage.total_tokens if response.usage else None})
            return response

    agent.model_client = RecordingModel()
    started = time.monotonic()
    try:
        for prompt in prompts:
            scope = Run("benchmark turn")
            timer = threading.Timer(180, scope.cancel)
            timer.start()
            try:
                answers.append(scope.execute(lambda: agent.run(
                    prompt + " 只操作当前任务目录；不要创建子agent，不要读取密钥或操作其他目录。")))
            except Exception as error:
                failures.append(f"{type(error).__name__}: {error}")
                break
            finally:
                timer.cancel()
    finally:
        agent.close()
        mcp.close()
    checks = verify(args.case, workspace)
    results = [item for item in trace if item["type"] == "result"]
    codes = [item for item in results if item["tool"] == "run_code"]
    report = {
        "backend": args.backend, "case": args.case, "requested_model": args.model,
        "seconds": round(time.monotonic() - started, 2), "checks": checks,
        "completed": all(checks.values()) and not failures, "failures": failures,
        "run_code_calls": len(codes), "run_code_errors": sum(not item["success"] for item in codes),
        "all_tool_calls": len(results), "all_tool_errors": sum(not item["success"] for item in results),
        "command_calls": sum(item["tool"] == "run_command" for item in results),
        "model_calls": model_calls, "answers": answers, "trace": trace,
    }
    (destination / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"trace", "answers", "model_calls"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
