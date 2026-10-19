"""Compare file-tool surfaces using the real Praxis loop and isolated fixtures.

Run with the repository virtualenv. No production modules or user sessions are
modified. Trial prompts, tool calls, provider usage and independent grades are
saved beside the fixtures; credentials and model reasoning are never recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock, Timer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from praxis_agent.agent import Agent
from praxis_agent.config import ModelConfig
from praxis_agent.context.capacity import HeuristicTokenEstimator
from praxis_agent.context.manager import ContextManager
from praxis_agent.context.model_catalog import ModelLimitOverrides
from praxis_agent.events import ToolCallPrepared, ToolFinished, ToolStarted
from praxis_agent.execution import ExecutionControl, Run
from praxis_agent.model import OpenAIModelClient
from praxis_agent.prompts import build_instructions
from praxis_agent.runtime import collect_runtime_context
from praxis_agent.skills import create_skill_tool, discover_skills
from praxis_agent.tools.code import CodeRuntime
from praxis_agent.tools.processes import CommandProcess
from praxis_agent.tools.registry import create_default_registry


FILE_TOOLS = {"read_file", "edit_file", "write_file", "list_files", "find_files", "search_text", "run_command"}
VARIANTS = {
    "full": FILE_TOOLS,
    "command_read_edit": {"run_command", "read_file", "edit_file", "write_file"},
    "command_edit": {"run_command", "edit_file", "write_file"},
    "command_only": {"run_command"},
}
CASES = ("search_noise", "cross_directory", "long_read", "precise_edit", "multifile_fix", "aggregate")


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def put(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


def snapshot(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and not any(part in {".git", "__pycache__", ".pytest_cache"} for part in p.relative_to(root).parts)
    }


@dataclass
class Fixture:
    root: Path
    workspace: Path
    task: str
    expected: dict
    allowed_changes: tuple[str, ...] = ()


def fixture(directory: Path, case: str, repeat: int) -> Fixture:
    root = directory / "sandbox"
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    put(root, ".gitignore", ".venv/\n.cache/\nbuild/\n__pycache__/\n*.pyc\n")
    seed = 731 + repeat
    rng = random.Random(seed)
    if case == "search_noise":
        target = 1700 + repeat
        put(root, "src/订单 服务/dispatch.py", f"RETRY_BACKOFF_MS = {target}\n\ndef dispatch(order):\n    return order\n")
        put(root, "src/订单 服务/consumer.py", "from .dispatch import RETRY_BACKOFF_MS\n\ndef backoff():\n    return RETRY_BACKOFF_MS\n")
        put(root, "src/health.py", "def healthy():\n    return True\n")
        for i in range(100):
            put(root, f".venv/cache_{i}.py", "RETRY_BACKOFF_MS = 999999\n" * 8)
            put(root, f"build/generated_{i}.py", "RETRY_BACKOFF_MS = -1\n" * 8)
        put(root, "notes/example.txt", "Example only: RETRY_BACKOFF_MS = 1\n")
        return Fixture(root, root,
            "只读分析：找出业务源码中 RETRY_BACKOFF_MS 的定义和引用，忽略依赖、构建产物和示例。"
            "确认 PAYMENT_GATEWAY_V9 在业务源码中不存在。最后仅返回 JSON："
            '{"definition_path":"相对路径","value":数字,"reference_paths":["相对路径"],"absent":true}。',
            {"definition_path": "src/订单 服务/dispatch.py", "value": target,
             "reference_paths": ["src/订单 服务/consumer.py"], "absent": True})
    if case == "cross_directory":
        project = root / "项目 with spaces"
        startup = root / "launcher"
        startup.mkdir()
        marker = f"区域-{repeat}-XA19"
        put(project, "settings/runtime.json", json.dumps({"region": marker, "timeout_ms": 3200 + repeat}, ensure_ascii=False))
        put(startup, "runtime.json", '{"region":"WRONG","timeout_ms":0}')
        return Fixture(root, startup,
            f"只读检查项目 {project}，当前启动目录不是项目目录。找到 settings/runtime.json，"
            '返回该项目的配置，最终仅输出 JSON：{"region":"...","timeout_ms":数字}。',
            {"region": marker, "timeout_ms": 3200 + repeat})
    if case == "long_read":
        lines = [f"record {i:05d}: ordinary documentation text; not the requested record.\n" for i in range(1, 9001)]
        code = f"发布校验-{repeat}-D8F4"
        lines[7499] = f"RELEASE_RECORD: code={code}; owner=配置团队; limit={6100 + repeat}\n"
        lines[7500] = "literal=$HOME; quote='keep'; windows=D:\\work space\\报告\\记录.txt\n"
        put(root, "资料 文档/large.txt", "".join(lines))
        return Fixture(root, root,
            "只读检查 资料 文档/large.txt 第 7500 和 7501 行（行号从 1 开始）。"
            "只返回这两行中的字段，quote 值去掉包裹它的单引号。最终仅输出 JSON："
            '{"code":"...","owner":"...","limit":数字,"literal":"...","quote":"...","windows":"..."}。',
            {"code": code, "owner": "配置团队", "limit": 6100 + repeat,
             "literal": "$HOME", "quote": "keep", "windows": "D:\\work space\\报告\\记录.txt"})
    if case == "precise_edit":
        original = (
            '# -*- coding: utf-8 -*-\n# 保留此注释和所有其他文本\n'
            'def regular_client():\n    timeout = 30\n    label = "普通"\n    return timeout, label\n\n'
            'def audit_client():\n    timeout = 30\n    label = "审计"\n    return timeout, label\n'
        )
        target = 47 + repeat
        note = "审计说明：保留 $HOME、'单引号'、\"双引号\"、反引号 ` 和路径 C:\\订单\\新文件。\n"
        put(root, "src/连接 设置.py", original)
        return Fixture(root, root,
            f"修改 src/连接 设置.py：仅把 audit_client 的 timeout 从 30 改成 {target}，regular_client 保持 30。"
            "除目标数字外原文件所有文本、空白及换行都必须保持不变。另新建 docs/审计说明.txt，内容必须逐字为：\n"
            + note + "文件末尾保留一个换行。完成后检查实际内容。",
            {"src/连接 设置.py": original.replace('def audit_client():\n    timeout = 30', f'def audit_client():\n    timeout = {target}'),
             "docs/审计说明.txt": note},
            ("src/连接 设置.py", "docs/审计说明.txt"))
    if case == "multifile_fix":
        put(root, "pricing.py", "# 金额单位：分\ndef total_price(unit_price, quantity, shipping):\n    return unit_price + quantity + shipping\n")
        put(root, "invoice.py", "from pricing import total_price\n\ndef invoice(items, shipping):\n    return sum(total_price(price, qty, shipping) for price, qty in items)\n")
        put(root, "tests/test_public.py", "import unittest\nfrom pricing import total_price\nfrom invoice import invoice\n\nclass Check(unittest.TestCase):\n    def test_price(self):\n        self.assertEqual(total_price(120, 3, 20), 380)\n    def test_invoice(self):\n        self.assertEqual(invoice([(100, 2), (50, 3)], 30), 380)\n")
        put(root, "untouched.txt", "此文件不得修改\n")
        return Fixture(root, root,
            "修复 pricing.py 和 invoice.py：total_price 应计算单价乘数量再加运费；"
            "invoice 应先汇总所有商品金额，再只加一次订单运费；没有商品时总额为 0。"
            "金额都是整数分。只允许修改这两个源文件，不要改测试及其他文件。运行 tests 中的 unittest 测试并检查边界情况。",
            {}, ("pricing.py", "invoice.py"))
    rows = []
    for month in range(1, 13):
        values = [{"id": month * 1000 + i, "region": rng.choice(["华东", "华南", "华北"]),
                   "status": rng.choice(["paid", "cancelled", "pending"]), "amount_cents": rng.randint(20, 9000)} for i in range(140)]
        rows.extend(values)
        put(root, f"data/月报 {month:02d}.json", json.dumps(values, ensure_ascii=False))
    selected = [r for r in rows if r["region"] == "华东" and r["status"] == "paid"]
    return Fixture(root, root,
        "只读汇总 data 目录中 12 个月报 JSON 文件：筛选 region=华东 且 status=paid 的订单，"
        "计算订单数、amount_cents 总和、最大 amount_cents。不要把完整原始数据逐条输出。"
        '最终仅输出 JSON：{"count":数字,"total_cents":数字,"max_cents":数字}。',
        {"count": len(selected), "total_cents": sum(r["amount_cents"] for r in selected),
         "max_cents": max(r["amount_cents"] for r in selected)})


def extract_json(text: str):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[index:])
                if isinstance(value, dict):
                    return value
            except ValueError:
                pass
    return None


def grade(f: Fixture, case: str, answer: str, before: dict) -> dict:
    after = snapshot(f.root)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    unexpected = [p for p in changed if p not in f.allowed_changes]
    checks = {"only_allowed_files_changed": not unexpected}
    detail = {"changed_files": changed, "unexpected_changes": unexpected}
    if case == "precise_edit":
        for name, text in f.expected.items():
            path = f.root / name
            checks[name] = path.exists() and path.read_bytes() == text.encode("utf-8")
    elif case == "multifile_fix":
        script = (
            "from pricing import total_price\nfrom invoice import invoice\n"
            "assert total_price(120,3,20)==380\nassert total_price(100,0,10)==10\n"
            "assert invoice([(100,2),(50,3)],30)==380\nassert invoice([],30)==0\n"
            "assert invoice([(37,4)],13)==161\nassert invoice([(0,8),(80,0)],11)==11\n"
        )
        completed = subprocess.run([sys.executable, "-X", "utf8", "-c", script], cwd=f.root,
                                   capture_output=True, text=True, encoding="utf-8", timeout=15)
        checks["independent_behavior_checks"] = completed.returncode == 0
        detail["validation_output"] = completed.stderr[-2000:]
    else:
        actual = extract_json(answer)
        if isinstance(actual, dict):
            if isinstance(actual.get("definition_path"), str):
                actual["definition_path"] = actual["definition_path"].replace("\\", "/").removeprefix("./")
            if isinstance(actual.get("reference_paths"), list):
                actual["reference_paths"] = sorted({p.replace("\\", "/").removeprefix("./")
                                                     for p in actual["reference_paths"] if isinstance(p, str)})
        checks["answer_matches_oracle"] = actual == f.expected
        detail.update(actual=actual, expected=f.expected)
    return {"passed": all(checks.values()), "checks": checks, **detail}


class Measurement:
    def __init__(self, directory):
        self.lock = Lock()
        self.log = directory / "events.jsonl"
        self.calls = {}
        self.responses = []
        self.approval_count = 0

    def record(self, kind, data):
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": kind, "data": data}, ensure_ascii=False) + "\n")

    def record_request(self, response):
        with self.lock:
            self.responses.append(response)
            self.record("model_response", response)

    def emit(self, event):
        with self.lock:
            if isinstance(event, ToolCallPrepared):
                self.calls[event.call_id] = {"id": event.call_id, "name": event.tool_name,
                    "arguments": dict(event.arguments), "requires_approval": event.requires_approval}
                self.approval_count += int(event.requires_approval)
                self.record("tool_prepared", self.calls[event.call_id])
            elif isinstance(event, ToolStarted):
                self.calls.setdefault(event.call_id, {"name": event.tool_name})["started"] = event.occurred_at
            elif isinstance(event, ToolFinished):
                item = self.calls.setdefault(event.call_id, {"name": event.tool_name})
                item.update(success=event.result.success, output=event.result.output,
                            output_chars=len(event.result.output), has_diff=event.result.file_change is not None)
                if "started" in item:
                    item["elapsed_seconds"] = event.occurred_at - item.pop("started")
                if event.tool_name == "run_command":
                    data = event.result.structured_output
                    item["command_result"] = data
                    item["exit_code"] = data["exit_code"] if data is not None else None
                    argv = item.get("arguments", {}).get("argv", [])
                    item["rg_no_match"] = bool(data is not None and data["status"] == "exited" and
                                                data["exit_code"] == 1 and argv and Path(argv[0]).stem.lower() == "rg"
                                                and not data["stderr"])
                self.record("tool_finished", item)


class MeasuredClient(OpenAIModelClient):
    def __init__(self, *args, measurement, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurement = measurement
        self.request_count = 0

    def create_response(self, *args, **kwargs):
        self.request_count += 1
        if self.request_count > 16:
            raise RuntimeError("Benchmark model-request limit reached")
        kwargs["max_output_tokens"] = 8192
        return super().create_response(*args, **kwargs)

    async def _request(self, *args, **kwargs):
        start = time.monotonic()
        response = await super()._request(*args, **kwargs)
        self.measurement.record_request({"model": response.model, "elapsed_seconds": time.monotonic() - start,
                                         "usage": response.usage.model_dump() if response.usage else None})
        return response


def trial(args) -> None:
    directory = args.output / f"r{args.repeat}-{args.case}-{args.variant}"
    directory.mkdir(parents=True, exist_ok=True)
    f = fixture(directory, args.case, args.repeat)
    before = snapshot(f.root)
    config = ModelConfig.from_env(model=args.model)
    # The alias has no provider capacity metadata: 1M is an explicit experiment assumption.
    config = ModelConfig(api_key=config.api_key, model=config.model, base_url=config.base_url,
                         model_limit_overrides=ModelLimitOverrides(context_window_tokens=1_000_000))
    control = ExecutionControl()
    measurement = Measurement(directory)
    registry = create_default_registry(config.context_policy, check_execution=control.check)
    for name in FILE_TOOLS - VARIANTS[args.variant]:
        registry.unregister(name)
    code = CodeRuntime()
    registry.register(code.tool("parent"))
    skills = discover_skills(f.workspace, user_root=directory / "no-user-skills")
    registry.register(create_skill_tool(skills, max_chars=config.context_policy.max_tool_output_chars))
    instructions = build_instructions(collect_runtime_context(f.workspace), skills=skills)
    # Use the same neutral guidance in every arm, without naming unavailable tools.
    guidance = (
        "<tools>\n根据实际提供的工具及其参数说明完成任务。文件读取、搜索、编辑可以通过可用专用工具或系统命令完成。\n"
        "run_code 使用受限 Python，通过 call_tool 调用已允许的工具；不提供主机文件系统、子进程及第三方库。\n"
        "完整 Python 脚本可通过 run_command 运行。\n</tools>"
    )
    instructions = re.sub(r"<tools>.*?</tools>", lambda _: guidance, instructions, count=1, flags=re.S)
    instructions += (f"\n<experiment_environment>\n所有任务文件都在 {f.root}；只允许在该目录内操作。"
                     "这是独立测试副本，可直接执行任务所需命令。不要访问网络、读取环境变量或修改测试框架。"
                     f"系统有 rg、PowerShell、pwsh 和 Python；Python 解释器为 {sys.executable}。"
                     "没有 bash。run_command 的 argv 是参数数组，不隐式解释 shell 语法。"
                     "目标是正确完成任务；不要为追求工具调用次数而省略必要检查。\n</experiment_environment>")
    client = MeasuredClient(config, instructions=instructions, execution_control=control, measurement=measurement)
    manager = ContextManager(model_limits=config.model_limits, policy=config.context_policy, instructions=instructions)
    agent = Agent(model_client=client, tool_registry=registry, workspace=f.workspace, event_sink=measurement,
                  context_manager=manager, execution_control=control, approve_tool_call=lambda *_: True,
                  close_callbacks=(code.close, client.close))
    agent.code_session_context = lambda: code.context_item("parent")
    scope = Run("benchmark")
    timer = Timer(240, scope.cancel)
    timer.daemon = True
    answer, error = "", None
    start = time.monotonic()
    timer.start()
    try:
        answer = scope.execute(lambda: agent.run(f.task))
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:1500]}
    finally:
        timer.cancel()
        agent.close()
    elapsed = time.monotonic() - start
    grading = grade(f, args.case, answer, before)
    calls = list(measurement.calls.values())
    result = {
        "variant": args.variant, "case": args.case, "repeat": args.repeat,
        "baseline_commit": args.baseline, "requested_model": config.model,
        "context_window_tokens": config.model_limits.context_window_tokens,
        "context_source": "experiment_user_assumption", "context_policy": asdict(config.context_policy),
        "tool_names": list(registry.names), "instructions": instructions, "task": f.task,
        "schema_estimated_tokens": HeuristicTokenEstimator().estimate_breakdown(
            instructions="", input_items=(), tools=registry.definitions())["tool_definitions"],
        "answer": answer, "error": error, "grade": grading,
        "passed": error is None and grading["passed"], "elapsed_seconds": elapsed,
        "requests": measurement.responses, "model_requests": len(measurement.responses),
        "input_tokens": sum((r["usage"] or {}).get("input_tokens", 0) for r in measurement.responses),
        "output_tokens": sum((r["usage"] or {}).get("output_tokens", 0) for r in measurement.responses),
        "cached_input_tokens": sum(((r["usage"] or {}).get("input_tokens_details") or {}).get("cached_tokens", 0)
                                   for r in measurement.responses),
        "tools": calls, "tool_calls": len(calls), "tool_errors": sum(not c.get("success", False) for c in calls),
        "command_nonzero": sum(c.get("exit_code") not in (None, 0) for c in calls),
        "rg_no_matches": sum(c.get("rg_no_match", False) for c in calls),
        "approval_count": measurement.approval_count,
        "diff_events": sum(c.get("has_diff", False) for c in calls),
    }
    save(directory / "result.json", result)
    print(json.dumps({k: result[k] for k in ("variant", "case", "repeat", "passed", "elapsed_seconds", "tool_calls", "tool_errors")}), flush=True)


def summary(output: Path) -> list[dict]:
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(output.glob("r*/result.json"))]
    rows = []
    for variant in VARIANTS:
        group = [r for r in results if r["variant"] == variant]
        if not group:
            continue
        rows.append({"variant": variant, "trials": len(group), "passed": sum(r["passed"] for r in group),
            "median_seconds": round(statistics.median(r["elapsed_seconds"] for r in group), 2),
            "total_input_tokens": sum(r["input_tokens"] for r in group),
            "total_output_tokens": sum(r["output_tokens"] for r in group),
            "total_cached_input_tokens": sum(r["cached_input_tokens"] for r in group),
            "model_requests": sum(r["model_requests"] for r in group),
            "tool_calls": sum(r["tool_calls"] for r in group),
            "tool_errors": sum(r["tool_errors"] for r in group),
            "command_nonzero": sum(r["command_nonzero"] for r in group),
            "rg_no_matches": sum(r["rg_no_matches"] for r in group),
            "approval_count": sum(r["approval_count"] for r in group),
            "schema_estimated_tokens": group[0]["schema_estimated_tokens"],
            "returned_models": sorted({q["model"] for r in group for q in r["requests"]}),
            "failures": [f'r{r["repeat"]}/{r["case"]}' for r in group if not r["passed"]]})
    save(output / "summary.json", rows)
    if rows:
        with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def launch(args, repeat, case, variant):
    name = f"r{repeat}-{case}-{variant}"
    directory = args.output / name
    if (directory / "result.json").exists():
        return name + " already recorded"
    directory.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", "--output", str(args.output),
           "--variant", variant, "--case", case, "--repeat", str(repeat), "--model", args.model, "--baseline", args.baseline]
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        process = CommandProcess(cmd, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            exit_code = process.process.wait(timeout=280)
        except subprocess.TimeoutExpired:
            exit_code = -1
        finally:
            process.close()
    if (directory / "result.json").exists():
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        return f'{name}: {"PASS" if result["passed"] else "FAIL"}, {result["elapsed_seconds"]:.1f}s, {result["tool_calls"]} tools, {result["tool_errors"]} errors'
    return f"{name}: worker failure {exit_code}; see worker.log"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="auto(free)")
    parser.add_argument("--baseline", default="0347ef9b332423bff046eba325c27908a97354b5")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--case", choices=CASES, default="search_noise")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        trial(args)
        return
    if args.summarize:
        print(json.dumps(summary(args.output), ensure_ascii=False, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    schedule = [(r, case, v) for r in range(args.repeats) for case in args.cases for v in args.variants]
    random.Random(4812).shuffle(schedule)
    save(args.output / "design.json", {"baseline": args.baseline, "model": args.model,
         "variants": {k: sorted(v) for k, v in VARIANTS.items()}, "context_window": 1_000_000,
         "repeats": args.repeats, "workers": args.workers, "schedule": schedule,
         "limits": {"model_requests": 16, "per_request_output_tokens": 8192, "trial_seconds": 240},
         "controls": "Real Praxis Agent/ToolCallExecutor; same neutral instructions, skills, auxiliary tools, auto-approved commands; no MCP, child agents, user skills, production session writes or UI. Independent fixture copies and external grading; provider token usage, not estimated usage. Context is user-specified, not a 1M stress test."})
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(launch, args, *item) for item in schedule]
        for done in as_completed(futures):
            print(done.result(), flush=True)
            summary(args.output)
    print(json.dumps(summary(args.output), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
