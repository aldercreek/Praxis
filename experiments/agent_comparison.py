"""Compare real execution engines on identical isolated tasks; no production edits.

WorkBuddy uses its installed lite-wb engine and documented custom-model file.
Codex uses its installed CLI with an enabled restricted-token Windows sandbox.
The primary comparison requests the same gateway alias, while codex_native is
an explicitly separate reference using the user's configured Codex model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import ssl
import statistics
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from threading import Thread, Timer

import tool_design_benchmark as base

REPO = base.REPO
DEFAULT_OUTPUT = Path("D:/Users/wanjl/Downloads/praxis-agent-comparison-20261018")
CODEX = Path("D:/Users/wanjl/AppData/Roaming/npm/node_modules/@openai/codex/node_modules/@openai/codex-win32-x64/vendor/x86_64-pc-windows-msvc/bin/codex.exe")
WORKBUDDY = Path("D:/Users/wanjl/AppData/Local/Programs/WorkBuddyAI/resources/app.asar.unpacked/cli/bin/codebuddy")
MODEL = "auto(free)"
CASES = (*base.CASES, "output_tail", "newline_matrix")
ENGINES = ("praxis", "codex", "workbuddy", "codex_native")


def make_fixture(directory: Path, case: str, repeat: int) -> base.Fixture:
    if case in base.CASES:
        return base.fixture(directory, case, repeat)
    root = directory / "sandbox"
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    base.put(root, ".gitignore", "__pycache__/\n*.pyc\n")
    if case == "output_tail":
        expected = {"records": 3000, "checksum": 918271 + repeat, "status": "verified"}
        script = (
            "import json\nfrom pathlib import Path\n"
            f"result = {expected!r}\n"
            "Path('executed.json').write_text(json.dumps(result), encoding='utf-8')\n"
            "print('BEGIN REPORT')\n"
            "for i in range(3000): print(f'row {i:05d}: ' + 'normal diagnostic data; ' * 5)\n"
            "print('FINAL_RESULT=' + json.dumps(result))\n"
        )
        base.put(root, "noisy_report.py", script)
        return base.Fixture(root, root,
            "运行 noisy_report.py；它会输出大量诊断日志，最终结果在输出末尾。"
            "只把 FINAL_RESULT 的 JSON 对象作为最终回答。不要修改程序。"
            "程序自己生成 executed.json，这是预期的运行产物。", expected, ("executed.json",))
    target = str(81 + repeat)
    expected = {}
    for name, ending, bom in (("lf.txt", "\n", b""), ("crlf.txt", "\r\n", b""), ("bom.txt", "\r\n", b"\xef\xbb\xbf")):
        text = ending.join(("# 配置；只修改 active 项", "backup_timeout=30", "active_timeout=30", ""))
        original = bom + text.encode("utf-8")
        path = root / name
        path.write_bytes(original)
        expected[name] = (bom + text.replace("active_timeout=30", "active_timeout=" + target).encode("utf-8")).hex()
    return base.Fixture(root, root,
        f"把 lf.txt、crlf.txt、bom.txt 中的 active_timeout=30 改成 active_timeout={target}。"
        "backup_timeout 保持 30。必须保留每个文件原来的编码、BOM、换行符和所有其他字节；检查修改结果。",
        expected, tuple(expected))


def grade(f: base.Fixture, case: str, answer: str, before: dict) -> dict:
    if case in base.CASES:
        return base.grade(f, case, answer, before)
    after = base.snapshot(f.root)
    changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    unexpected = [p for p in changed if p not in f.allowed_changes]
    auxiliary_output_files = []
    if case == "output_tail":
        # This task permits capturing command output; it never required deleting
        # the resulting log. Accept exact stdout copies, not arbitrary new files.
        stdout = "BEGIN REPORT\n" + "".join(f"row {i:05d}: " + "normal diagnostic data; " * 5 + "\n" for i in range(3000))
        stdout += "FINAL_RESULT=" + json.dumps(f.expected) + "\n"
        for name in unexpected:
            path = f.root / name
            if name not in before and path.suffix in {".txt", ".log"}:
                try:
                    if path.read_text(encoding="utf-8-sig") == stdout:
                        auxiliary_output_files.append(name)
                except (OSError, UnicodeError):
                    pass
        unexpected = [p for p in unexpected if p not in auxiliary_output_files]
    checks = {"only_allowed_files_changed": not unexpected}
    if case == "newline_matrix":
        checks.update({name: (f.root / name).is_file() and (f.root / name).read_bytes().hex() == data
                       for name, data in f.expected.items()})
    else:
        checks["answer_matches_oracle"] = base.extract_json(answer) == f.expected
        marker = f.root / "executed.json"
        try:
            checks["program_actually_executed"] = json.loads(marker.read_text(encoding="utf-8")) == f.expected
        except (OSError, ValueError):
            checks["program_actually_executed"] = False
    return {"passed": all(checks.values()), "checks": checks, "changed_files": changed, "unexpected_changes": unexpected,
            "auxiliary_output_files": auxiliary_output_files}


def task_prompt(f: base.Fixture) -> str:
    return (
        f"这是独立的执行实验。唯一允许操作的任务范围是 {f.root}，当前目录是 {f.workspace}。"
        "请直接执行任务；不要访问网络、读取环境变量/凭据、操作其他目录或调用其他 Agent。"
        "不需要写报告或 Markdown 文件。请使用当前产品已有的本地文件/搜索/命令工具。"
        f"本机是 Windows，提供 rg、PowerShell、pwsh，Python 解释器为 {sys.executable}。"
        "任务结果以实际文件与执行结果为准，不能只声称完成。\n\n" + f.task
    )


def redact(text: str) -> str:
    for key in ("ADA_API_KEY", "OPENAI_API_KEY", "CODEX_API_KEY", "CODEBUDDY_API_KEY"):
        secret = os.environ.get(key, "")
        if len(secret) > 8:
            text = text.replace(secret, "[REDACTED]")
    return text


def sanitize_event(event: dict) -> dict | None:
    if event.get("item", {}).get("type") == "reasoning":
        return None
    message = event.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), list):
        message["content"] = [part for part in message["content"] if part.get("type") not in {"thinking", "reasoning", "redacted_thinking"}]
    event.pop("_meta", None)
    return event


def command_options(engine: str, f: base.Fixture, directory: Path, output: Path) -> tuple[list[str], dict, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    url = base.ModelConfig.from_env().base_url
    if engine.startswith("codex"):
        cmd = [str(CODEX), "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "--json",
               "-C", str(f.workspace), "--add-dir", str(f.root), "-s", "workspace-write",
               "-c", 'windows.sandbox="unelevated"', "-c", 'approval_policy="never"',
               "-c", "features.memories=false", "-c", "features.hooks=false"]
        if engine == "codex_native":
            config = tomllib.loads(Path("D:/Users/wanjl/.codex/config.toml").read_text(encoding="utf-8"))
            model, provider = config["model"], config["model_provider"]
            cmd += ["-c", "model=" + json.dumps(model), "-c", "model_provider=" + json.dumps(provider),
                    "-c", "model_reasoning_effort=" + json.dumps(config.get("model_reasoning_effort", "high"))]
            for key, value in config["model_providers"][provider].items():
                cmd += ["-c", f"model_providers.{provider}.{key}=" + json.dumps(value)]
        else:
            model = MODEL
            cmd += ["-c", "model=" + json.dumps(model), "-c", "model_context_window=1000000",
                    "-c", 'model_provider="benchmark"', "-c", 'model_providers.benchmark.name="Experiment gateway"',
                    "-c", "model_providers.benchmark.base_url=" + json.dumps(url),
                    "-c", 'model_providers.benchmark.env_key="ADA_API_KEY"',
                    "-c", 'model_providers.benchmark.wire_api="responses"']
        return cmd + ["-"], env, model
    config = directory / "engine-config"
    config.mkdir()
    base.save(config / "models.json", {"models": [{"id": MODEL, "name": "Experiment DeepSeek",
              "url": url, "apiKey": "${ADA_API_KEY}", "maxAllowedSize": 1000000, "maxInputTokens": 1000000,
              "maxOutputTokens": 8192, "supportsToolCall": True, "supportsImages": False}], "availableModels": [MODEL]})
    env.update(CODEBUDDY_CONFIG_DIR=str(config), WORKBUDDY_CONFIG_DIR=str(config),
               CODEBUDDY_FORCE_LITE_WB_BUNDLE="1", NODE_EXTRA_CA_CERTS=str(output / "windows-trusted-ca.pem"))
    cmd = ["node", str(WORKBUDDY), "-p", "--model", "custom-local:" + MODEL, "--output-format", "stream-json", "--verbose",
           "--no-session-persistence", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
           "--permission-mode", "acceptEdits", "--allowedTools", "Read", "Grep", "Glob", "Bash", "PowerShell", "Edit", "Write",
           "--max-turns", "24"]
    return cmd, env, MODEL


def run_external(engine: str, f: base.Fixture, directory: Path, output: Path) -> dict:
    cmd, env, model = command_options(engine, f, directory, output)
    events, stderr = [], []
    started = time.monotonic()
    timeout = False
    with closing(base.CommandProcess(cmd, cwd=f.workspace, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=env, text=True, encoding="utf-8", errors="replace")) as process:
        def read_stdout():
            with (directory / "events.jsonl").open("w", encoding="utf-8") as log:
                for line in process.process.stdout:
                    try:
                        event = sanitize_event(json.loads(redact(line)))
                    except ValueError:
                        continue
                    if event is not None:
                        event["observed_seconds"] = round(time.monotonic() - started, 4)
                        events.append(event)
                        log.write(json.dumps(event, ensure_ascii=False) + "\n")
                        log.flush()

        def read_stderr():
            for line in process.process.stderr:
                stderr.append(redact(line))

        readers = [Thread(target=read_stdout), Thread(target=read_stderr)]
        for reader in readers:
            reader.start()
        process.process.stdin.write(task_prompt(f))
        process.process.stdin.close()
        try:
            process.process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            timeout = True
        finally:
            process.close()
            for reader in readers:
                reader.join(timeout=10)
        exit_code = process.process.returncode
    (directory / "stderr.log").write_text("".join(stderr), encoding="utf-8")
    result = parse_external(engine, events)
    result.update(requested_model=model, elapsed_seconds=time.monotonic() - started,
                  process_exit_code=exit_code, timed_out=timeout)
    return result


def parse_external(engine: str, events: list[dict]) -> dict:
    tools, requests, answers, errors, usage = {}, {}, [], [], {}
    for event in events:
        if engine.startswith("codex"):
            item = event.get("item", {})
            if event.get("type") == "item.completed":
                kind = item.get("type")
                if kind == "agent_message":
                    answers.append(item.get("text", ""))
                elif kind in {"command_execution", "file_change", "mcp_tool_call"}:
                    tools[item["id"]] = {"name": kind, **item, "success": item.get("status") == "completed",
                                         "output": item.get("aggregated_output", ""), "seconds": event["observed_seconds"]}
                elif kind == "error":
                    errors.append(item.get("message", ""))
            if event.get("type") == "turn.completed":
                usage = event.get("usage", {})
            if event.get("type") in {"turn.failed", "error"}:
                errors.append(str(event.get("error", event.get("message", ""))))
        else:
            message = event.get("message", {})
            if event.get("type") == "assistant":
                token_usage = message.get("usage", {})
                if token_usage.get("input_tokens", 0):
                    requests[message["id"]] = {"model": message.get("model"), "usage": token_usage}
                for part in message.get("content", []):
                    if part.get("type") == "tool_use":
                        tools[part["id"]] = {"id": part["id"], "name": part["name"], "arguments": part["input"],
                                             "seconds": event["observed_seconds"]}
                    elif part.get("type") == "text":
                        answers.append(part["text"])
            elif event.get("type") == "user":
                for part in message.get("content", []):
                    if part.get("type") == "tool_result":
                        call = tools.setdefault(part["tool_use_id"], {"name": "unknown"})
                        content = part.get("content", "")
                        call.update(success=not part.get("is_error", False), output=content if isinstance(content, str)
                                    else "\n".join(x.get("text", "") for x in content))
            elif event.get("type") == "result":
                usage = event.get("usage", {})
                errors.extend(event.get("errors", []))
                if event.get("result"):
                    answers.append(event["result"])
    calls = list(tools.values())
    return {"answer": answers[-1] if answers else "", "tools": calls, "tool_calls": len(calls),
            "tool_errors": sum(not c.get("success", False) for c in calls), "engine_errors": errors,
            "usage": usage, "requests": list(requests.values()), "model_requests": len(requests) if requests else None,
            "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
            "cached_input_tokens": usage.get("cached_input_tokens", usage.get("cache_read_input_tokens", 0))}


def run_praxis(f: base.Fixture, directory: Path) -> dict:
    config0 = base.ModelConfig.from_env(model=MODEL)
    config = base.ModelConfig(api_key=config0.api_key, model=MODEL, base_url=config0.base_url,
                             model_limit_overrides=base.ModelLimitOverrides(context_window_tokens=1_000_000))
    measurement = base.Measurement(directory)
    control = base.ExecutionControl()
    runtime_context = base.collect_runtime_context(f.workspace)
    registry = base.create_default_registry(config.context_policy, check_execution=control.check,
                                            command_shell=runtime_context.command_shell)
    code = base.CodeRuntime()
    registry.register(code.tool("parent"))
    skills = base.discover_skills(f.workspace, user_root=directory / "no-user-skills")
    registry.register(base.create_skill_tool(skills, max_chars=config.context_policy.max_tool_output_chars))
    instructions = base.build_instructions(runtime_context, skills=skills)
    client = base.MeasuredClient(config, instructions=instructions, execution_control=control, measurement=measurement)
    agent = base.Agent(model_client=client, tool_registry=registry, workspace=f.workspace, event_sink=measurement,
        context_manager=base.ContextManager(model_limits=config.model_limits, policy=config.context_policy, instructions=instructions),
        execution_control=control, approve_tool_call=lambda *_: True, close_callbacks=(code.close, client.close))
    agent.code_session_context = lambda: code.context_item("parent")
    scope = base.Run("comparison")
    timer = Timer(240, scope.cancel)
    timer.daemon = True
    started = time.monotonic()
    timer.start()
    answer, errors = "", []
    try:
        answer = scope.execute(lambda: agent.run(task_prompt(f)))
    except Exception as exc:
        errors.append(redact(f"{type(exc).__name__}: {exc}"))
    finally:
        timer.cancel()
        agent.close()
    calls = list(measurement.calls.values())
    requests = measurement.responses
    return {"requested_model": MODEL, "answer": answer, "elapsed_seconds": time.monotonic() - started,
        "tools": calls, "tool_calls": len(calls), "tool_errors": sum(not c.get("success", False) for c in calls),
        "engine_errors": errors, "requests": requests, "model_requests": len(requests), "timed_out": scope.cancelled.is_set(),
        "input_tokens": sum((r["usage"] or {}).get("input_tokens", 0) for r in requests),
        "output_tokens": sum((r["usage"] or {}).get("output_tokens", 0) for r in requests),
        "cached_input_tokens": sum(((r["usage"] or {}).get("input_tokens_details") or {}).get("cached_tokens", 0) for r in requests),
        "approval_count": measurement.approval_count, "instructions": instructions}


def worker(args):
    directory = args.output / "trials" / f"r{args.repeat}-{args.case}-{args.engine}"
    directory.mkdir(parents=True, exist_ok=True)
    f = make_fixture(directory, args.case, args.repeat)
    before = base.snapshot(f.root)
    base.save(directory / "fixture.json", {"case": args.case, "task": task_prompt(f), "expected": f.expected,
              "allowed_changes": f.allowed_changes, "before": before})
    result = run_praxis(f, directory) if args.engine == "praxis" else run_external(args.engine, f, directory, args.output)
    grading = grade(f, args.case, result["answer"], before)
    result.update(engine=args.engine, case=args.case, repeat=args.repeat, grade=grading,
                  passed=grading["passed"] and not result["timed_out"])
    base.save(directory / "result.json", result)


def summarize(output: Path):
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output / "trials").glob("*/result.json"))]
    rows = []
    for engine in ENGINES:
        group = [r for r in results if r["engine"] == engine]
        if group:
            rows.append({"engine": engine, "trials": len(group), "passed": sum(r["passed"] for r in group),
                "mean_seconds": round(statistics.mean(r["elapsed_seconds"] for r in group), 2),
                "median_seconds": round(statistics.median(r["elapsed_seconds"] for r in group), 2),
                "tool_calls": sum(r["tool_calls"] for r in group), "tool_errors": sum(r["tool_errors"] for r in group),
                "input_tokens": sum(r["input_tokens"] for r in group), "output_tokens": sum(r["output_tokens"] for r in group),
                "cached_input_tokens": sum(r["cached_input_tokens"] for r in group),
                "failures": [f'r{r["repeat"]}/{r["case"]}' for r in group if not r["passed"]]})
    base.save(output / "summary.json", rows)
    for name, data in (("summary.csv", rows), ("trials.csv", [{k: r[k] for k in
            ("engine", "case", "repeat", "passed", "elapsed_seconds", "tool_calls", "tool_errors", "input_tokens", "output_tokens")} for r in results])):
        if data:
            with (output / name).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    return rows


def launch(args, repeat, case, engine):
    directory = args.output / "trials" / f"r{repeat}-{case}-{engine}"
    if (directory / "result.json").exists():
        return directory.name + " already recorded"
    directory.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", "--output", str(args.output),
           "--repeat", str(repeat), "--case", case, "--engine", engine]
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        process = base.CommandProcess(cmd, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            process.process.wait(timeout=285)
        except subprocess.TimeoutExpired:
            pass
        finally:
            process.close()
    if not (directory / "result.json").exists():
        return directory.name + ": WORKER FAILURE"
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    return f'{directory.name}: {"PASS" if result["passed"] else "FAIL"}, {result["elapsed_seconds"]:.1f}s, {result["tool_calls"]} tools'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--engine", choices=ENGINES, default="praxis")
    parser.add_argument("--case", choices=CASES, default="search_noise")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES[:3]))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    if args.summarize:
        print(json.dumps(summarize(args.output), ensure_ascii=False, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    certs = []
    for store in ("ROOT", "CA"):
        for cert, encoding, trust in ssl.enum_certificates(store):
            if encoding == "x509_asn" and (trust is True or ssl.Purpose.SERVER_AUTH.oid in trust):
                certs.append(ssl.DER_cert_to_PEM_cert(cert))
    (args.output / "windows-trusted-ca.pem").write_text("".join(dict.fromkeys(certs)), encoding="ascii")
    schedule = [(repeat, case, engine) for repeat in range(args.repeats) for case in args.cases for engine in args.engines]
    random.Random(20261018).shuffle(schedule)
    native = args.engines == ["codex_native"]
    native_config = tomllib.loads(Path("D:/Users/wanjl/.codex/config.toml").read_text(encoding="utf-8")) if native else {}
    base.save(args.output / ("design-native.json" if native else "design.json"), {
        "baseline": "0347ef9b332423bff046eba325c27908a97354b5",
        "requested_model": native_config.get("model", MODEL),
        "context_window_assumption": None if native else 1000000,
        "schedule": schedule, "workers": args.workers, "timeout_seconds": 240,
        "protocols": {"codex": "responses", "praxis": "responses", "workbuddy": "chat_completions"},
        "controls": ("Separate reference using the configured Codex model/provider and reasoning effort; do not merge with the same-alias group. " if native else "Same requested gateway alias; returned model recorded where exposed. The gateway rejects direct requests for its returned DeepSeek model id, so routing cannot be pinned independently. ") + "Identical per-case task instructions and fixture content, independent byte/behavior grades, no user assistance, no MCP or delegated agents; retain product prompts and local tools. WorkBuddy bundled engine with custom model, not desktop login/UI. Codex unknown-model metadata fallback is recorded. Model output/turn limits and tool security semantics remain product-specific. External raw timings include CLI startup; Praxis starts after initialization. Token accounting is not equal to billed cost.",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = [executor.submit(launch, args, *item) for item in schedule]
        for done in as_completed(pending):
            print(done.result(), flush=True)
            summarize(args.output)
    print(json.dumps(summarize(args.output), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
