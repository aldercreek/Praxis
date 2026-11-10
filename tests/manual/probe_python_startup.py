"""Bounded real-process probe for intermittent kernel startup failures.

Does not change production code. Capture process exit/diagnostics BEFORE close
destroys the process handle, to distinguish transport failure from early exit.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from complex_code_cases import dump
from run_complex_code_tasks import alive

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from praxis_agent.planning import RunState
from praxis_agent.tools import ToolContext
from praxis_agent.tools.code import CodeRuntime
from praxis_agent.tools.python_kernel import PythonKernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    records, closes = [], []
    lock = threading.Lock()
    original = PythonKernel.close

    async def capture_close(kernel):
        if kernel.process:
            entry = dict(pid=kernel.worker_pid, exit_before_close=kernel.process.process.poll(),
                         diagnostics=kernel.diagnostics, connection_directory=str(kernel.directory.name) if kernel.directory else None)
            with lock:
                closes.append(entry)
        await original(kernel)

    PythonKernel.close = capture_close

    def probe(index):
        workspace = root / f"case-{index:02}"
        workspace.mkdir()
        runtime = CodeRuntime(environment_dir=workspace / "environment")
        before = time.monotonic()
        try:
            result = runtime.run(ToolContext(workspace, RunState("startup probe")), "import os, sys\n{'pid':os.getpid(),'python':sys.executable}")
            return dict(index=index, success=result.success, seconds=round(time.monotonic() - before, 4), output=json.loads(result.output))
        except Exception as error:
            return dict(index=index, success=False, seconds=round(time.monotonic() - before, 4), error=f"{type(error).__name__}: {error}")
        finally:
            runtime.close()

    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for future in as_completed([pool.submit(probe, i) for i in range(args.count)]):
                record = future.result()
                records.append(record)
                dump(root / "progress.json", records)
                print(json.dumps({k: record[k] for k in ("index", "success", "seconds")}), flush=True)
    finally:
        PythonKernel.close = original
    report = dict(count=args.count, jobs=args.jobs, successes=sum(r["success"] for r in records),
                  all_processes_dead=all(not alive(c["pid"]) for c in closes),
                  records=sorted(records, key=lambda r: r["index"]), closes=closes)
    dump(root / "result.json", report)


if __name__ == "__main__":
    main()
