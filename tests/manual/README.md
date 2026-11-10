# Code execution evaluations

These opt-in scripts exercise the real Agent and model. They are not collected
by pytest. Use the normal Praxis model configuration and credentials; generated
workspaces, traces and reports belong under the ignored `build/` directory.

```powershell
.venv/Scripts/python.exe tests/manual/run_complex_code_tasks.py --case all --output build/complex-code --jobs 2
.venv/Scripts/python.exe tests/manual/probe_python_startup.py --output build/startup-probe --count 24 --jobs 2
```

The startup probe does not call a model. To prepare deterministic fixtures
without calling the model, add `--prepare-only` to the complex-task runner.
Choose a new output directory for each run to preserve earlier evidence.

`compare_code_backends.py` accepts `--source`, `--output`, `--backend` (a report
label), `--case` and `--model`. Point `--source` at a separate historical checkout
to compare an old implementation; the current production code contains no Monty
backend. Cases are `csv_state`, `project_library`, `sqlite_zip`, `mcp_parallel`.

## Observed results, November 10, 2026

Environment: Windows, CPython 3.14.5, ipykernel and jupyter_client. Requested model
`deepseek-v4.1-flash`; provider reported `deepseek-v4-1-flash-261017`.

The initial four-task comparison completed all artifacts on both backends.
Monty produced 3 failed calls out of 8 `run_code` calls and used 6 shell calls
for project-library and SQLite tasks. The initial CPython run produced no failed
calls out of 14 `run_code` calls and used no shell calls. Later runs exposed a
callback-result-shape mistake, so these small samples do not establish a stable
error-rate or speed advantage.

Complex evaluation covered reconciliation, project repair, SQLite synchronization,
paged MCP data, large logs, persistent-state recovery, cancellation/resumption,
and offline package installation. Eight initial tasks plus two repeats yielded:

- Correct artifacts: 10/10 attempts.
- All process constraints: 9/10 attempts, after review of the initial MCP verdict.
- `run_code`: 74 calls, 6 failures, including 2 deliberate error/cancellation cases.
- Unexpected failures: 3 model code/resource-management errors and 1 kernel startup timeout.
- Additional cold-start probe: 24/24 successes at concurrency 2, 1.10–1.83 seconds;
  all tracked kernel processes exited after cleanup.

These are observations from this machine and model, not guarantees. The original
traces remain under `build/complex-code-20261110-r1/` and `-r2/`; the reviewed
report and issue list are under `build/complex-code-report/` locally.

## Recorded issues

| Issue | Observation | Status |
| --- | --- | --- |
| PY-01 | One kernel startup waited 30 seconds and timed out before user code ran. Retry and 24 subsequent cold starts passed. | Unresolved; not reproduced. Capture exit code and startup-stage diagnostics before claiming a fix. |
| MCP-01 | A repeat read `rate` instead of `cny_rate`, then retried successful requests. Artifacts were correct, but 12 rate queries exceeded the required 4. | Unresolved model contract/retry-budget failure. |
| MCP-02 | Confused batch `output` with MCP `structuredContent`. | Recovered using retained results; clarify the envelope in examples. |
| PY-02 | Missing `import io` during an extra validation step. | Model corrected the import. |
| PY-03 | Mixed `Fraction` and `Decimal` in a sum. | Model switched to integer minor units. |
| PY-04 | An open SQLite connection prevented deleting its file on Windows. | Explicit close recovered; a SQLite transaction context does not close the connection. |
| MCP-03 | Tool activation with arguments performed an extra query whose result was not reused. | Recorded efficiency issue; absent in the repeat. |

The UI adaptation does not fix the unresolved runtime/model issues above. It
displays execution results and cancellation without exposing kernel-state metadata.
