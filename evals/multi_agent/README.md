# Live multi-agent evaluation

Runs the real Praxis factory, model/tool loop, worker pool, team snapshot, mailboxes,
context inheritance and finalization against a live Responses-compatible model.
An empty MCP configuration keeps user integrations out of the test.

```powershell
$env:PYTHONPATH = (Resolve-Path src)
python evals/multi_agent/run.py --output D:/temp/praxis-eval
```

Use `ADA_API_KEY` and optionally `ADA_BASE_URL` from the environment. The default
model is `DeepSeek-V4-Pro-discount`; `--model` selects another provider model ID.
Keys are never written to artifacts. `--cases` selects tasks, `--parallel` controls
concurrent sessions, and `--seconds` / `--max-calls` bound each run. Use a fresh
output directory for every run. No running Praxis service is reconfigured.

| Task | Difficulties | Independent check |
|---|---|---|
| incident | 240 requests, cross-system joins, three clock offsets, tenant contamination, replayed ledger events and negative price differences | Exact affected ID sets, net financial delta and defective key |
| settlement | 350 logical events with revisions/replays, three currencies, explicit half-cent ties, repeating thirds, half-even rounding, ceiling fees and six bank balances | Exact tenant totals, differences and dedup counts |
| routes | Three cargo orders, 56 flights, transfers, carbon/deadline/leg limits and shared capacity; cancellation injected while parent waits | Exhaustive joint path optimum and deterministic tie-breaking after disruption |
| schedule | Eight dependent jobs, two workers, CPU/lab capacity and maintenance; close a specialist and create a verifier | Exhaustive time-state optimum and independent schedule feasibility |
| inventory | Repair an actual engine: tenant isolation, successful/failed replay, conflicting payloads, expiry, atomic validation and concurrency | Fresh-process tests withheld from model workspace |

Each task requests at least three actual workers, direct peer exchanges, inherited
background and parent waits. The scheduler task also requests close/replacement.
The grader separately reports artifact correctness and observed collaboration;
plausible prose alone does not pass. It records requested and returned model IDs,
requests, token usage, wall time, tool failures and collaboration counts.

Artifacts per task: `task.txt`, `trace.jsonl`, `session.sqlite3`, `final.txt`,
`report.json` and the model's workspace. `summary.json` aggregates results.
Expected answers and grading code are not placed in the model workspace.

These are bounded synthetic end-to-end checks, not a statistical model ranking.
Token totals cover recorded normal model responses, not an invoiced cost estimate
or unobserved provider retries. A waiting worker occupies a pool thread; the runner
reserves five worker threads per session for these tasks.
