---
name: monty
description: Official Monty 1.2.0 guide for Praxis run_code. Load when composing file, MCP, or history tools, parsing CSV/JSON/TOML, reusing session state, or recovering from execution failures. Covers actual tool return formats, resource limits, and recovery examples.
---

# Using Monty in Praxis

Supported version: the official PyPI `pydantic-monty==1.2.0` package and matching client/runtime. This guide describes the Praxis integration. Official API support does not mean Praxis exposes the required host resources.

## Choose an execution method first

| Task | Method |
| --- | --- |
| A simple read, search, or edit | Call the corresponding tool directly |
| Use one tool result in the next call, loop over inputs, or return only a summary to the model | `run_code` + `call_tool` |
| Parse JSON, CSV, or TOML text already retrieved | Built-in modules in `run_code` |
| Run project code, tests, third-party packages, system commands, or full Python scripts | Call `run_command` separately, outside the Monty bridge |

If the current role lacks `run_command`, explain the missing capability. Reimplementing an algorithm in Monty does not prove that the actual project code ran or its tests passed.

## Execution, inputs, and outputs

Pass Python source as `code` to `run_code`; there is no `session_id` parameter. Each agent creates its own Monty worker on first use. Later calls reuse variables and functions in the same memory. Different agents are isolated.

Praxis locates the current agent's worker (creating it only on first use), type-checks the code, evaluates it in the existing environment, executes tools when `call_tool` is reached, passes results back, and returns the result while retaining the worker. There is no snapshot loading, saving, or rollback.

Praxis provides `call_tool` and `call_parallel_tools` without imports. Tools handle host operations such as files and network services; Monty handles loops, conditions, parsing, and aggregation. A single `call_tool` waits synchronously. Use `call_parallel_tools` for independent calls executed concurrently on the host; code waits for the whole batch without managing an event loop. Calls to `run_code` within one agent remain sequential; different agents can run concurrently.

The `run_code` response is a JSON object. Check these fields:

| Field | Meaning |
| --- | --- |
| `result` | Value of the last expression; prefer a small object or a few samples |
| `stdout` / `stderr` | Separate output previews; each stream retains up to the first 6,000 and last 24,000 Python characters, with an omission notice if needed |
| `output_files` | Entries only for streams exceeding the preview limit, including the full log's `path`, `total_chars`, `omitted_chars`, and `complete`; read the source with `read_file` |
| `output_errors` | Diagnostics when saving logs fails; logs with `complete=false` may be incomplete |
| `calls` | Records of tool calls that returned, including names, success, and failure details; excludes full business results |
| `worker_created` | Whether this call created a new empty execution environment |
| `state_lost` | Whether the environment was discarded; when true, previous variables are lost and must be initialized again |
| `error` / `error_kind` / `error_type` / `phase` | Failure diagnostics, category, exception type, and phase |
| `effects` | Reminder that completed external operations are not rolled back |

Print only the summaries and samples needed for the task. Output exceeding preview limits is saved in full as UTF-8 files under `~/.praxis/outputs/code/`, including the displayed beginning and end. History retains previews and file references. `read_history` does not expand log files; use `read_file` with absolute paths from `output_files`. Output already received is preserved on failure or cancellation. `result` is independent of print preview limits and follows the normal tool preview/history flow. Use file tools for deliverables that must persist.

## Use the tool bridge correctly

Call tools with `call_tool(name, arguments)`, where `arguments` is a dictionary matching the current tool schema. Follow the tool definition for parameter names, pagination fields, and coordinate conventions.

For independent operations, use `call_parallel_tools([{"name": "actual_tool_name", "arguments": {...}}, ...])`. The list can be built dynamically and contain different tools; an empty list returns `[]`. Including a call explicitly requests concurrency without checking `parallel_safe`. Role restrictions, the programmatic tool allowlist, argument validation, and approvals still apply. Do not batch calls that depend on earlier results or file edits that could overwrite each other.

By default, up to 16 calls execute concurrently per batch. The host can adjust this with `PRAXIS_CODE_MAX_PARALLEL_TOOLS`. Remaining calls queue without a total item limit. Results follow input order as `{"success": True, "output": data}` or `{"success": False, "error": explanation}`. Successful `output` values follow the structured-result preference below. One item failing does not fail the batch or trigger a retry; inspect every `success` field. A malformed call list is rejected before any tool runs. Cancellation stops dispatch, cancels running tools that support cancellation, and waits for cleanup. Completed external operations are not rolled back.

- Call file tools, history search/read tools, and activated MCP tools allowed for the current agent. The original executor still handles role restrictions, validation, approvals, and events.
- Discover and activate MCP tools through direct calls first. Use the registered callable name after activation; do not guess how service and tool names are combined.
- Do not call `run_command`, management tools, or nested `run_code`. Finish the current computation and call the command tool separately when needed.
- Call tools in ordinary loops. Do not call tools inside built-in callbacks such as `sorted(key=...)`, `map`, or `filter`. Retrieve data first, then process it with pure computation callbacks.

### Return values are not always JSON strings

The bridge prefers a tool's structured object and returns an output string only when there is no structured result.

| Tool | Value received in Monty | Recommended handling |
| --- | --- | --- |
| `read_file` | Dictionary with `text`, `truncated`, `next_position`, and `note` | Read `reply["text"]` directly; do not apply `json.loads` to the whole dictionary |
| Other file tools | Structured dictionary; see tool definitions for fields | For example, writes return `path` and file searches return `files` |
| `read_history` | Structured dictionary | Read text and pagination fields according to the schema |
| `search_history` | JSON string | Decode once with `json.loads` |
| Activated MCP tools | Complete response dictionary: `content`, optional `structuredContent`, `isError`, and any metadata | Read fields directly; never apply `json.loads` to the whole response. `structuredContent` may be any JSON value, including `{}`, `[]`, `0`, or `False` |

MCP `content` can have multiple blocks. A block with `type == "text"` may contain ordinary text; decode it only after confirming it is JSON. Do not assume the first block contains JSON. Images, audio, resource links, and embedded resources remain in their original MCP format in code; avoid printing binary/base64 payloads. Read `reply.get("structuredContent")` for business data and `reply["content"]` for accompanying text or media. Failure still raises `RuntimeError` with details; no automatic retry occurs. For an unfamiliar tool, retrieve a small sample to confirm its structure, then reuse that sample and parsing logic.

Programmatic `read_file` and `read_history` avoid model-preview character truncation but still return only the requested range, which may cover just part of a file. Remote pagination and service limits still apply; a successful call does not imply complete data. Decode JSONL one nonempty line at a time rather than passing the whole text to one `json.loads`.

### Example: read CSV and TOML, then save a summary

Assume the workspace contains `input.csv` with columns `category,count,note`, where `count` is an integer string, and `settings.toml` with an integer `min_count` under `[report]`. Use this as the first `run_code` call's `code`:

```python
import csv
import tomllib

source = call_tool("read_file", {"relative_path": "input.csv"})
settings = tomllib.loads(
    call_tool("read_file", {"relative_path": "settings.toml"})["text"]
)
totals = {}
for row in csv.DictReader(source["text"].splitlines(keepends=True)):
    category = row["category"]
    totals[category] = totals.get(category, 0) + int(row["count"])

report = [
    {"category": category, "count": count}
    for category, count in totals.items()
    if count >= settings["report"]["min_count"]
]
report = sorted(report, key=lambda item: item["count"], reverse=True)
{"groups": len(report), "preview": report[:10]}
```

`keepends=True` preserves newlines inside quoted CSV fields. Do not replace CSV parsing with `split(",")` or line splitting that discards newlines. `DictReader` values are usually strings; convert them according to field meaning, and report invalid data or handle it under an explicit rule.

After checking the result, the same agent can call `run_code` again to save the in-memory `report`. Choose an output path within the authorized scope and check existing files before fully overwriting them:

```python
import json

saved = call_tool("write_file", {
    "relative_path": "out/report.json",
    "content": json.dumps(report, ensure_ascii=False, indent=2),
})
saved
```

There is no need to reread the CSV or send the full `report` back to the model and then pass it in again. Use `edit_file` for targeted changes and `write_file` to create or fully rewrite files.

## Persistent state, memory, and failures

Each agent has a separate subprocess retaining variables, functions, and imported modules directly in its execution environment. The environment remains while the agent is idle after responding. Closing the agent, `/clear`, or exiting the app releases the subprocess. Restoring conversation history does not restore variables.

Use `'report' in locals()` to check whether a variable exists. `locals()` returns a snapshot dictionary; changing it does not assign values in the real scope. Do not explore the environment with `globals()`, `dir()`, or `vars()`.

Splitting work into calls does not automatically reduce memory usage. Keeping raw text, split lines, parsed objects, and summaries retains all of them. After verification, rebind unneeded variables, such as `source = {}`, `rows = []`, or `raw_text = ""`, considering other references to the same objects. `clear()` empties a container in place and affects every alias; `pop()` removes a specific item.

**Execution is not a transaction.** Suppose `count = 1` already exists, and a call writes a file, assigns `count = 2`, then raises an ordinary exception. The next call still sees `count = 2`, and the file is already written. Inspect variables, `calls`, and actual artifacts before deciding what remains. If a call has not returned, its absence from `calls` does not prove that the remote operation did not complete.

Results must be JSON-encodable. A final dictionary with tuple or bytes keys can return `error_kind=serialize`, but earlier variable changes and tool operations have already taken effect. Log write failures do not roll back variables. In `call_parallel_tools`, arguments that cannot be JSON-encoded fail only that item; valid calls still run. An invalid overall list structure rejects the whole batch before execution.

Catch expected item failures, record them, and return completion counts and unfinished items. In-memory changes preceding ordinary exceptions remain. Report partial completion explicitly instead of counting omitted items as successes. Cancellation during execution, resource exhaustion, or worker failure discards the environment and returns `state_lost=true`. The next call creates an empty environment. Initialize data again from files or tools without automatically replaying code that may already have external effects. Cancelling a queued call does not affect the running call; cancelling after completion does not undo completed changes.

### Bounded retries: retry only reads that are safe to repeat

A failed `call_tool` raises `RuntimeError` with the tool name and failure details. Individual failures in `call_parallel_tools` appear in `success` / `error`. Retrying unchanged arguments does not fix argument errors, permission problems, or business rejections. A timed-out write may already have completed remotely; check its status first or use an idempotency mechanism explicitly provided by that service.

The following helper can remain in session state for MCP queries confirmed to have no side effects. Pass the actual activated tool name and arguments. It makes at most three attempts only for failures containing `TimeoutError`; other errors return to the caller immediately:

```python
import json
import time

def read_mcp_with_retry(tool_name: str, arguments: dict):
    for attempt in range(3):
        try:
            raw = call_tool(tool_name, arguments)
        except RuntimeError as error:
            if "TimeoutError" not in str(error) or attempt == 2:
                raise
            time.sleep(0.25 * (attempt + 1))
        else:
            return raw
```

Parsing failures do not trigger retries. Extract content according to the confirmed response structure. When accumulating a batch, catch final read failures and record failed items so remaining work can continue.

## Official 1.2.0 capabilities and limits

Prefer ordinary containers, functions, loops, comprehensions, `try/except`, and small helpers that perform pure computation. These are common capabilities, not a promise of full CPython compatibility.

| Capability | Usage and limits in Praxis |
| --- | --- |
| `json` | `loads` / `dumps` process text from tools; `dumps` supports `indent`, `sort_keys`, `ensure_ascii`, `allow_nan`, `separators`, and `skipkeys` |
| `csv` | `reader` / `DictReader` parse text lines; writers exist, but `io.StringIO` does not. No `Sniffer` or dialect registration; the fixed field limit is 131072 characters |
| `tomllib` | `loads` parses text without `parse_float`. Dates and times become corresponding objects; explicitly convert them to strings before writing JSON |
| `bisect` | Binary search and insertion in sorted lists |
| `match/case`, list slice assignment | Use for data branching and partial replacement without falling back to host Python |
| Simple classes and `dataclasses` | Supports `field(default=...)`, `field(default_factory=...)`, and `__post_init__`; factories and post-init functions cannot call external tools. No class inheritance, `slots`, `asdict`, or full dataclass support |
| `math`, `re`, `collections`, `datetime`, `itertools`, `hashlib`, etc. | Built-in standard-library subsets; consult official limitations for specific functions. This does not imply the full Python standard library is installed |
| Pure path operations in `os.path` / `pathlib` | POSIX semantics; Windows host drive-letter and backslash rules are not followed automatically |
| `open`, file-based `json.load/dump`, `tomllib.load`, walk/glob | Official interfaces exist, but Praxis configures no host filesystem mounts. Retrieve workspace files with file tools, then use text parsers |
| `external_modules` | Monty supports host module integration, but Praxis registers none. Do not `import tools`, project packages, or arbitrary host modules |

## Type checking and diagnostics

Praxis enables official type checking before execution and declares `call_tool(name: str, arguments: dict[str, Any]) -> Any`. The checker includes declarations from previous successful executions, but tool results are dynamic. Passing type checks does not guarantee fields exist, data is correct, or every API works. Do not bypass checking with `eval` / `exec`.

| Diagnostic | Response |
| --- | --- |
| `phase=preflight`, `error_kind=type_check` / `syntax` | This script did not execute. Correct its type or syntax; no tool-operation compensation is needed for this rejected script |
| `error_kind=tool` | Inspect the failed tool and `calls`, correct arguments, permissions, or service problems, and decide on retries based on tool semantics |
| `error_kind=runtime` | Inspect the exception and source location. Distinguish missing fields, type mismatches, and unsupported APIs; check for earlier successful tools |
| `error_kind=resource` | Reduce batch size or retained data, or choose a suitable tool. Do not repeat the same large computation unchanged |
| `queue`, `timeout`, `worker_crash`, or `phase=startup` | Inspect the phase and `state_lost`, verify external operations, and initialize from persisted inputs if the environment was lost |

Keep helper signatures stable when changing implementations. If diagnostics across calls still reference an old signature, use a new helper name and update callers rather than discarding the entire existing data session.

## Default budgets and work batches

| Praxis environment variable | Default | Meaning |
| --- | --- | --- |
| `PRAXIS_CODE_COMPUTE_SECONDS` | 5 seconds | Interpreter computation budget for one feed; time suspended for tools and sleep does not count |
| `PRAXIS_CODE_QUEUE_SECONDS` | 30 seconds | Budget for waiting for the previous code call in the same agent |

Praxis sets no Monty memory budget or global worker limit. Each agent that has used `run_code` and remains open owns a subprocess. Monty 1.2.0 cannot disable the suspension counter, so Praxis uses the maximum supported value, `2^64 - 1`. Raw text, parsed results, `stdout`, and host tool data still consume real memory.

Split large tasks into meaningful batches: process a batch, return progress and necessary samples, then continue in the same agent. Retain completed item IDs and necessary results to avoid duplicate reads or writes. Persist intermediate results with file tools when they must survive the process, then release large objects no longer needed.

## Sources and verification

Official references are pinned to v1.2.0 to avoid presenting unreleased main-branch features as supported:

- [v1.2.0 release notes](https://github.com/pydantic/monty/releases/tag/v1.2.0)
- [Standard-library modules and limitations](https://github.com/pydantic/monty/blob/v1.2.0/docs/limitations/modules.md), [language limitations](https://github.com/pydantic/monty/blob/v1.2.0/docs/limitations/language.md), [built-in differences](https://github.com/pydantic/monty/blob/v1.2.0/docs/limitations/builtins.md)
- [Host functions](https://github.com/pydantic/monty/blob/v1.2.0/docs/host-functions.md), [type checking](https://github.com/pydantic/monty/blob/v1.2.0/docs/type-checking.md)
- [Persistent execution environments](https://github.com/pydantic/monty/blob/v1.2.0/docs/quickstart/python.md#sessions-keep-state), [resource limits](https://github.com/pydantic/monty/blob/v1.2.0/docs/resource-limits.md)

Praxis integration details come from `src/praxis_agent/tools/code.py`, `tool_execution.py`, `tools/registry.py`, `context/tools.py`, and `mcp.py`. When upgrading or changing the bridge, check the dependency, these integration points, and this skill together. Do not just change the version number or copy every optional official capability.
