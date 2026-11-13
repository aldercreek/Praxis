---
name: python
description: Native CPython guide for Praxis run_code. Use for Python data processing, project imports, installed packages, persistent variables, file and MCP tool composition, or recovering from execution failures.
---

# Python in Praxis

Pass Python source to `run_code(code)`. Each agent lazily owns one persistent CPython kernel. Variables, functions and imports survive subsequent calls, including changes completed before ordinary exceptions. Different agents have separate memory and share the session's dependency environment. Calls within one agent are sequential.

Use normal Python libraries, project imports, subprocesses and native filesystem paths. The working directory initially matches the agent workspace. There is no container, mount mapping or security sandbox. Type annotations are not statically checked. Inspect `sys.executable` for this session's interpreter; install a needed package with `subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)` when authorized by the task. Restarted kernels retain installed packages but lose variables.

The final expression is the result. JSON-compatible values return structured data; other objects return a text representation. Large results are truncated: write large datasets to a file and return its path plus a summary. `print()` emits stdout; stderr is separate. Each stream preview keeps 30,000 characters, and overflow files save at most 10 million characters. Check `output_files.*.complete` before assuming a log is complete.

## Calling Praxis tools

`call_tool` and `call_parallel_tools` are already defined; do not import them. They use the current agent's existing tool registry, approvals and MCP connections.

```python
source = call_tool("read_file", {"relative_path": "input.csv"})
import csv, io
rows = list(csv.DictReader(io.StringIO(source["text"])))
totals = {}
for row in rows:
    totals[row["category"]] = totals.get(row["category"], 0) + float(row["amount"])
totals
```

`call_tool` waits for one result and raises `RuntimeError` on tool failure. Arguments must be JSON-compatible. File tools return structured values when available: `read_file` returns a dictionary with `text` and paging metadata, not a bare string. Honor continuation offsets for large files. Other tools may return plain text; inspect their schema/result before parsing.

Activate MCP tools through Praxis's `activate_tools` tool first. An MCP callback returns the complete envelope, including `content`, optional `structuredContent` and `isError`. Prefer `structuredContent` when present; do not assume every content block is text. Tool errors preserve the kernel, so inspect their cause before retrying operations with effects.

Configured Python extension tools use names such as `release__get_release_status`
and are already registered; they do not need MCP activation. Unless their author
disables programmatic calls, call them with JSON arguments through `call_tool`.
Praxis validates and constructs dataclass inputs on the service side after
approval. Object results return dictionaries; other JSON results return serialized
text. Extension dependencies belong to the service interpreter, not this kernel's
session environment. Their module globals are shared by service sessions.

```python
results = call_parallel_tools([
    {"name": "read_file", "arguments": {"relative_path": "a.json"}},
    {"name": "read_file", "arguments": {"relative_path": "b.json"}},
])
# Ordered results: {success: True, output: ...} or {success: False, error: ...}.
for item in results:
    if item["success"]:
        print(item["output"]["text"][:200])
```

Inside Python, `results` is already a list. Do not use `results["result"]`: the `result` wrapper belongs to the outer `run_code` tool response seen by the model, not to variables inside the kernel. Likewise `call_tool` returns the underlying tool value directly.

Use parallel calls for independent work. A failed item does not automatically stop the batch. This bridge cannot call shell tools, agent management tools or recursive `run_code`. Use Praxis's external command tools separately when needed, or Python's standard libraries for native Python work.

## Lifetime and recovery

- `worker_created=true`: a fresh kernel was started; initialize required variables.
- Ordinary Python exceptions preserve changes made before the exception. Syntax errors execute no code.
- Cancellation, wall-time timeout or kernel death discards this agent's kernel and descendants; `state_lost=true` indicates memory was lost. Associated host tools/MCP requests receive cancellation and are drained before completion. Remote effects may already have happened.
- The default execution deadline is 120 seconds, including tool waits. Kernel startup is excluded. Idle kernels are reclaimed after 15 minutes outside an active agent turn. Closing an agent, resetting the conversation or restarting Praxis also discards variables.
- Files and installed dependencies remain on disk. There are no snapshots, rollback or automatic code replay. Inspect existing effects before retrying.
- Do not leave tool calls in background Python threads after the code call completes. Late callback requests are rejected.
