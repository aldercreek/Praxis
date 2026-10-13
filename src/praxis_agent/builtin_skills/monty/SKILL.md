---
name: monty
description: Monty Python compatibility, tool-result handling, reusable sessions and recovery for Praxis run_code. Consult for unfamiliar language features or execution failures.
---
## Choose the execution tool

Use `run_code` to compose permitted tools and process their structured results in a reusable session.
Use `run_command`, when available in your role, for full Python, installed packages, direct host file access, subprocesses and project tests. Otherwise report the missing capability; a reimplementation does not verify actual project execution.
Monty is not a substitute for the host Python interpreter. Use `write_file` to create or completely
rewrite a file through the bridge, and `edit_file` for local replacements.


# Monty in Praxis

Use `run_code` for tool-call loops, dependent operations, and processing intermediate data without sending every result to the model. Prefer ordinary tools for simple actions. Monty implements a Python subset, not a full CPython environment. This guidance targets Praxis's locally patched Monty runtime; its package version alone does not identify the patches.

## Sessions and results

Supply a meaningful `session_id`. Reuse it for related computation; a new name starts empty. Each Agent owns its sessions independently. Variables and functions survive successful calls and idle/resume, but not Agent closure or application restart. Only successful calls update saved state; inspect `state_saved` after errors. Previously completed external operations are never rolled back.

Check existing variables with `'data' in locals()`. `locals()` returns a snapshot dictionary; mutating that dictionary does not assign real variables. Use `type(value)` or `isinstance` to inspect values. Return a concise final expression; `print()` output is also collected in full. Large returned output goes through normal tool preview/history handling. Keep reusable intermediate values in the session and durable deliverables in files when permitted.

Each successful call serializes the whole retained session. Splitting work across calls does not reduce memory if all raw data stays live. Retain the data needed for later transformations or verification; release obsolete buffers after preserving useful results. `del name` removes that binding, not other aliases to the same object. Snapshot creation and restoration need memory in addition to live data. Another Agent cannot access these variables: deliver files or continue the follow-up in the producing Agent.

## Tool bridge

Call `call_tool(name, arguments)` with a registered tool name and a parameter dictionary matching its current schema. Use the tool definitions for parameters, coordinates and pagination; do not guess them.

- The bridge allows the current Agent's file tools, active MCP tools, and history search/read. Role restrictions and approvals still apply. Commands, management tools and nested `run_code` are outside this whitelist.
- Discover and activate MCP tools through ordinary model tool calls first. Use the registered callable name, not the service/tool identifier.
- File tools and history reads return structured objects. Read their text fields when applicable; do not JSON-decode an object. History search and MCP return JSON strings: decode the envelope once. An MCP text block is not necessarily JSON. JSONL contains separate JSON documents per line, not one JSON array. Confirm an unfamiliar response shape on a small sample and reuse that sample in subsequent work.
- Programmatic file/history reads return the requested range without model-preview truncation. Remote pagination and individual tool limits still apply. Reuse data already obtained instead of fetching it again unnecessarily.
- A failed tool raises `RuntimeError` naming the tool and including its failure message. Catch it when the script has a meaningful recovery path. Otherwise let it reach `run_code`; inspect completed effects before retrying.

Use ordinary loops for tool calls. Host calls cannot suspend inside callbacks to builtins such as `map`, `filter`, or `sorted(key=...)`. Fetch data first, then sort or transform it using pure computations. The current bridge executes tool calls synchronously; `asyncio.gather` does not make synchronous tool handlers concurrent.

## Important Python differences

- Lists, dictionaries, sets, functions, comprehensions, loops and ordinary `try/except` are useful building blocks. Prefer them to introspection or elaborate class machinery.
- `globals()`, `dir()` and `vars()` are unavailable. Builtin values do not expose `.__class__`; use `type`. Supported methods of lists, dictionaries, tuples, strings, bytes, `defaultdict` and `Counter` can be stored and passed as callbacks, such as `append = items.append` or `sorted(keys, key=mapping.get)`. These bindings survive session save/restore and retain their original receiver. Other receiver types and unbound type descriptors such as `str.lower` are not covered; use direct calls or a pure lambda when needed. Method introspection is limited.
- No `yield` generators. Generator expressions materialize lists, and several iterator-like builtins are eager. Do not assume streaming memory use or pass a generator expression directly to `next`; use a loop or an explicit iterator over a finite collection.
- Standard-library modules are bundled subsets. Common supported modules include `json`, `math`, `re`, `collections`, `datetime`, `itertools` and `hashlib`, but not every CPython API is present. `json.loads`/`dumps` work; the local runtime also supports synchronous, pure `json.dumps(default=...)` callbacks. These callbacks cannot call host tools. `json.load`/`dump` and `JSONDecoder` are unavailable.
- `csv`, `zlib`, `decimal` and third-party packages are unavailable. There is no pip environment inside Monty. Do not handwrite complex replacements for mature libraries merely to stay in this interpreter; use an available tool or, when your role permits, ordinary command execution with full Python.
- Pure `os.path` and `pathlib.Path` operations use POSIX-style paths, not host Windows path semantics. Praxis configures no host filesystem mounts: use file tools for actual workspace reads and writes rather than `open` or `Path.read_text`.
- The local runtime supports `del name` and nested tuple/list groups of names. Attribute/item deletion (`del obj.attr`, `del data[key]`) is not implemented; use supported container operations where appropriate.
- `time.sleep` is handled by Monty; waits do not count as interpreter compute time. Its native system-sleep policy caps a single wait at ten seconds. No direct sockets or subprocesses are exposed.

## Checks and errors

Praxis enables Monty's built-in type checker before each script executes, with a declaration for `call_tool`. A `type_check` or `syntax` failure means this submitted script did not execute. Fix the reported problem; no tool replay is needed for that rejected script. The checker retains declarations from successful session runs.

Known checker limitation: redefining a function with a different signature can leave a union of historical signatures when another function calls it. If diagnostics reference an obsolete signature, give the revised helper a distinct name and update its callers while keeping the session's existing data. Do not reset the whole session or bypass checking with `exec` to repair this.

Passing the check is not proof of full compatibility or correct data. Dynamic tool values use `Any`; some unsupported APIs and runtime data errors escape checking. Strings executed through `eval`/`exec` are not type checked. Do not use them to evade preflight.

Failures include `error_kind`, `error_type`, `phase` and diagnostics. Distinguish preflight failures, tool failures, Python runtime exceptions, resource limits, and session/worker errors. Runtime failures may follow successful tool calls; inspect the call records and existing artifacts. Correcting a script must not blindly repeat writes.

The default memory limit is 512 MiB (536870912 bytes), configurable through `PRAXIS_CODE_MEMORY_BYTES`; snapshot overhead also consumes memory. The default host-interaction budget is 5000 per execution checkout, configurable by Praxis. This counts suspensions, including tool calls and name resolution, not just successful MCP requests. Compute limits also apply. For work approaching these budgets, use manageable successful calls with the same session rather than assuming an arbitrarily long script will finish.

## Sources

Derived from Monty's [documentation at the fork base](https://github.com/pydantic/monty/tree/e74249051f6bb3df88fe8a0178e022272e14c1c6/docs), especially `type-checking.md`, `host-functions.md`, `snapshots.md`, `resource-limits.md` and `limitations/`. Local patches add the explicitly noted JSON callback, name-deletion and captured builtin-method support, reduce snapshot transport copying, and adjust cross-feed checking. Check the deployed runtime before adopting APIs from other revisions.
