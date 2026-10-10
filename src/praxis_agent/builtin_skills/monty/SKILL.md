---
name: monty
description: Write and run Monty Python through Praxis run_code. Read before first use, or when a script needs language-compatibility, tool-bridge, session or error-handling guidance.
---

# Monty in Praxis

Use `run_code` for tool-call loops, dependent operations, and processing intermediate data without sending every result to the model. Prefer ordinary tools for simple actions. Monty implements a Python subset, not a full CPython environment. This guidance targets Praxis's pinned `pydantic-monty==1.1.0`.

## Sessions and results

Supply a meaningful `session_id`. Reuse it for related computation; a new name starts empty. Each Agent owns its sessions independently. Variables and functions survive successful calls and idle/resume, but not Agent closure or application restart. Only successful calls update saved state; inspect `state_saved` after errors. Previously completed external operations are never rolled back.

Check existing variables with `'data' in locals()`. `locals()` returns a snapshot dictionary; mutating that dictionary does not assign real variables. Use `type(value)` or `isinstance` to inspect values. Return a concise final expression; `print()` output is also collected in full. Large returned output goes through normal tool preview/history handling. Keep reusable intermediate values in the session and durable deliverables in files when permitted.

## Tool bridge

Call `call_tool(name, arguments)` with a registered tool name and a parameter dictionary matching its current schema. Use the tool definitions for parameters, coordinates and pagination; do not guess them.

- The bridge allows the current Agent's file tools, active MCP tools, and history search/read. Role restrictions and approvals still apply. Commands, management tools and nested `run_code` are outside this whitelist.
- Discover and activate MCP tools through ordinary model tool calls first. Use the registered callable name, not the service/tool identifier.
- File tools and history reads return structured objects. Read their text fields when applicable; do not JSON-decode an object. History search and MCP return JSON strings: decode the envelope once. An MCP text block is not necessarily JSON.
- Programmatic file/history reads return the requested range without model-preview truncation. Remote pagination and individual tool limits still apply. Reuse data already obtained instead of fetching it again unnecessarily.
- A failed tool raises `RuntimeError` naming the tool and including its failure message. Catch it when the script has a meaningful recovery path. Otherwise let it reach `run_code`; inspect completed effects before retrying.

Use ordinary loops for tool calls. Host calls cannot suspend inside callbacks to builtins such as `map`, `filter`, or `sorted(key=...)`. Fetch data first, then sort or transform it using pure computations. The current bridge executes tool calls synchronously; `asyncio.gather` does not make synchronous tool handlers concurrent.

## Important Python differences

- Lists, dictionaries, sets, functions, comprehensions, loops and ordinary `try/except` are useful building blocks. Prefer them to introspection or elaborate class machinery.
- `globals()`, `dir()` and `vars()` are unavailable. Builtin values do not expose `.__class__`; use `type`. Builtin methods must be called directly: a stored `items.append` or `mapping.get` callback is not supported. A lambda calling the method can be used for pure transformations.
- No `yield` generators. Generator expressions materialize lists, and several iterator-like builtins are eager. Do not assume streaming memory use or pass a generator expression directly to `next`; use a loop or an explicit iterator over a finite collection.
- Standard-library modules are bundled subsets. Common supported modules include `json`, `math`, `re`, `collections`, `datetime` and `itertools`, but not every CPython API is present. `json.loads`/`dumps` work; `json.load`/`dump` and `JSONDecoder` are unavailable.
- `csv`, `zlib`, `decimal`, `hashlib` and third-party packages are unavailable in 1.1.0. There is no pip environment inside Monty. Do not handwrite complex replacements for mature libraries merely to stay in this interpreter; use an available tool or, when your role permits, ordinary command execution with full Python.
- `os.path` is unavailable. Monty's `pathlib.Path` provides POSIX-style path operations, not host Windows path semantics. Praxis configures no host filesystem mounts: use file tools for actual workspace reads and writes rather than `open` or `Path.read_text`.
- `time.sleep` is handled by Monty; waits do not count as interpreter compute time. Its native system-sleep policy caps a single wait at ten seconds. No direct sockets or subprocesses are exposed.

## Checks and errors

Praxis enables Monty's built-in type checker before each script executes, with a declaration for `call_tool`. A `type_check` or `syntax` failure means this submitted script did not execute. Fix the reported problem; no tool replay is needed for that rejected script. The checker retains declarations from successful session runs.

Passing the check is not proof of full compatibility or correct data. Dynamic tool values use `Any`; some unsupported APIs and runtime data errors escape checking. Strings executed through `eval`/`exec` are not type checked. Do not use them to evade preflight.

Failures include `error_kind`, `error_type`, `phase` and diagnostics. Distinguish preflight failures, tool failures, Python runtime exceptions, resource limits, and session/worker errors. Runtime failures may follow successful tool calls; inspect the call records and existing artifacts. Correcting a script must not blindly repeat writes.

The default host-interaction budget is 5000 per execution checkout, configurable by Praxis. This counts suspensions, including tool calls and name resolution, not just successful MCP requests. Memory and compute limits also apply. For work approaching these budgets, use manageable successful calls with the same session rather than assuming an arbitrarily long script will finish.

## Sources

Derived from the official [1.1.0 documentation](https://github.com/pydantic/monty/tree/v1.1.0/docs): [type checking](https://github.com/pydantic/monty/blob/v1.1.0/docs/type-checking.md), [host functions](https://github.com/pydantic/monty/blob/v1.1.0/docs/host-functions.md), [snapshots](https://github.com/pydantic/monty/blob/v1.1.0/docs/snapshots.md), [resource limits](https://github.com/pydantic/monty/blob/v1.1.0/docs/resource-limits.md), and [Python compatibility](https://github.com/pydantic/monty/tree/v1.1.0/docs/limitations). Check the installed version before adopting APIs from newer documentation.
