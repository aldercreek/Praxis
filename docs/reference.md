# Praxis reference

This reference covers execution, tools, configuration, and extension points. Start with the [README](../README.md) for installation and everyday use.

## Service API

All endpoints are local HTTP, authenticated with `Authorization: Bearer <service.token>`. The service rejects browser-origin requests; native clients send requests through their own transport. `service.json` identifies the current instance and port. There is no public-network listener.

| Method | Path | Operation |
| --- | --- | --- |
| GET | `/api/health` | Check identity, protocol, readiness, and configuration |
| PUT | `/api/settings` | Change shared settings while no turns are active |
| GET / POST | `/api/sessions` | List / create sessions |
| DELETE | `/api/sessions/{id}` | Stop owned work, then permanently delete the session |
| GET | `/api/sessions/{id}/snapshot` | Read the current projection and its sequence |
| POST | `/api/sessions/{id}/inputs` | Submit `{command_id, text}` |
| POST | `/api/sessions/{id}/inputs/resume` | Resume pending input with a new command ID |
| DELETE | `/api/sessions/{id}/inputs/{command_id}` | Delete unconsumed input |
| POST | `/api/sessions/{id}/turns/{turn_id}/cancel` | Cancel that specific turn |
| POST | `/api/approvals/{approval_id}/decision` | Submit `{approved: true/false}` |
| GET | `/api/sessions/{id}/events?after=N` | Replay events after N, then continue streaming |
| GET | `/api/sessions/{id}/commands` | List the main agent's background commands |
| POST | `/api/sessions/{id}/commands/{command_id}/stop` | Stop a managed command |

Session creation accepts `{command_id, workspace, model?}` and returns a server-generated session ID. Repeated command IDs must refer to the same operation and content. The input's `command_id` also identifies it in pending-input actions; there is no separate public input ID.

All business events are persisted before streaming. Per-session sequences order events across clients and survive restarts. Snapshots include transcript messages, pending approvals, pending input, the active/latest turn, and the sequence of the projection. Initial clients buffer events while requesting a snapshot and discard buffered events at or below that sequence. Only an applied event advances the reconnect cursor.

The service sends an SSE heartbeat every 15 seconds; native clients reconnect after a 45-second read timeout. Replay and live data use the same ordered SQL reader per connection, with no transaction held while sending to a slow client. Events currently have no retention limit.

Deletion first persists a SQLite marker and closes admission under the session lock. Concurrent deletion requests wait for the same cleanup outside the lock. The response is `{deleted: true, session_id}` after cleanup; deleting an absent session also succeeds. Marked sessions disappear from lists and reject operations with HTTP 410, while removed sessions return 404. Existing SSE streams end with an unsequenced `event: deleted` transport control; clients also treat HTTP 404/410 on reconnect as terminal. Workspace files and the shared MCP manager are preserved. Startup loads deletion markers before accepting requests and finishes their cleanup asynchronously, without restoring Agents or replaying work. A failed cleanup keeps its marker for explicit retry or the next startup; there is no periodic cleanup scanner.

## Execution boundaries

Each session has one runtime and at most one main turn. Different sessions can execute concurrently. Viewing a session does not construct an agent or connect its model/tools.

Input received during execution is saved immediately, then consumed before the next model request. The final-response boundary checks input again. Input received after that check starts a successor turn only after successful completion. Cancellation and failure retain unconsumed input for an explicit resume, deletion, or new instruction.

Cancellation is a request followed by cleanup, not an immediate completion receipt. The service keeps the active turn until its work has settled. A cancellation for an older turn returns that turn's result without affecting its successor. Approval decisions are atomic: the first valid decision wins; retries return it, and cancelled approvals cannot start tools.

Startup recovery runs only after acquiring the process lock. It fails unfinished turns, closes pending approvals, and records unknown tool outcomes. A terminal result already committed before a crash is preserved. Recovery does not automatically repeat user input or tools with external side effects.

Closing a client does not close the runtime. A normal turn can leave background commands running for subsequent turns. Agent processes and live tool state are not durable across service restart.

## Built-in tools

| Area | Tools |
| --- | --- |
| Files and search | `read_file`, `list_files`, `find_files`, `search_text` |
| Editing | `edit_file`, `write_file` |
| Commands | `run_command`, `wait_command` |
| Planning | `update_plan` |
| Context | `delete`, `compress`, `search_history`, `read_history` |
| Skills | `load_skill` |
| Programmatic composition | `run_code` |

File reads accept workspace-relative or absolute paths, including files outside the workspace. File writes must stay inside the assigned workspace. `edit_file` requires each old-text match to be unique; replacements are applied in order and written only when all succeed.

`read_file` uses one-based lines and zero-based character offsets. Omitting an end position reads to the end; an end line without an offset includes that line, while an end offset is exclusive. Direct reads use a character budget and return `text`, `truncated`, and `next_position`. Continue with that cursor and the original end position. Programmatic reads can retrieve the full requested range.

Search uses ripgrep, includes hidden/ignored files unless explicitly filtered, and returns `{results, next_offset}`. `search_text` defaults to case-sensitive literal matching, with optional regex, case folding, and `content` / `files` / `count` modes. It does not support multiline matches or backreferences. Pages default to 50 results; context lines do not consume the match limit. Character budgets can shorten a page, and long lines use a marked preview. Keep the original filters when following a cursor. Search has a 30-second timeout.

Adjacent tools marked `parallel_safe` can execute concurrently. Other tools act as ordering barriers. Results enter model history in call order. `PRAXIS_MAX_PARALLEL_TOOLS` defaults to 4; set it to 1 for serial execution.

### Commands

`run_command` takes either an `argv` array or shell `command`, plus an optional existing `workdir`. Its `yield_time_ms` defaults to 10,000 and accepts 0–600,000. This limits the wait, not the process lifetime. A running command returns a command-session ID.

`wait_command(session_id, yield_time_ms=10000)` waits on that same process and returns newly available output. Use 0 for an immediate read. A completed agent turn does not stop background commands; subsequent turns can continue waiting on the same ID. Processes do not resume after service restart.

Output is spooled to temporary files with separate display and model cursors. There is no TTY or interactive stdin. The default limits are 16 MiB per command, 128 MiB per agent, and 32 retained commands. A shared maintenance loop checks limits and process exit. Completed results remain available for up to 15 minutes; successfully persisted final results release their temporary files.

Cancelling a turn stops commands created in that turn and commands it is currently waiting on. Unrelated commands from earlier turns remain available. Explicit stop, root-process exit, or agent shutdown cleans up managed descendants. Commands should run in the foreground under Praxis ownership.

## Subagents

The parent has `spawn_agent`, `send_messages`, `wait_agents`, and `close_agent`. Workers have `send_message(target, content)` and `wait_agents(timeout)`; the target is `parent` or an exact teammate ID. Workers do not receive spawning tools by default.

Each worker has its own history, plan, context, and Python state. It inherits approval configuration. The optional `spawn_agent.model` selects a configured model; otherwise workers use `subagents.model` or inherit the parent model. `spawn_agent` requires an existing working directory; Praxis does not automatically create worktrees, copy files, or merge changes. Coordinate file ownership and integration explicitly. Optional `inherit_context=true` copies the retained parent conversation, including existing summaries, as background before the new task. It defaults to false. Unfinished tool-call batches and parent-only developer instructions are excluded. Future inputs, running commands, Python variables and context-operation state are not shared.

Messages to idle workers start work; running workers receive them at execution boundaries. Closing/closed recipients reject new mail. Every agent sees a fresh, transient team snapshot on each model request: its identity, the parent coordinator, teammates' responsibilities, workspaces and status. Creating or closing a member or changing its responsibility can wake existing waits. Ordinary running/idle transitions do not broadcast new work. Idle agents are not started by membership changes.

`wait_agents(timeout=60)` waits for any mail to the caller, team membership/responsibility changes, parent user input, or a timeout of at most 60 seconds. It takes no agent-ID filter. Its result has `reason` (`message`, `team_changed`, `user_input`, `timeout`, or `closed`), `unread_messages`, `team_version` and `timed_out`. It never removes messages: the normal input boundary accepts them before acknowledging their exact IDs. Persistence failure retains unread mail. A wake is not proof that all workers finished; consult the current team and outcomes. Worker terminal reports go to the parent, while peer replies go only to their addressed recipient. Streaming display output does not wake waits.

A parent cannot finalize while workers are queued, running, or closing. `close_agent` cancels and releases the worker without deleting its files. `subagents.max_concurrent` in the configuration file defaults to 4; additional tasks wait in the worker pool. A waiting worker still occupies a pool thread: leave capacity for peers that must answer it. Workers and queued messages do not survive a process restart.

Service transcript snapshots retain each incoming worker message separately, with optional `sender_id` and `recipient_id` fields. Clients can label main-agent instructions and peer messages consistently during live execution and after reopening the session. These fields describe message provenance; they do not change the model's conversation history.

## MCP servers

Configure external tools in `<PRAXIS_DATA_DIR>/mcp.json` (default `~/.praxis/mcp.json`). Built-in tools work without it. Supported transports are local stdio and remote Streamable HTTP. The shared service loads this file at startup; restart the service after editing it.

Enabled servers connect in the background, up to eight at once. Each gets at most three attempts, waiting one then two seconds between failures; each attempt uses `startup_timeout_sec` (default 30 seconds). Disabled servers remain visible in status without starting a transport. Agents share connected clients, but keep their own activated tools and approval context. Closing an agent does not close these shared clients. Available service descriptions refresh before each model request.

`GET /api/mcp/servers` returns all configured server names and states through the authenticated local API. States are `disabled`, `queued`, `connecting`, `retrying`, `connected`, `failed`, and `disconnected`. Errors contain only their category, never server responses or credentials. Disconnections observed during calls update status; no tool call is automatically retried. Both clients display names and states with `/mcp`.

```json
{
  "servers": {
    "local-example": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"],
      "env_vars": ["EXAMPLE_API_KEY"],
      "startup_timeout_sec": 30,
      "tool_timeout_sec": 60
    },
    "remote-example": {
      "url": "https://example.com/mcp",
      "bearer_token_env_var": "EXAMPLE_MCP_TOKEN",
      "enabled": false
    }
  },
  "search": {"limit": 5}
}
```

Stdio servers support `cwd`, `env`, and forwarded `env_vars`; the config directory is the default working directory. HTTP supports `http_headers`, `env_http_headers`, and `bearer_token_env_var`. Timeouts must be positive and no greater than 3,600 seconds. Filter tools with `enabled_tools`, `disabled_tools`, or `tools.<name>.enabled`.

Tools are discovered before their full definitions enter model context. `search_tools` performs keyword/BM25 or name lookup, and `list_tools` browses a server. `activate_tools` loads definitions; supplying `arguments` also executes the tool. `deactivate_tools` removes definitions from the current agent. MCP tools require approval by default; `tools.<name>.approval_mode="approve"` allows a specific tool automatically.

Programmatic MCP calls return the full response dictionary. Direct calls adapt text, images, embedded PDFs, resource descriptions, and structured data for model input. Unsupported media does not discard other successful content. Resource links are described without automatic downloading. Timeouts and transport failures report an unknown completion outcome rather than retrying automatically.

Model configuration explicitly controls `capabilities.image_input` and `capabilities.pdf_input`; both default to false. PDF is supported by the Responses adapter only. Audio input is not supported.

## Skills

Skills are discovered from packaged `builtin_skills`, `~/.agents/skills`, and the workspace's `.agents/skills`, with later locations overriding earlier names. Each `<name>/SKILL.md` needs YAML frontmatter with a matching `name` and a `description`. Names are at most 64 characters; descriptions are at most 1,024 characters.

Only names and descriptions load initially. `load_skill(name, offset=0, version="")` reads the content in pages. Follow its cursor/version, restart if the file changes, and resolve referenced files using the returned resource directory. The bundled `python` skill documents programmatic tool composition.

## Persistent Python and programmatic tools

`run_code(code)` runs native CPython through `ipykernel` and `jupyter_client`. Each agent starts one kernel on first use and retains variables/functions across calls. Different agents have separate process memory; agents in the same session share a lazy virtual environment under `~/.praxis/python/<session-id>/`. The environment inherits Praxis's installed runtime libraries; use `sys.executable -m pip` to install additional session packages. No Docker, WSL, mounts or sandbox restrictions are involved. Project imports and absolute host paths work directly.

Praxis manages kernels directly. Jupyter/ZeroMQ carries code, results and streams; an independent authenticated loopback TCP channel carries tool requests and replies. Execution IDs reject callbacks after completion. Tool callbacks run in the original execution/cancellation context and use existing approvals and MCP connections. The loopback token is routing protection, not a security sandbox.

`call_tool(name, arguments)` calls supported file/history tools and activated MCP tools. Shell commands, management tools and recursive `run_code` remain unavailable through this function. `call_parallel_tools([{name, arguments}, ...])` runs explicit independent calls concurrently, applies normal validation/approval, and returns ordered `{success, output}` or `{success, error}` entries. One failure does not retry or cancel unrelated items.

Calls within one agent are serialized. Ordinary Python errors preserve partial variable/file changes. Cancellation, wall-time timeout or kernel death kills that kernel's process tree, cancels associated host tool/MCP operations and waits for their cleanup. Remote cancellation is best effort; already-completed effects are not undone. `state_lost=true` requires rebuilding variables; Praxis never automatically replays code. The next call starts a fresh kernel. Non-JSON final expressions return a bounded text representation; JSON results larger than 256,000 characters also fall back to a preview.

Idle kernels are reclaimed after 15 minutes. An entire active agent turn (including model thinking and tool waits) holds a lease, so gaps between code calls do not count as idle. Agent close or conversation reset terminates kernels. Dependency files remain on disk; variables do not survive reclamation or application restart. There are no pressure-based eviction, global concurrency budgets or CPU/memory quotas in this version.

Each stdout/stderr preview retains the first 6,000 and last 24,000 characters. Overflow logs under `~/.praxis/outputs/code/` save at most 10 million characters per stream; `output_files` reports `saved_chars`, `total_chars` and `complete`. Logs remain available for history references; there is no automatic aggregate disk-retention cleanup.

| Variable | Default | Meaning |
| --- | ---: | --- |
| `PRAXIS_CODE_EXECUTION_SECONDS` | 120 | Wall time per execution, including host/tool waits; startup excluded |
| `PRAXIS_CODE_IDLE_SECONDS` | 900 | Kernel idle lifetime outside active agent turns |
| `PRAXIS_CODE_QUEUE_SECONDS` | 30 | Wait for the same agent's previous code call |
| `PRAXIS_CODE_MAX_PARALLEL_TOOLS` | 16 | Concurrent explicit tool calls per batch |

## Context and history

Praxis keeps original conversation history, persisted context operations, and the temporary model-input view separate. Hiding or compressing a message does not remove its SQLite record.

Ordinary messages and tool results carry `[message_id=mN]`. `delete(message_ids)` replaces selected content with placeholders. `compress` takes either an explicit `summary` or summarizer `instructions`, but not both. Operations apply to the next request and replay on session restoration. Tool calls/results stay paired; protocol-specific, system/developer, and multimodal content cannot be targeted directly.

`search_history(query, max_results)` performs case-insensitive literal search, defaulting to 10 results and capped at 20, with queries up to 200 characters. `read_history(message_id, start_offset=0, end_offset=null)` reads a half-open character range and returns a continuation cursor. Offsets count Python characters, not bytes or tokens.

Before a model request, Praxis estimates instructions, messages, and tool definitions, then calibrates using observed input-token usage. If necessary, it compresses earlier history while keeping recent interactions and tool pairs. Oversized summaries are built in chronological chunks. If fixed content or the compressed result still cannot fit, the request fails instead of silently dropping content. There is no special retry path for provider-reported context overflow.

Model capacities come exclusively from the required YAML file. `context_window` and `max_output_tokens` are required; `max_input_tokens` is optional. The output limit is sent to the provider and bounds generation. Context budgets and compaction policies remain generic and can be overridden under `context`.

Compression uses `compression.model` when specified, otherwise the model of the agent being compressed. It uses its own model's input/output capacities for chronological chunking. See [configuration](configuration.md).
