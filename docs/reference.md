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

The parent has `spawn_agent`, `send_messages`, `wait_agents`, and `close_agent`. Workers use `send_message` to report to the parent and do not receive spawning tools by default.

Each worker has its own history, plan, context, and Monty state. It inherits model and approval configuration. `spawn_agent` requires an existing working directory; Praxis does not automatically create worktrees, copy files, or merge changes. Coordinate file ownership and integration explicitly.

Messages to idle workers start work; running workers receive them at execution boundaries. `wait_agents` waits for new messages, completion, or a timeout of at most 60 seconds. A parent cannot finalize while workers are queued, running, or closing. `close_agent` cancels and releases the worker without deleting its files. `PRAXIS_AGENT_MAX_WORKERS` defaults to 8; additional tasks wait in the worker pool.

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

The model catalog controls image/PDF input capabilities. Unknown models disable these by default; override with `ADA_MODEL_IMAGE_INPUT` and `ADA_MODEL_PDF_INPUT` only when your provider supports them. Audio is not sent through the current Responses integration.

## Skills

Skills are discovered from packaged `builtin_skills`, `~/.agents/skills`, and the workspace's `.agents/skills`, with later locations overriding earlier names. Each `<name>/SKILL.md` needs YAML frontmatter with a matching `name` and a `description`. Names are at most 64 characters; descriptions are at most 1,024 characters.

Only names and descriptions load initially. `load_skill(name, offset=0, version="")` reads the content in pages. Follow its cursor/version, restart if the file changes, and resolve referenced files using the returned resource directory. The bundled `monty` skill documents programmatic tool composition.

## Programmatic tools with Monty

Praxis uses the official `pydantic-monty==1.2.0` package. Each agent lazily starts its own persistent worker; variables and functions are reused within that process. There are no saved worker snapshots or shared process-pool slots.

`run_code(code)` can use `call_tool` for supported file/history tools and activated MCP tools. It cannot invoke shell commands, management tools, or nested `run_code`. Use `run_command` for project imports, third-party libraries, and tests. Praxis does not mount a filesystem or configure Monty `external_modules`; built-in `json`, `csv`, and `tomllib` can parse tool results.

`call_parallel_tools([{name, arguments}, ...])` runs explicit independent calls concurrently, with permissions, validation, and approval still applied. The default concurrency is 16; results preserve input order and contain either `{success: true, output}` or `{success: false, error}`. One failure does not retry or cancel unrelated items. Cancellation stops dispatch and waits for started work to clean up; external side effects are not rolled back.

Ordinary execution or serialization failures do not roll back interpreter variables. Cancellation, resource exhaustion, or worker failure releases that worker and returns `state_lost=true`; the next call starts fresh without replaying code. Praxis sets no Monty memory budget and uses Monty's maximum supported suspension counter.

Each stdout/stderr preview retains at most 30,000 Python characters: the first 6,000 and last 24,000. Longer output is saved as UTF-8 under `~/.praxis/outputs/code/`; `output_files` reports the path and completeness. Files remain available for history references after a session closes.

| Variable | Default | Meaning |
| --- | ---: | --- |
| `PRAXIS_CODE_COMPUTE_SECONDS` | 5 | Compute time per interpreter advance, excluding tool waits |
| `PRAXIS_CODE_QUEUE_SECONDS` | 30 | Wait for the same agent's previous code call |
| `PRAXIS_CODE_MAX_PARALLEL_TOOLS` | 16 | Concurrent explicit tool calls per batch |

## Context and history

Praxis keeps original conversation history, persisted context operations, and the temporary model-input view separate. Hiding or compressing a message does not remove its SQLite record.

Ordinary messages and tool results carry `[message_id=mN]`. `delete(message_ids)` replaces selected content with placeholders. `compress` takes either an explicit `summary` or summarizer `instructions`, but not both. Operations apply to the next request and replay on session restoration. Tool calls/results stay paired; protocol-specific, system/developer, and multimodal content cannot be targeted directly.

`search_history(query, max_results)` performs case-insensitive literal search, defaulting to 10 results and capped at 20, with queries up to 200 characters. `read_history(message_id, start_offset=0, end_offset=null)` reads a half-open character range and returns a continuation cursor. Offsets count Python characters, not bytes or tokens.

Before a model request, Praxis estimates instructions, messages, and tool definitions, then calibrates using observed input-token usage. If necessary, it compresses earlier history while keeping recent interactions and tool pairs. Oversized summaries are built in chronological chunks. If fixed content or the compressed result still cannot fit, the request fails instead of silently dropping content. There is no special retry path for provider-reported context overflow.

| Variable | Behavior |
| --- | --- |
| `ADA_CONTEXT_WINDOW_TOKENS` | Override the model context window |
| `ADA_MAX_INPUT_TOKENS` | Override the input limit |
| `ADA_MAX_OUTPUT_TOKENS` | Output-capacity metadata used to constrain the reserve |
| `ADA_CONTEXT_SAFETY_RATIO` | Input safety ratio, default 0.95 |
| `ADA_DESIRED_OUTPUT_TOKENS` | Reserved output capacity, not an API generation limit |
| `ADA_MAX_TOOL_OUTPUT_CHARS` | Tool-result preview budget for model context |
| `ADA_MAX_FILE_READ_CHARS` | Direct file-read character budget |
| `ADA_MAX_HISTORY_READ_CHARS` | Direct history-read character budget |

Explicit overrides take precedence over the [model catalog](../src/praxis_agent/context/model_catalog.py). Unknown models fall back to a 128,000-token window. Tiered output reserves and character budgets are defined in [context/budgets.py](../src/praxis_agent/context/budgets.py). There is no separate summary-target-length or summary-output-limit setting.
