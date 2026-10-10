# Praxis

Praxis is a small coding-agent project used to learn Python by building a real application step by step.

## Setup

Activate the project virtual environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project and its development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Praxis reads the existing ADA API key directly. If it is already configured in
Windows, no additional setup is needed. To set it for the current PowerShell
session:

```powershell
$env:ADA_API_KEY = "your-coding-plan-token"
```

Praxis uses the ADA gateway and `gpt-5.6-terra-2026-07-09` by default for both the main and summary models. These
optional variables can override either value:

```powershell
$env:ADA_MODEL = "gpt-5.6-sol-2026-07-09"
$env:ADA_BASE_URL = "https://ada-cli-go.ctripcorp.com/coding-plan/openai/v1"
```

Praxis estimates the complete model request before sending it. Model limits
are resolved from explicit environment overrides first, then the bundled
catalog, and finally a conservative 128,000-token fallback. The catalog covers
current coding-oriented OpenAI, Claude, Gemini, DeepSeek, and Qwen models,
including their dated snapshot names. Unset override fields still inherit from
the catalog or fallback. By default, Praxis uses 95% of the selected window and
selects output, summary and read budgets from six tiers. Override the model limits and context policy
when needed:

```powershell
$env:ADA_CONTEXT_WINDOW_TOKENS = "128000"
$env:ADA_MAX_INPUT_TOKENS = "120000"
$env:ADA_MAX_OUTPUT_TOKENS = "16000"
$env:ADA_DESIRED_OUTPUT_TOKENS = "8000"
$env:ADA_CONTEXT_SAFETY_RATIO = "0.95"
```

Defaults are centralized in `context/budgets.py`. Windows and token counts below
use decimal thousands (128k = 128,000); each row includes its upper boundary.

| Context window | Output reserve | Summary target / generation cap | Tool preview / file chars | File lines | History chars |
| --- | ---: | ---: | ---: | ---: | ---: |
| ≤32k | 4k | 2k / 4k | 16,000 | 400 | 8,000 |
| ≤64k | 8k | 3k / 6k | 24,000 | 600 | 12,000 |
| ≤128k | 16k | 4k / 8k | 32,000 | 800 | 16,000 |
| ≤256k | 24k | 6k / 12k | 48,000 | 1,200 | 24,000 |
| ≤512k | 32k | 8k / 16k | 64,000 | 1,600 | 32,000 |
| >512k | 32k | 12k / 24k | 96,000 | 2,400 | 48,000 |

Explicit overrides take precedence: `ADA_DESIRED_OUTPUT_TOKENS`,
`ADA_SUMMARY_TARGET_TOKENS`, `ADA_SUMMARY_MAX_OUTPUT_TOKENS`,
`ADA_MAX_TOOL_OUTPUT_CHARS`, `ADA_MAX_FILE_READ_CHARS`,
`ADA_MAX_FILE_READ_LINES`, and `ADA_MAX_HISTORY_READ_CHARS`.
Model input/output limits still apply. The main output reserve only affects
input budgeting; it does not set the main request's generation cap. Summary
targets guide the prompt, while the generation cap sets `max_output_tokens`.
File reads still default to 200 lines and history reads to 8,000 characters;
larger reads must be requested explicitly. Standalone tools without model
configuration use the smallest tier. Search limits and timeouts are unchanged.

The first model request uses a heuristic token estimate. Later requests use
the previous response's exact `input_tokens` plus the estimated change in the
model context, avoiding a full uncalibrated estimate on every request.

Read-only tools (`read_file`, `list_files`, `find_files`, `search_text`,
`search_history`, `read_history`) have an internal `parallel_safe` flag.
Consecutive eligible calls run in bounded groups, with at most four handlers
running simultaneously. Set `PRAXIS_MAX_PARALLEL_TOOLS` to a positive integer
to override this (1 disables simultaneous execution). Other tools are serial
barriers; no calls move across them. The next group starts only after the
previous group's results have been saved in their original call order.
Parsing, validation, approval, events, state updates and SQLite writes remain
on the main thread. Only eligible handlers run in worker threads. Terminal
completion events may arrive out of order, but history and model results
retain the original order and `call_id`. The flag is not a model parameter.
The prompt encourages batching independent known-argument calls and waiting
for results before making dependent calls.

## Context management

Praxis keeps three separate layers: complete conversation history, persisted
context operations, and the temporary input view sent to the model. Original
conversation messages in SQLite are never replaced by hidden-output markers.

User/assistant text and tool outputs carry textual labels such as
`[message_id=m12]` in the model view. IDs come from the original history order,
not the current view position, and are not extra API protocol fields.

- `delete(message_ids)` replaces selected messages with one placeholder.
- `compress(message_ids, instructions)` asks a separate model request, without
  tools, to summarize the selected content using the model's preservation
  instructions. The summary is generated once and saved with the operation.

Tools only generate and persist operations. Before each model request,
`build_view()` rebuilds the view from complete history and applies all operations
in order. Tools select messages only from
the last built view; compression never reads hidden original messages. Changes
become visible on the next request. Selecting the same message again after it
was changed in the current turn returns a tool error.

Delete and compress each return an ID such as `op1` and one new `message_id`.
The operation stores the selected `message_ids` and a single `replacement`
message containing its ID and content. This message replaces all selected
messages at the position of the last target; unselected messages remain.
Placeholders and summaries use ordinary message IDs saved with the operation
and reused on recovery. Summaries, placeholders, and context-management tool
results can be deleted or compressed again as ordinary text messages.
Use `search_history(query, max_results)` to find hidden details by literal,
case-insensitive text matching. It returns up to 20 matches with `message_id`,
`offset`, and `text`, plus a `truncated` flag. The offset is the start of the
returned snippet, including up to 80 characters around the match. Search checks
saved replacement messages first, then original history, newest first within
each group. The query may contain up to 200 characters.

Use `read_history(message_id, offset, length)` to read more of a message. Offsets
start at zero and count Python string characters, not bytes or tokens. Reading
returns `text`, `next_offset`, `total_length`, and `eof`; reuse `next_offset` to
continue. Each call reads up to 8,000 characters. Search and reading use the same
text: string bodies are unchanged, and text blocks are concatenated in order.
Images and protocol-only items are not searched. Retrieval adds ordinary tool
results to the conversation without modifying existing context operations.

When a tool result is replaced, its matching call is also omitted from the model
view so no unpaired call remains. Both remain stored in complete history.
Protocol-only items, system/developer
messages and multimodal messages are protected.
Invalid targets are checked before persistence.
Tools do not preview candidate views, compare token savings, or calculate summary
budgets. Capacity is estimated and checked only when building the model request;
a summary may be longer and cause the next request to exceed the limit.
SQLite stores operations separately from original messages and
replays them on recovery. `/clear` starts a new session with no operations.
Legacy call-ID operations are not migrated or loaded.

Large user inputs remain ordinary messages: no automatic attachment conversion,
deletion, or summarization. The model decides when to manage context. A request
already exceeding the estimated input limit is rejected before calling the model;
the user must shorten the input or start a new session.

After context changes, token accounting starts with a fresh estimate and
is calibrated again using the next model response's actual input usage.

At startup, Praxis tells the model only the operating system, resolved
workspace root, and which common shells are available or unavailable. It does
not send environment-variable values or the full system `PATH`.

## Langfuse observability

Praxis enables Langfuse automatically when both project keys are present. The
default endpoint is Ctrip's production Langfuse instance:

```powershell
$env:LANGFUSE_PUBLIC_KEY = "your-project-public-key"
$env:LANGFUSE_SECRET_KEY = "your-project-secret-key"
```

The endpoint and trace environment can be overridden when needed:

```powershell
$env:LANGFUSE_BASE_URL = "https://langfuse.flight.ctripcorp.com"
$env:LANGFUSE_TRACING_ENVIRONMENT = "development"
```

Each interactive CLI process is recorded as one Langfuse session. Each task is
one trace containing its model requests and tool calls. Praxis still runs when
Langfuse is not configured or temporarily unavailable, and flushes pending
observations before the CLI exits. Large inputs and outputs are truncated
before upload.

On Windows, Praxis uses the system certificate store for Langfuse connections
so that internal certificate authorities trusted by Windows are also trusted by
the Python client.

## Run

Pass a task directly. Praxis can inspect and modify files under the current
directory through its registered tools. Tool calls are shown in the terminal.
Commands require confirmation before they run:

```powershell
praxis "inspect this repository"
```

For example, Praxis can run tests after changing code:

```powershell
praxis "run the tests, analyze any failures, and fix them"
```

When `run_command` is requested, review the displayed argument list and enter
`y` to allow it. Any other answer declines the command.

Use `--bypass` to skip tool approval prompts for the current process:

```powershell
praxis --bypass
praxis --bypass "run the tests and fix failures"
praxis --bypass --continue
```

This option applies to all tools requiring approval. It is not saved in the
session; subsequent launches without `--bypass` use normal approval prompts.
Tool argument validation and workspace path restrictions still apply.

Start it without an argument to keep a conversation in the current process:

```powershell
praxis
```

Praxis persists sessions in `~/.praxis/state.sqlite3`. Start a new process and
continue the latest session for the current workspace with:

```powershell
praxis --continue
```

List the 20 most recently updated sessions for the current workspace without
requiring model credentials:

```powershell
praxis --sessions
```

Resume a known session explicitly with its full ID or the unique prefix shown
by `--sessions`:

```powershell
praxis --resume a3f21c8d
```

In interactive mode, each new task includes the earlier conversation and tool
results. Use `/clear` to preserve the old session and switch to a new one, or
`/exit` to leave:

```text
praxis> inspect config.py
praxis> explain the previous result in more detail
praxis> /clear
Conversation cleared.
praxis> /exit
```

Sessions contain individual task runs and their ordered model conversation
items. If a process stops during a run, resuming marks that run failed and adds
an interrupted result for any tool call whose completion was not recorded.
Praxis never reruns an interrupted tool automatically.

For non-trivial work, Praxis can maintain a short task plan through the
`update_plan` tool. Before each model request, Praxis adds a temporary snapshot
of the current plan and the complete, sorted list of files successfully changed
by file tools. The snapshot is rebuilt after tool calls and is not saved in the
conversation history. Successful plan updates are displayed as a checklist in
the terminal. Each task also exposes its latest in-memory `RunState`,
including the original request and final `running`, `completed`, or `failed`
status.

Context management normally remains model-driven through `delete` and `compress`.
As a display safeguard, tool results show a prefix according to the selected
tier (`ContextPolicy.max_tool_output_chars`). The limit counts decoded tool
text, excluding the file line header (up to 512 characters);
JSON escaping does not prematurely truncate a bounded file read. Continuation
offsets still refer to the saved original serialized result.
If the full request still exceeds the
estimated input budget, Praxis reserves `ContextPolicy.summary_target_tokens`
(4,000 tokens at 128k), keeps the recent interactions that fit, and summarizes
the older prefix. Tool calls and their results stay together at the cut point.
Previews include the original message ID, character offsets, and instructions
for retrieving more text with `search_history` or `read_history`.

These display limits do not create context operations or alter the saved tool
results. Command stdout/stderr are saved without the previous output truncation.
File/search tools still honor their requested ranges and result limits; history
retrieval can only recover text the tool actually returned. View safeguards are
recomputed when a session is resumed. Summarization processes the selected history
in one request when it fits. Otherwise, it updates one running checkpoint with
successive chronological chunks. Each request budgets for the actual previous
summary, instructions, and output reserve; only the final checkpoint is retained.
Summary requests have an output token cap; complete-response validation remains
a known limitation. After the final request fits, one
`compress` operation is saved, so resume preserves its effect. Ordinary requests
do not automatically summarize history. Fixed context or recent interactions
that cannot fit cause a clear error rather than silent removal. This initial
implementation handles locally detected overflow; provider-side overflow retry
is not yet implemented. Emergency compaction shows terminal progress, estimated
tokens before/after, duration, and errors. Langfuse records a `context-compaction`
span with a generation per summary/merge chunk, including actual token usage
when returned by the provider. These events contain metrics, not history bodies.

## Test

```powershell
python -m pytest
```

## Structure

```text
src/praxis_agent/  Application package
  context/         Complete history, model views, and capacity awareness
  sessions/        SQLite session persistence and recovery
tests/             Automated tests
pyproject.toml     Project metadata, dependencies, and CLI entry point
```
