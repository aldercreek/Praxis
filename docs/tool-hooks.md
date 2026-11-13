# Tool hooks

Hooks are ordinary synchronous Python functions loaded from explicit local files.
They can inspect tool context, replace arguments, deny execution, require approval,
and replace the final result. They work across built-in tools, registered MCP tools,
Python extensions, workers, and `run_code` calls to other tools. No plugin package is needed.

## Configure

```yaml
hook_extensions:
  - id: release_policy
    path: ./extensions/release_hooks.py
    enabled: true
```

Paths are relative to the configuration file. Disabled entries are not imported.
Hooks load once at service startup; code/configuration changes require a service
restart. An ordinary settings reload rejects hook configuration changes. A file
may contain both tools and hooks and be enabled in both configuration sections;
its module is imported once. Importing executes Python with service permissions.

## Write hooks

```python
import json
from praxis_agent.sdk import hook, HookDecision, HookOutput

@hook("before_tool_call", tools=["release__*"])
def normalize(event):
    arguments = dict(event.arguments)
    if arguments.get("environment") == "production":
        arguments["environment"] = "prod"
    return HookDecision.allow(
        arguments=arguments,
        requires_approval=arguments.get("environment") == "prod",
    )

@hook("after_tool_call", tools=["release__*"])
def redact(event):
    data = dict(event.result.structured_output or {})
    data.pop("access_token", None)
    return HookOutput(json.dumps(data), structured_output=data)

@hook("on_tool_error", tools=["release__*"])
def explain(event):
    return HookOutput("Release query failed. Inspect the service configuration before retrying.")
```

`tools` is a list of case-sensitive shell-style patterns against the registered
tool name. Omitting it matches all tools. Hooks run in configuration order, then
function definition order. A function alias does not register twice; imported
functions are not registered. Use separate functions for different events.

## Execution flow

1. Parse JSON and confirm tool availability.
2. Run `before_tool_call` hooks in order. Each sees previous replacements.
3. Validate the final arguments using the tool's normal validation.
4. Approve those same final arguments, when required.
5. Execute the tool (including Pydantic object construction for Python tools).
6. Run `after_tool_call` on success or `on_tool_error` on execution failure.
7. Publish and return the processed result.

A `before_tool_call` hook returns `None`, `HookDecision.allow(...)`, or
`HookDecision.deny(reason)`. Replacement arguments are a complete JSON object,
not a patch. The original arguments remain available to later hooks. A denial
stops the chain. Invalid replacement arguments never reach execution or approval.
Allowing does not remove a tool's approval requirement. An explicit
`requires_approval=True` survives normal approval bypass and cannot be undone by
a later hook. Rejected approval, denied calls and invalid inputs do not trigger
execution-result hooks.

Result hooks return `None` to retain the current result or `HookOutput` to replace
its model-visible content. `HookOutput` replaces **all** content channels: text,
structured output, and model content. Omitted channels and the original display
diff are cleared to prevent stale content from bypassing redaction. The actual
success/failure status and persistence acknowledgement remain unchanged. The
runtime still tracks the file actually changed, using the final arguments.

A callback receives a detached snapshot. Mutating the event has no effect on the
actual arguments/results: return an explicit decision or output instead.

## Context available to hooks

`HookEvent` includes:

- `event`, `tool_name`, `description`, `parameters` (the tool's input schema).
- `requires_approval` (tool or hook requirement), `approval_requested` (explicitly added by a hook).
- `session_id`, `agent_id`, `turn_id`, `call_id`, `parent_call_id`.
- `workspace`, `programmatic_call`, `original_arguments`, `arguments`.
- For result hooks: `original_result`, `result`, and `duration_ms`.

`HookResult` exposes `success`, `output`, `structured_output`, and `model_content`.
Duration excludes approval and hook time. Session IDs may be absent for unpersisted
SDK agents; workers receive the parent session ID and their own agent ID. Nested
tool calls have separate call IDs linked by `parent_call_id`.

Hooks are shared across sessions/workers. Parallel tool results may run hooks
concurrently; keep mutable state local to the event or synchronize it explicitly.

## Errors and output handling

A failing before hook blocks execution. Errors identify the hook and exception
class without echoing arbitrary callback exception payloads.

A failing result hook stops processing and substitutes a clear processing-error
message; it never falls back to the original output. The tool's actual execution
status is retained, and it must not be rerun automatically. No failure hook is
recursively dispatched for an error inside another hook.

Commands with matching result hooks do not stream raw output to the client; they
return the final processed output. Hooks transform the tool response, not external
side effects, files created by tools, model-generated input messages, or logging
performed directly by extension code.

## Offline example

```powershell
.venv/Scripts/python.exe examples/tool_hooks/verify.py
```

This runs a simulated query, converts `production` to `prod`, requests demo approval,
and redacts a token before returning the result. It does not call a model or deploy.


## Client visibility

Use `/hooks` in the terminal or **Settings → Tools & hooks** in Praxis-UI to
inspect the connected service's loaded hooks, source files, event filters and
execution order. `/tools` lists Python tools. These are read-only catalogs;
restart the Praxis service after editing extension configuration or code.
Reconnecting a client does not reload extensions.

Tool details show when hooks modify arguments or process output. Approval cards show
only the final arguments and approval choices. Hooks can still require approval
when normal approvals are bypassed. Hook blocks and result-processing errors remain visible
in collapsed tool activity. A processing error is distinct from execution
failure; clients never offer the unfiltered original result. The same metadata
is retained when reopening a session.
