# Python tool extensions

Write an independent Python file and enable it in `praxis.yaml` to add named tools.
Users edit their own files and configuration; they do not modify Praxis source.
The SDK is included in `praxis-agent`, with no separate SDK package to install.

## Try the offline example

From the Praxis repository, using the same interpreter as the service:

```powershell
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/python.exe examples/tool_extensions/verify.py
```

The verifier loads `examples/tool_extensions/praxis.yaml`, prints the registered
tool names, and simulates calls through the normal executor, including an explicit
demo approval. It does not call a model or any external API. Its model endpoint is
a placeholder; use your actual model configuration for normal agent sessions.

## Write a tool

```python
from dataclasses import dataclass
from praxis_agent.sdk import tool

@tool(description="Greet a person", requires_approval=False)
def greet(name: str, *, prefix: str = "Hello") -> str:
    return f"{prefix}, {name}!"

@dataclass
class ReleaseQuery:
    release_id: str
    include_logs: bool = False

@tool(description="Return a simulated release status")
def get_release_status(query: ReleaseQuery) -> dict:
    return {"release_id": query.release_id, "status": "success"}
```

`import praxis_agent.sdk as Praxis` followed by `@Praxis.tool(...)` also works.
The decorator records metadata and returns the original function. Helpers and
functions imported from other modules are not registered. An alias to the same
decorated function does not register it twice.

| Decorator argument | Default | Meaning |
| --- | --- | --- |
| `description` | Required | Nonempty tool description shown to the model |
| `name` | Function name | Tool name before the extension prefix |
| `requires_approval` | `True` | Use normal Praxis approval and bypass policy |
| `parallel_safe` | `False` | Allow normal parallel scheduling |
| `programmatic` | `True` | Allow existing Python `call_tool` / `call_parallel_tools` |

## Configure and restart

Add this to your existing model configuration:

```yaml
tool_extensions:
  - id: release
    path: ./extensions/release_tools.py
    enabled: true
```

Paths are relative to **the configuration file's directory**, not the workspace
or shell directory. Absolute paths and `~` are supported. IDs must be unique,
start with an ASCII letter, and contain only ASCII letters, digits, `_` and `-`,
with a maximum length of 32. Tool names follow the same character rule; the final
`<id>__<name>` must be no more than 64 characters. Collisions fail rather than
overwriting tools. `tool_extensions` defaults to an empty list.

Restart the **Praxis service process**, then reconnect your clients. Closing a
terminal or desktop client alone does not stop the shared service. The service's
PID is recorded in `<PRAXIS_DATA_DIR>/service.json`; stop running work and shut
down that service before launching it again with the updated configuration.
For foreground development, use
`python -m praxis_agent.service --data-dir ./local-state --config /absolute/path/praxis.yaml`,
stop with Ctrl+C, and relaunch.

Enabled extensions are imported at service startup, before sessions are created.
Disabled entries are not imported (their files need not exist). Invalid enabled
entries abort loading with an error. Nothing scans or executes workspace Python
files automatically. This version supports single files plus installed packages,
without changing `sys.path`; package-relative imports and automatic dependency
installation are not provided.

The model sees `release__greet` and `release__get_release_status` directly, without
MCP search or activation. For the latter it can send:

```json
{"query": {"release_id": "123", "include_logs": true}}
```

The full nested Schema is checked before approval. After approval, Praxis builds
`ReleaseQuery` and invokes the function. You can also ask the agent to use it from
`run_code`:

```python
status = call_tool("release__get_release_status", {"query": {"release_id": "123"}})
status["status"]
```

## Types and defaults

Input schemas, validation, object construction and output serialization use
**Pydantic v2**. Standard dataclasses remain supported; using `BaseModel` is optional.
Every input needs a resolvable annotation; string and postponed annotations work.
Return annotations are optional and do not constrain results.

Supported JSON-compatible annotations include primitives, typed or bare containers,
`Optional`, general unions, `Literal`, `Enum`, `datetime`/`date`, `UUID`, `Path`,
sets, tuples, `TypedDict`, nested/recursive dataclasses and Pydantic models.
`Annotated` and Pydantic `Field` provide descriptions and constraints. Custom types
can use Pydantic validators, serializers and schema hooks; they must define a
usable JSON Schema and JSON validation. Arbitrary Python objects are not enabled
as a fallback. Model providers may impose additional limits on JSON Schema support.

For example:

```python
from datetime import datetime
from uuid import UUID
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field
from praxis_agent.sdk import tool

class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    release_id: UUID
    since: datetime | None = None
    limit: Annotated[int, Field(ge=1, le=100)] = 10

@tool(description="Query releases")
def query_releases(query: Query):
    # UUID and datetime are already Python objects here.
    return query
```

`Optional[T]` allows null but does not make a parameter optional: only a Python
default makes it omissible. Omitted function arguments are left to Python.
Dataclass/model defaults and factories are handled by Pydantic after approval;
factories are not evaluated during loading. Dataclass `init=False` fields are
excluded from input schemas. Unknown top-level arguments and, by default, unknown
dataclass fields are rejected. Explicit Pydantic model/dataclass configuration
controls its own extra-field policy.

Before approval, the existing executor checks the generated JSON Schema without
running constructors or input validators. After approval, Pydantic validates in
strict **JSON mode** and constructs objects. Thus strings are not automatically
coerced into integers or booleans, while dates/UUIDs/paths accept JSON strings and
sets/tuples accept JSON arrays. Custom validators can define their own conversions.
Format checks and custom validation may fail after approval; constructors or
validators may have run, but the tool function is not called if validation fails.
Custom schema hooks execute during loading, just like module-level Python code.

Only ordinary synchronous functions with named or keyword-only parameters are
supported. Positional-only parameters, `*args`, `**kwargs` and async/generator
functions are rejected.

Return strings for text or Pydantic-serializable values for data, including
standard dataclasses and Pydantic models. Serialization uses Pydantic's native
JSON rules: dates/UUIDs become strings, tuples/sets become arrays, and supported
non-string dictionary keys become strings. Use string keys to avoid collisions.
Model serializers are honored. Cycles, unsupported objects and non-finite JSON
numbers fail explicitly.

Dictionary results populate `ToolResult.structured_output`; other JSON values are
serialized to text, preserving the existing result contract. Consequently,
`call_tool` returns a dictionary for object results, and a string for text or
non-object JSON results. Wrap a list in `{"items": ...}` when structured
programmatic consumption is useful. This version does not add custom media,
attachments or output schemas.

## Lifecycle and execution boundaries

Modules execute **inside the Praxis service process**, with its OS permissions;
importing runs top-level Python code. No new sandbox is introduced. Functions and
module globals are shared across sessions and workers, while tool registries and
approval contexts remain separate. Avoid storing per-session data in module
globals; shared mutable state needs synchronization when calls can overlap.

Modules are imported once per canonical file path in the process. Both code
changes and extension configuration changes require a service restart. A normal
model-settings reload keeps the existing tool catalog. A reload changing the
extension list (including IDs, resolved paths, enabled flags or order) is rejected
before updating settings or closing session agents. A failed batch publishes no
tools and removes its newly imported module entries, but cannot undo arbitrary
import-time filesystem/network effects.

Execution uses existing validation, approval, events, history, parallel scheduling
and programmatic dispatch. Cancellation is checked before execution; arbitrary
blocking synchronous Python cannot be safely forcibly terminated. There is no
hot reload, plugin marketplace, package installer or public internal `ToolContext`
injection in this version. Read credentials from service environment variables.

## Troubleshooting

- **Missing dependency:** install it with the service's Python interpreter, e.g.
  `.venv/Scripts/python.exe -m pip install package-name`, then restart the service.
  The import error identifies the interpreter. The `run_code` session environment
  is distinct from the service environment.
- **Unsupported type/signature:** the loading error identifies the extension,
  function (and the input field when available). Use a Pydantic-supported type
  with a JSON Schema. Explicit `Any`/`object` annotations allow arbitrary JSON values.
- **Invalid arguments:** inspect the tool error and correct missing fields, types
  or unknown keys. Schema-invalid inputs do not run constructors. Custom
  validation happens after approval; failed validation never invokes the tool function.
- **No new tools after editing:** restart the service; reopening a client or
  refreshing MCP status does not reload Python extensions.
- **Function/output failure:** inspect the explicit execution error. The runtime
  does not stringify arbitrary objects or automatically replay business actions.
