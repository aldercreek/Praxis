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

Praxis uses the ADA gateway and `gpt-5.6-terra-2026-07-09` by default. These
optional variables can override either value:

```powershell
$env:ADA_MODEL = "gpt-5.6-sol-2026-07-09"
$env:ADA_BASE_URL = "https://ada-cli-go.ctripcorp.com/coding-plan/openai/v1"
```

At startup, Praxis tells the model only the operating system, resolved
workspace root, and which common shells are available or unavailable. It does
not send environment-variable values or the full system `PATH`.

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

Start it without an argument to keep a conversation in the current process:

```powershell
praxis
```

In interactive mode, each new task includes the earlier conversation and tool
results. Use `/clear` to start a fresh conversation and `/exit` to leave:

```text
praxis> inspect config.py
praxis> explain the previous result in more detail
praxis> /clear
Conversation cleared.
praxis> /exit
```

Conversation history is currently kept only in memory and disappears when the
Praxis process exits.

## Test

```powershell
python -m pytest
```

## Structure

```text
src/praxis_agent/  Application package
tests/             Automated tests
pyproject.toml     Project metadata, dependencies, and CLI entry point
```
