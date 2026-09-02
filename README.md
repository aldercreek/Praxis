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

Configure the model client in the current PowerShell session:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:PRAXIS_MODEL = "your-model"

# Only set this when using an OpenAI-compatible gateway.
$env:OPENAI_BASE_URL = "https://your-gateway.example.com"
```

## Run

Pass a task directly. Praxis can inspect and modify files under the current
directory through its registered tools:

```powershell
praxis "inspect this repository"
```

Or start it without an argument and enter one task when prompted:

```powershell
praxis
```

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
