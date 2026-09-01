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

## Run

Pass a task directly:

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

