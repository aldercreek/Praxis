"""Slash-command vocabulary and compact, selectable result renderers."""

from rich.text import Text


COMMANDS = (
    ("/clear", "Start a new chat"),
    ("/help", "Keyboard shortcuts"),
    ("/exit", "Quit Praxis"),
    ("/commands", "Command sessions"),
    ("/stop-command", "Stop a running command"),
    ("/mcp", "MCP tools and connections"),
    ("/observability", "Tracing status"),
    ("/delete", "Delete this chat"),
)


def matching_commands(text, sessions=()):
    if not text.startswith("/") or "\n" in text:
        return []
    query = text.casefold()
    if " " in query:
        name, _ = query.split(" ", 1)
        if name == "/observability":
            choices = [("/observability retry", "Retry tracing connection")]
        elif name == "/stop-command":
            choices = [(f"/stop-command {item['session_id']}", " ".join(item['argv']))
                       for item in sessions if item["status"] == "running"]
        else:
            return []
        return [item for item in choices if item[0].casefold().startswith(query)]
    matches = [item for item in COMMANDS if query[1:] in item[0][1:]]
    return sorted(matches, key=lambda item: (item[0] != query, not item[0].startswith(query)))


def command_error(text):
    parts = text.split()
    name = parts[0].casefold()
    if name not in dict(COMMANDS):
        return "Unknown command. Type / to browse commands."
    if name == "/stop-command":
        return None if len(parts) == 2 else "Usage: /stop-command <session_id>"
    if name == "/observability":
        return None if len(parts) == 1 or (len(parts) == 2 and parts[1].casefold() == "retry") else "Usage: /observability [retry]"
    return None if len(parts) == 1 else f"{name} does not take arguments."


def mcp_result(servers):
    text = Text("MCP Tools\n", style="")
    text.stylize("bold", 0, 9)
    if not servers:
        text.append("\n  No MCP servers configured.", style="#a0a0a0")
    for item in servers:
        state = item.get("state", "unknown")
        color = {"connected": "green", "failed": "red", "connecting": "yellow",
                 "queued": "yellow", "retrying": "yellow"}.get(state, "#808080")
        text.append("\n• ", style=color)
        text.append(item["name"] + ": ", style="bold")
        text.append(state, style=color)
        count = item.get("tool_count")
        text.append(f" ({count} tools)" if count is not None else " (tools unavailable)", style="#808080")
        error = item.get("error")
        if error and error not in ("ExceptionGroup", "BaseExceptionGroup"):
            text.append("\n  " + str(error), style="#a0a0a0")
    return text


def sessions_result(sessions, width):
    text = Text("Command sessions", style="bold")
    if not sessions:
        text.append("\n\n  No command sessions.", style="not bold #a0a0a0")
        return text
    id_width = max(len(item["session_id"]) for item in sessions)
    for item in sessions:
        state = item["status"]
        text.append("\n\n" if width < id_width + 24 else "\n", style="not bold")
        text.append(item["session_id"].ljust(id_width) + "  ", style="not bold")
        text.append(state.ljust(10), style="not bold " + ("yellow" if state == "running" else "#a0a0a0"))
        text.append(("\n  " if width < id_width + 24 else "  ") + " ".join(item["argv"]), style="not bold")
    text.append("\n\nStop a running session: /stop-command <session_id>", style="not bold #a0a0a0")
    return text


def tracing_result(status):
    text = Text("Tracing\n\n", style="bold")
    state = status["state"]
    for label, value in (("Provider", status["provider"]), ("Status", state), ("Viewer", status.get("url"))):
        if value:
            text.append(f"  {label:<10}", style="not bold #a0a0a0")
            text.append(str(value) + "\n", style="not bold " + ("red" if label == "Status" and state == "failed" else "default"))
    if status.get("message"):
        text.append("\n" + status["message"], style="not bold")
    if state == "failed":
        text.append("\n\nRetry: /observability retry", style="not bold #a0a0a0")
    return text


def help_result():
    text = Text("Keyboard shortcuts", style="bold")
    groups = (
        ("Commands", (("/", "Browse commands"), ("↑ / ↓", "Select a command"),
                      ("Tab", "Complete command or parameter"), ("Enter", "Run selected command"),
                      ("Esc", "Dismiss menu or cancel confirmation"))),
        ("Input", (("Enter", "Send message or follow-up"), ("Shift+Enter", "New line"),
                   ("↑ / ↓", "Recall input at editor boundaries"))),
        ("Navigation", (("PgUp / PgDn", "Scroll conversation"), ("Ctrl+Home / End", "First / latest message"),
                        ("Ctrl+T", "Full execution transcript"), ("Alt+← / →", "Switch agent"))),
        ("Run and copy", (("Esc", "Stop turn when no menu is open"), ("Ctrl+R", "Send retained follow-ups"),
                         ("Shift+←", "Edit pending message"), ("Ctrl+C", "Copy selection; otherwise stop or quit"),
                         ("Ctrl+Shift+C", "Copy selection"), ("Ctrl+Q", "Quit"))),
    )
    for title, entries in groups:
        text.append("\n\n" + title + "\n", style="bold")
        for key, description in entries:
            text.append(f"  {key:<18}", style="not bold")
            text.append(description + "\n", style="not bold #a0a0a0")
    return text
