from textual import on
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class DeleteSessionDialog(ModalScreen[bool]):
    DEFAULT_CSS = """
    DeleteSessionDialog { align: center middle; background: $background 75%; }
    #delete-dialog { width: 64; max-width: 95%; height: auto; padding: 1 2; border: solid $primary; }
    #delete-dialog Static { height: auto; margin-bottom: 1; }
    #delete-actions { height: auto; align-horizontal: right; }
    #delete-actions Button { margin-left: 1; }
    """

    def __init__(self, summary):
        super().__init__()
        self.summary = summary

    def compose(self):
        active = self.summary.get("active_turn")
        message = "Permanently remove this chat and its history?"
        if active:
            message = ("This chat is stopping. Delete it after the task stops?"
                       if active.get("cancel_requested") else
                       "This chat is running. Stop the task and delete its history?")
        with Vertical(id="delete-dialog"):
            yield Static("Delete chat?", markup=False)
            yield Static(self.summary.get("title") or "New chat", markup=False)
            yield Static(message + " Files in your workspace will stay.", markup=False)
            with Horizontal(id="delete-actions"):
                yield Button("Cancel", id="keep-chat")
                yield Button("Delete chat", id="delete-chat", variant="error")

    def on_mount(self):
        self.query_one("#keep-chat", Button).focus()

    @on(Button.Pressed)
    def respond(self, event):
        self.dismiss(event.button.id == "delete-chat")
