"""A growing message editor; submission stays on the UI thread."""

from rich.style import Style
from textual import events, on
from textual.message import Message
from textual.widgets import TextArea
from textual.widgets.text_area import TextAreaTheme


class Composer(TextArea):
    class Submitted(Message):
        def __init__(self, editor: "Composer") -> None:
            super().__init__()
            self.editor = editor
            self.text = editor.text

    def __init__(self) -> None:
        super().__init__(id="task", soft_wrap=True, show_line_numbers=False)
        self.register_theme(TextAreaTheme("praxis", cursor_style=Style(), cursor_line_style=Style(),
                                          bracket_matching_style=Style()))
        self.theme = "praxis"
        self.cursor_blink = False
        self.match_cursor_bracket = False

    async def _on_key(self, event: events.Key) -> None:
        self.app.clear_text_selection()
        if event.key in {"enter", "shift+enter"}:
            event.stop()
            event.prevent_default()
            if not self.read_only:
                if event.key == "enter":
                    self.post_message(self.Submitted(self))
                else:
                    self.replace("\n", *self.selection, maintain_selection_offset=False)
            return
        if event.character == "?" and not self.text:
            event.stop()
            event.prevent_default()
            self.app.action_show_shortcuts()
            return
        if not self.read_only:
            row = self.wrapped_document.location_to_offset(self.cursor_location).y
            direction = -1 if event.key == "up" else 1 if event.key == "down" else 0
            at_edge = row == (0 if direction < 0 else self.wrapped_document.height - 1)
            if direction and at_edge and self.selection.is_empty and self.app.recall_input(direction):
                event.stop()
                event.prevent_default()
                return
            if event.key == "tab":
                event.stop()
                event.prevent_default()
                self.app.action_complete_command()
                return
        await super()._on_key(event)

    @on(TextArea.Changed)
    def resize_to_content(self) -> None:
        self.styles.height = min(max(1, self.wrapped_document.height),
                                 max(1, min(10, self.app.size.height // 3)))
        self.call_after_refresh(self.scroll_cursor_visible)

    def on_resize(self) -> None:
        self.call_after_refresh(self.resize_to_content)

    def load_draft(self, text: str) -> None:
        self.text = text
        self.move_cursor(self.document.end)
        self.focus()
