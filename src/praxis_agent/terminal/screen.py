"""Settle the transcript viewport before Textual submits a frame."""

from textual.containers import VerticalScroll
from textual.screen import Screen


class ConversationScreen(Screen):
    _settling = False

    def _refresh_layout(self, size=None, scroll=False) -> None:
        if not self._settling and getattr(self.app, "_main_screen", None) is self:
            # Capture user scrolling against the old geometry, before content grows.
            self.app._remember_scroll()
        super()._refresh_layout(size, scroll)

    def _compositor_refresh(self) -> None:
        if self._settling:
            return
        app = self.app
        if getattr(app, "_main_screen", None) is self:
            self._settling = True
            try:
                pane = self.query_one("#transcript", VerticalScroll)
                view = app.store.get(app.selected_agent_id)
                self._position_transcript(pane, view)
                app._switching_page = False
                button = self.query_one("#jump-latest")
                was_visible = button.display
                app._refresh_jump_latest()
                if button.display != was_visible:
                    # The affordance changes the available height by one row.
                    super()._refresh_layout()
                    self._position_transcript(pane, view)
            finally:
                self._settling = False
        super()._compositor_refresh()

    def _position_transcript(self, pane, view) -> None:
        target = pane.max_scroll_y if view.follow_output else view.scroll_position
        previous = pane.scroll_y
        pane.scroll_to(y=target, animate=False, immediate=True)
        if pane.scroll_y != previous:
            # Reflow without publishing the intermediate, unscrolled frame.
            super()._refresh_layout(scroll=True)
        view.scroll_position = pane.scroll_y
