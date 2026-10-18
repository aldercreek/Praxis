"""Mouse selection over Rich-rendered terminal output."""

from rich.cells import get_character_cell_size
from rich.segment import Segment
from rich.style import Style
from textual import events, on
from textual.containers import VerticalScroll
from textual.geometry import Offset
from textual.strip import Strip
from textual.widgets import Static


class SelectableText(Static):
    """Keep Rich formatting while selecting by terminal cells, including wide glyphs."""

    _highlight = Style(color="#f2f2f2", bgcolor="#334d55", dim=False)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._anchor: tuple[int, int] | None = None
        self._end: tuple[int, int] | None = None
        self._drag_position: Offset | None = None
        self._drag_started = False
        self._scroll_timer = None
        self._selection_width = 0

    def _point(self, position: Offset) -> tuple[int, int]:
        offset = position - self.content_region.offset
        if offset.y < 0:
            return 0, 0
        row = min(offset.y, max(0, self.content_size.height - 1))
        limit = self.content_size.width if offset.y >= self.content_size.height else max(0, offset.x)
        # Snap to character boundaries so a Chinese glyph never becomes half a space.
        column = 0
        for char in super().render_line(row).text.rstrip():
            cells = get_character_cell_size(char)
            if column + cells > limit:
                break
            column += cells
        return row, column

    def _range(self, row: int, width: int) -> tuple[int, int]:
        if self._anchor is None or self._end is None:
            return 0, 0
        first, last = sorted((self._anchor, self._end))
        if not first[0] <= row <= last[0]:
            return 0, 0
        return (first[1] if row == first[0] else 0,
                last[1] if row == last[0] else width)

    @property
    def selected_text(self) -> str:
        if self._anchor is None or self._end is None or self._anchor == self._end:
            return ""
        first, last = sorted((self._anchor, self._end))
        lines = []
        for row in range(first[0], last[0] + 1):
            line = super().render_line(row)
            start, end = self._range(row, line.cell_length)
            lines.append(line.crop(start, end).text.rstrip())
        return "\n".join(lines)

    def render_line(self, y: int) -> Strip:
        line = super().render_line(y)
        start, end = self._range(y, line.cell_length)
        if start >= end:
            return line
        selected = line.crop(start, end)
        highlighted = Strip(Segment.apply_style(selected, post_style=self._highlight), selected.cell_length)
        return Strip.join((line.crop(0, start), highlighted, line.crop(end)))

    def clear_selection(self) -> None:
        self._finish_drag()
        self._anchor = self._end = None
        self.refresh()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 1:
            return
        event.stop()
        self.app.set_text_selection(self)
        self._anchor = self._end = self._point(event.screen_offset)
        self._selection_width = self.content_size.width
        self._drag_position = event.screen_offset
        self.capture_mouse()
        self._scroll_timer = self.set_interval(0.05, self._scroll_selection)
        self.refresh()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._drag_position is not None:
            event.stop()
            self._drag_started |= event.screen_offset != self._drag_position
            self._drag_position = event.screen_offset
            self._extend_selection()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if event.button == 1 and self._drag_position is not None:
            event.stop()
            self._drag_position = event.screen_offset
            self._extend_selection()
            if self._anchor != self._end:
                self.suppress_click()
                self._finish_drag()
            else:
                self.app.clear_text_selection()

    def _extend_selection(self) -> None:
        if self._drag_position is not None:
            self._end = self._point(self._drag_position)
            self.refresh()

    def _scroll_selection(self) -> None:
        if self._drag_position is None or not self._drag_started:
            return
        pane = next((node for node in self.ancestors if isinstance(node, VerticalScroll)), None)
        if pane is None:
            return
        visible = pane.content_region.intersection(self.screen.region)
        y = self._drag_position.y
        direction = -1 if y <= visible.y else 1 if y >= visible.bottom - 1 else 0
        if direction:
            pane.scroll_relative(y=direction, animate=False, immediate=True)
            self.call_after_refresh(self._extend_selection)

    def _finish_drag(self) -> None:
        self._drag_started = False
        if self._scroll_timer is not None:
            self._scroll_timer.stop()
            self._scroll_timer = None
        if self._drag_position is not None:
            self.release_mouse()
            self._drag_position = None

    def on_resize(self) -> None:
        if self._anchor is not None and self.content_size.width != self._selection_width:
            self.app.clear_text_selection()

    @on(events.Hide)
    @on(events.Unmount)
    def forget_selection(self) -> None:
        if self.app._selection_widget is self:
            self.app.clear_text_selection()
