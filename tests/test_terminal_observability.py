import asyncio

from textual.widgets import TextArea

from praxis_agent.terminal.app import PraxisApp
from praxis_agent.terminal.views import ViewEventSink
from praxis_agent.terminal.widgets import DetailsScreen
from tests.support.terminal import FakeConnection


def test_terminal_warns_once_and_shows_retry_status(tmp_path):
    class Connection(FakeConnection):
        def __init__(self):
            super().__init__()
            self.state = "failed"
            self.retries = 0
        def observability_status(self, *, retry=False):
            if retry:
                self.retries += 1
                self.state = "starting"
            return dict(provider="trip", state=self.state, url="",
                        message="trip is not installed. Agent execution continues.")
    async def exercise():
        connection = Connection()
        app = PraxisApp(ViewEventSink(), connection=connection)
        async with app.run_test(size=(100, 30)) as pilot:
            await app._poll_observability()
            await pilot.pause()
            await app._poll_observability()
            await pilot.pause()
            warnings = [b for b in app.store.get("main").committed_blocks if "Tracing unavailable" in str(b)]
            assert len(warnings) == 1
            assert "trip is not installed" in str(warnings[0])
            assert any(str(span.style) == "red" for span in warnings[0].spans)
            assert app.query_one("#record-0").is_on_screen
            assert not app.busy
            assert app.screen is app._main_screen
            app.save_screenshot("observability-warning.svg", path=str(tmp_path))
            app.query_one("#task", TextArea).text = "/observability"
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen is app._main_screen
            assert "trip is not installed" in str(app.store.get("main").committed_blocks[-1])
            app.save_screenshot("observability-details.svg", path=str(tmp_path))
            app.query_one("#task", TextArea).text = "/observability retry"
            await pilot.press("enter")
            await pilot.pause()
            assert connection.retries == 1
            assert connection.state == "starting"
    asyncio.run(exercise())
