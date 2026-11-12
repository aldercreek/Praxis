import asyncio
from threading import Event

from praxis_agent.terminal.cells import MessageBlock
from praxis_agent.terminal.composer import Composer
from praxis_agent.terminal.slash_commands import matching_commands, mcp_result
from support.terminal import FakeConnection


class CommandConnection(FakeConnection):
    def __init__(self):
        super().__init__()
        self.id = "S1"
        self.stopped = []
        self.deleted = False
        self.calls = 0
        self.release = None

    def commands(self):
        return [dict(session_id='CMD-1', status='running', argv=['python', 'worker.py']),
                dict(session_id='CMD-2', status='exited', argv=['echo', 'done'])]

    def stop_command(self, session_id):
        self.stopped.append(session_id)

    def delete_session(self):
        self.deleted = True

    def mcp_servers(self):
        self.calls += 1
        if self.release:
            self.release.wait(5)
        return [dict(name='DOT', state='connected', tool_count=16),
                dict(name='captain-mcp', state='failed', tool_count=0, error='ConnectionError'),
                dict(name='codex_app', state='disabled', tool_count=0)]


def test_matching_requires_slash_and_prioritizes_prefix():
    assert not matching_commands('help')
    assert [x[0] for x in matching_commands('/COMMAND')] == ['/commands', '/stop-command']
    assert matching_commands('/observability r')[0][0] == '/observability retry'
    assert not matching_commands('/observability invalid')


def test_escape_dismisses_menu_without_stopping_task(terminal_app):
    async def exercise():
        app = terminal_app
        app.connection = FakeConnection(active=True)
        async with app.run_test() as pilot:
            editor = app.query_one(Composer)
            editor.load_draft('/m')
            await pilot.pause()
            await pilot.press('escape')
            await pilot.pause(.2)
            assert editor.text == '/m'
            assert not app.query_one('#shortcuts').display
            assert not app.connection.cancelled
            await pilot.press('c')
            assert app.query_one('#shortcuts').display
    asyncio.run(exercise())


def test_invalid_commands_preserve_draft_and_never_send_to_model(terminal_app):
    async def exercise():
        app = terminal_app
        app.connection = CommandConnection()
        async with app.run_test() as pilot:
            editor = app.query_one(Composer)
            for draft in ('/unknown', '/mcp extra', '/observability wrong', '/stop-command one two'):
                editor.load_draft(draft)
                await pilot.press('enter')
                assert editor.text == draft
                assert not app.connection.pending
                assert app.query_one('#hint').renderable.style == 'red'
    asyncio.run(exercise())


def test_parameter_completion_and_inline_delete(terminal_app):
    async def exercise():
        app = terminal_app
        app.connection = CommandConnection()
        async with app.run_test(size=(80, 24)) as pilot:
            editor = app.query_one(Composer)
            editor.load_draft('/stop')
            await pilot.press('enter')
            await pilot.pause()
            assert editor.text == '/stop-command '
            assert len(app._matching_commands(editor.text)) == 1
            await pilot.press('tab')
            assert editor.text == '/stop-command CMD-1'
            await pilot.press('enter')
            await pilot.pause()
            assert app.connection.stopped == ['CMD-1']
            assert 'Stop requested' in str(app.store.get('main').committed_blocks[-1])
            editor.load_draft('/delete')
            await pilot.press('enter')
            assert app.screen is app._main_screen and app._delete_confirmation
            await pilot.press('enter')  # Cancel is the default.
            assert not app._delete_confirmation and not app.connection.deleted
            editor.load_draft('/delete')
            await pilot.press('enter', 'escape')
            assert not app._delete_confirmation and not app.connection.deleted
            editor.load_draft('/delete')
            await pilot.press('enter', 'down', 'enter')
            await pilot.pause()
            assert app.connection.deleted
    asyncio.run(exercise())


def test_slow_query_preserves_new_draft_and_history_position(terminal_app):
    async def exercise():
        app = terminal_app
        connection = app.connection = CommandConnection()
        connection.release = Event()
        app.store.get('main').committed_blocks.append(MessageBlock('\n\n'.join(f'old {i}' for i in range(60))))
        async with app.run_test(size=(80, 24)) as pilot:
            editor = app.query_one(Composer)
            editor.load_draft('/mcp')
            await pilot.press('enter')
            await pilot.pause()
            assert '/mcp' in app._command_jobs
            editor.load_draft('/mcp')
            await pilot.press('enter')
            assert editor.text == '/mcp' and connection.calls == 1
            editor.load_draft('new draft')
            await pilot.press('pageup')
            await pilot.pause()
            pane = app.query_one('#transcript')
            position = pane.scroll_y
            try:
                connection.release.set()
                async with asyncio.timeout(3):
                    while app._command_jobs:
                        await pilot.pause(.02)
                assert editor.text == 'new draft'
                assert pane.scroll_y == position
                assert app.screen is app._main_screen
                assert 'DOT: connected (16 tools)' in str(app.store.get('main').committed_blocks[-1])
            finally:
                connection.release.set()
    asyncio.run(exercise())


def test_mcp_renderer_uses_status_colors_and_real_counts():
    text = mcp_result(CommandConnection().mcp_servers())
    assert 'DOT: connected (16 tools)' in text.plain
    assert 'captain-mcp: failed (0 tools)' in text.plain
    assert any(text.plain[s.start:s.end] == 'connected' and s.style == 'green' for s in text.spans)
    assert any(text.plain[s.start:s.end] == 'failed' and s.style == 'red' for s in text.spans)


def test_query_failure_stays_inline_and_retry_succeeds(terminal_app):
    class Connection(CommandConnection):
        def mcp_servers(self):
            if self.calls == 0:
                self.calls += 1
                raise RuntimeError('Service unavailable')
            return super().mcp_servers()

    async def exercise():
        app = terminal_app
        app.connection = Connection()
        async with app.run_test(size=(44, 24)) as pilot:
            editor = app.query_one(Composer)
            editor.load_draft('/mcp')
            await pilot.press('enter')
            await pilot.pause()
            assert app.screen is app._main_screen
            assert not app._command_jobs
            assert 'Retry: /mcp' in str(app.store.get('main').committed_blocks[-1])
            editor.load_draft('/mcp')
            await pilot.press('enter')
            await pilot.pause()
            assert 'connected (16 tools)' in str(app.store.get('main').committed_blocks[-1])
            assert app.focused is editor
            editor.load_draft('/help')
            await pilot.press('enter')
            await pilot.pause()
            assert app.screen is app._main_screen
            assert 'Keyboard shortcuts' in str(app.store.get('main').committed_blocks[-1])
    asyncio.run(exercise())
