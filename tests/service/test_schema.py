import sqlite3

import pytest

from praxis_agent.sessions import SQLiteSessionStore, SessionStoreError
from praxis_agent.service.store import ServiceStore


@pytest.mark.parametrize("version", [1, 5, 7])
def test_unsupported_database_is_rejected_without_migration(tmp_path, version):
    path = tmp_path / "state.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {version}")
        connection.execute("CREATE TABLE saved_marker (value TEXT)")
        connection.execute("INSERT INTO saved_marker VALUES ('keep')")
    with pytest.raises(SessionStoreError, match="unsupported session database"):
        SQLiteSessionStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == version
        assert connection.execute("SELECT value FROM saved_marker").fetchone()[0] == "keep"
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("saved_marker",)]


def test_current_database_reopens_with_service_columns(tmp_path):
    path = tmp_path / "state.sqlite3"
    sessions = SQLiteSessionStore(path)
    service = ServiceStore(sessions)
    session = sessions.create_session(tmp_path, "test")
    service.append_event(session.id, "TurnStarted", "turn", data={"task": "saved"})
    reopened = ServiceStore(SQLiteSessionStore(path))
    assert reopened.snapshot(session.id)["messages"][0]["content"] == "saved"
