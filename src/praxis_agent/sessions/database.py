"""Connection ownership and nested units of work shared by session repositories."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import local

from .models import SessionStoreError


def data_directory() -> Path:
    return (
        Path(os.environ.get("PRAXIS_DATA_DIR", Path.home() / ".praxis"))
        .expanduser()
        .resolve()
    )


def default_database_path() -> Path:
    return data_directory() / "state.sqlite3"


class Database:
    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = local()

    @contextmanager
    def connection(self):
        active = getattr(self._local, "connection", None)
        if active is not None:
            yield active
            return
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
        except sqlite3.Error as error:
            raise SessionStoreError(f"session database error: {error}") from error
        finally:
            connection.close()

    @contextmanager
    def transaction(self, *, write=True):
        active = getattr(self._local, "connection", None)
        if active is not None:
            try:
                yield active
            except BaseException:
                self._local.rollback = True
                raise
            return
        with self.connection() as connection:
            # WAL readers see a consistent snapshot without blocking Agent
            # writers while a transcript is reconstructed.
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            self._local.connection = connection
            self._local.rollback = False
            try:
                yield connection
                if self._local.rollback:
                    raise SessionStoreError("nested transaction failed")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                self._local.connection = None
