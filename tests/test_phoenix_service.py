import socket
import sqlite3
import sys
import pytest
from threading import Event
from time import monotonic, sleep

from praxis_agent.events import TurnStarted, ModelRequestStarted, ModelRequestCompleted, TurnCompleted
from praxis_agent.model import ModelResponse, TokenUsage
from praxis_agent.phoenix_service import LocalPhoenix
from praxis_agent.phoenix_sink import PhoenixSink


def test_missing_phoenix_reports_the_actual_python_environment(tmp_path, monkeypatch):
    monkeypatch.setattr("praxis_agent.phoenix_service.find_spec", lambda name: None)
    service = LocalPhoenix(tmp_path)
    with pytest.raises(RuntimeError) as error:
        service.start(Event())
    assert sys.executable in str(error.value)
    assert "pip install" in str(error.value)
    assert service.process is None


def test_real_phoenix_dynamic_port_ingestion_persistence_and_shutdown(tmp_path):
    # Occupy an unrelated local port throughout; Phoenix must own a different socket.
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        service = LocalPhoenix(tmp_path)
        sink = None
        errors = []
        try:
            url = service.start(Event())
            assert not url.endswith(":" + str(occupied.getsockname()[1]))
            process = service.process
            sink = PhoenixSink(url, on_error=errors.append)
            sink.emit_for_session("test-session", TurnStarted("turn", "test task"))
            sink.emit_for_session("test-session", ModelRequestStarted("turn", 1, model="test-model"))
            response = ModelResponse((), "done", (), model="test-model", usage=TokenUsage(10, 2, 12))
            sink.emit_for_session("test-session", ModelRequestCompleted("turn", 1, response))
            sink.emit_for_session("test-session", TurnCompleted("turn"))
            sink.close()
            deadline = monotonic() + 10
            while monotonic() < deadline:
                with sqlite3.connect(tmp_path / "phoenix.db") as connection:
                    count = connection.execute("SELECT count(*) FROM spans").fetchone()[0]
                if count == 2:
                    break
                sleep(0.1)
            assert count == 2
            assert errors == []
        finally:
            if sink:
                sink.close()
            service.close()
        assert process.poll() is not None
        assert occupied.getsockname()[1] != 0
        try:
            service.start(Event())
            with sqlite3.connect(tmp_path / "phoenix.db") as connection:
                assert connection.execute("SELECT count(*) FROM spans").fetchone()[0] == 2
        finally:
            service.close()
