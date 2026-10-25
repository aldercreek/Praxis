import socket
from threading import Thread
from time import monotonic, sleep
from uuid import uuid4

import pytest
import uvicorn

from praxis_agent.agent import Agent
from praxis_agent.client import ServiceClient
from praxis_agent.model import ModelResponse
from praxis_agent.service.http import create_app
from praxis_agent.service.manager import SessionManager
from praxis_agent.service.store import ServiceStore
from praxis_agent.sessions import SQLiteSessionStore
from praxis_agent.tools import create_default_registry


def answer(text="done"):
    return ModelResponse(({"role": "assistant", "content": text},), text, ())


def wait_until(predicate, timeout=5):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        value = predicate()
        if value:
            return value
        sleep(0.01)
    raise AssertionError("Condition did not become true")


class EchoModel:
    def create_response(self, input_items, tools, on_text_delta=None, **kwargs):
        if on_text_delta:
            on_text_delta("done")
        return answer()


class Harness:
    def __init__(self, path, models):
        self.path = path
        self.models = list(models)
        self.agents = []
        self.store = ServiceStore(SQLiteSessionStore(path / "state.sqlite3"))
        self.store.recover()

        def factory(workspace, **options):
            store = options["session_store"]
            session_id = options["resume_session_id"]
            agent = Agent(
                self.models.pop(0) if self.models else EchoModel(),
                create_default_registry(),
                workspace,
                event_sink=options["event_sink"],
                approve_tool_call=options["approve_tool_call"],
                bypass_approvals=options["bypass_approvals"],
                session_store=store,
                session_id=session_id,
                session_model="test",
            )
            agent.restore_history(store.load_conversation_items(session_id))
            self.agents.append(agent)
            return agent

        self.manager = SessionManager(self.store, agent_factory=factory)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        self.endpoint = {
            "url": f"http://127.0.0.1:{listener.getsockname()[1]}",
            "token": "test-token",
            "instance_id": "test-instance",
        }
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_app(
                    self.manager,
                    token="test-token",
                    instance_id="test-instance",
                    heartbeat=0.1,
                ),
                log_level="error",
                timeout_graceful_shutdown=1,
            )
        )
        self.thread = Thread(
            target=lambda: self.server.run(sockets=[listener]), daemon=True
        )
        self.thread.start()
        wait_until(lambda: self.server.started)
        self.clients = []

    def client(self):
        client = ServiceClient(endpoint=self.endpoint)
        self.clients.append(client)
        return client

    def create(self, client=None):
        return (client or self.client()).create(self.path, model="test")

    def submit(self, session, text="task", command_id=None):
        return self.client().request(
            "POST",
            f"/api/sessions/{session['id']}/inputs",
            {"command_id": command_id or uuid4().hex, "text": text},
        )

    def settled(self, session):
        return wait_until(
            lambda: (
                snapshot
                if (snapshot := self.store.snapshot(session["id"]))["session"][
                    "latest_turn"
                ]
                and not snapshot["session"]["active_turn"]
                else None
            )
        )

    def close(self):
        for client in self.clients:
            client.close()
        self.manager.close()
        self.server.should_exit = True
        self.thread.join(5)
        assert not self.thread.is_alive()


@pytest.fixture
def service(tmp_path):
    harnesses = []

    def create(*models):
        path = tmp_path / str(len(harnesses))
        path.mkdir()
        harness = Harness(path, models)
        harnesses.append(harness)
        return harness

    yield create
    for harness in harnesses:
        harness.close()
