"""Client session state. Textual owns rendering; this object owns transport."""

from queue import Empty, SimpleQueue
from uuid import uuid4

from .wire_events import decode_event


class TerminalSession:
    def __init__(self, client, summary):
        self.client = client
        self.summary = summary
        self.frames = SimpleQueue()
        self.pending = []
        self.approvals = {}
        self.sequence = 0
        self.subscription = None
        self.connection_error = ""
        self.initial_events = []
        self.deleted = False
        self.open(summary)

    @property
    def id(self):
        return self.summary["id"]

    def open(self, summary):
        if self.subscription:
            self.subscription.close()
        self.summary = summary
        self.deleted = False
        self.frames = SimpleQueue()
        self.subscription = self.client.subscribe(
            self.id, self.frames.put, on_error=self._error, on_ready=self._connected
        )
        snapshot = self.client.request("GET", f"/api/sessions/{self.id}/snapshot")
        self.summary, self.sequence = snapshot["session"], snapshot["sequence"]
        self.pending = snapshot["pending_inputs"]
        self.approvals = {item["approval_id"]: item for item in snapshot["approvals"]}
        self.initial_events = snapshot["view_events"]

    def _error(self, message):
        self.connection_error = message

    def _connected(self):
        self.connection_error = ""

    def drain(self, store, limit=1000):
        changed = set()
        initial, self.initial_events = self.initial_events, []
        for frame in initial:
            event = decode_event(frame)
            if event:
                changed.add(store.accept(event))
        self.subscription.acknowledge(self.sequence)
        for _ in range(limit):
            try:
                frame = self.frames.get_nowait()
            except Empty:
                break
            if frame["event"] == "SessionDeleted":
                self.deleted = True
                self.summary = {**self.summary, "active_turn": None}
                self.pending = []
                self.approvals.clear()
                self.connection_error = ""
                break
            if self.deleted:
                continue
            if frame["sequence"] <= self.sequence:
                continue
            data, kind = frame["data"], frame["event"]
            if kind == "SessionUpdated":
                self.summary = data["session"]
            elif kind == "InputsChanged":
                self.pending = data["inputs"]
            elif kind == "ApprovalPending":
                self.approvals[data["approval_id"]] = data
            elif kind == "ApprovalClosed":
                self.approvals.pop(data["approval_id"], None)
            event = decode_event(frame)
            if event:
                changed.add(store.accept(event))
            self.sequence = frame["sequence"]
            self.subscription.acknowledge(self.sequence)
            self.connection_error = ""
        return changed

    def send(self, text, command_id=None):
        return self.client.request(
            "POST",
            f"/api/sessions/{self.id}/inputs",
            {"command_id": command_id or uuid4().hex, "text": text},
        )

    def cancel(self, turn_id):
        return self.client.request(
            "POST", f"/api/sessions/{self.id}/turns/{turn_id}/cancel"
        )

    def decide(self, approval_id, approved):
        return self.client.request(
            "POST", f"/api/approvals/{approval_id}/decision", {"approved": approved}
        )

    def resume(self):
        return self.client.request(
            "POST",
            f"/api/sessions/{self.id}/inputs/resume",
            {"command_id": uuid4().hex},
        )

    def delete(self, command_id):
        return self.client.request(
            "DELETE", f"/api/sessions/{self.id}/inputs/{command_id}"
        )

    def delete_session(self):
        return self.client.request("DELETE", f"/api/sessions/{self.id}")

    def clear(self):
        self.open(
            self.client.create(self.summary["workspace"], model=self.summary["model"])
        )

    def commands(self):
        return self.client.request("GET", f"/api/sessions/{self.id}/commands")[
            "commands"
        ]

    def mcp_servers(self):
        return self.client.request("GET", "/api/mcp/servers")["servers"]

    def stop_command(self, command_id):
        return self.client.request(
            "POST", f"/api/sessions/{self.id}/commands/{command_id}/stop"
        )

    def close(self):
        self.client.close()
