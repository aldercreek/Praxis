"""In-memory HTTP boundary for isolated layout/selection tests, not an Agent."""


class FakeConnection:
    def __init__(
        self, *, workspace="D:/Projects/Praxis", model="auto(free)", active=False
    ):
        self.summary = {
            "id": "S1",
            "workspace": workspace,
            "model": model,
            "active_turn": {"id": "T1", "cancel_requested": False} if active else None,
        }
        self.pending = []
        self.approvals = {}
        self.connection_error = ""
        self.cancelled = False

    def drain(self, store):
        return set()

    def send(self, text):
        self.pending.append({"command_id": str(len(self.pending) + 1), "content": text})

    def delete(self, command_id):
        self.pending = [
            item for item in self.pending if item["command_id"] != command_id
        ]

    def cancel(self, turn_id):
        assert turn_id == self.summary["active_turn"]["id"]
        self.cancelled = True
        self.summary["active_turn"]["cancel_requested"] = True

    def close(self):
        pass

    def decide(self, approval_id, approved):
        return {
            "approval_id": approval_id,
            "status": "approved" if approved else "denied",
        }
