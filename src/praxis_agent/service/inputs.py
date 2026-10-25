"""The execution boundary consumes durable input under the session's lock."""

from praxis_agent.execution import current_run


class TurnInputs:
    def __init__(self, runtime, first_id):
        self.runtime = runtime
        self.first_id = first_id

    def consume_initial(self, turn_id):
        store = self.runtime.store
        with store.db.transaction() as connection:
            connection.execute(
                "UPDATE inputs SET consumed_turn_id = ? WHERE command_id = ? AND consumed_turn_id IS NULL",
                (turn_id, self.first_id),
            )
            store.pending_changed(self.runtime.id)

    def receive(self, turn_id, accept, *, final=False):
        runtime = self.runtime
        with runtime.lock:
            current_run().check()
            pending = runtime.store.pending(runtime.id)
            if not pending:
                return False
            with runtime.store.db.transaction() as connection:
                # Context insertion, consumption and display events commit together.
                accept(
                    [{"role": "user", "content": item["content"]} for item in pending]
                )
                connection.executemany(
                    "UPDATE inputs SET consumed_turn_id = ? WHERE command_id = ?",
                    [(turn_id, item["command_id"]) for item in pending],
                )
                runtime.store.pending_changed(runtime.id)
            return True
