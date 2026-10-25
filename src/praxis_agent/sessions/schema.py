_SCHEMA_VERSION = 5
_INTERRUPTED_RUN_ERROR = "Praxis stopped before this run completed."
_INTERRUPTED_TOOL_OUTPUT = (
    "Tool execution was interrupted. Its completion is unknown.\n"
    "Inspect the workspace before deciding whether to retry."
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    model TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS sessions_workspace_updated_idx
    ON sessions(workspace, updated_at DESC);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    original_task TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('running', 'completed', 'failed', 'cancelled')
    ),
    error TEXT,
    plan_json TEXT NOT NULL,
    changed_files_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    UNIQUE(session_id, id)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_running_turn_per_session ON runs(session_id) WHERE status = 'running';

CREATE INDEX IF NOT EXISTS runs_session_started_idx
    ON runs(session_id, started_at);

CREATE TABLE IF NOT EXISTS conversation_items (
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    item_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(session_id, sequence),
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY(session_id, run_id)
        REFERENCES runs(session_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS conversation_items_run_idx
    ON conversation_items(run_id);

CREATE TABLE IF NOT EXISTS context_view_operations (
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(session_id, sequence),
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

"""
