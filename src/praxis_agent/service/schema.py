SCHEMA = """
CREATE TABLE IF NOT EXISTS commands (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    result TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inputs (
    command_id TEXT PRIMARY KEY REFERENCES commands(id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    content TEXT NOT NULL,
    consumed_turn_id TEXT,
    deleted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS inputs_session ON inputs(session_id);
CREATE TABLE IF NOT EXISTS active_turns (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id),
    turn_id TEXT NOT NULL UNIQUE,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    turn_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'approved', 'denied', 'cancelled'))
);
CREATE TABLE IF NOT EXISTS events (
    session_id TEXT NOT NULL REFERENCES sessions(id),
    sequence INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(session_id, sequence)
);
CREATE TABLE IF NOT EXISTS tool_calls (
    session_id TEXT NOT NULL REFERENCES sessions(id),
    agent_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments TEXT,
    status TEXT NOT NULL,
    result TEXT,
    PRIMARY KEY(session_id, agent_id, call_id)
);
"""
