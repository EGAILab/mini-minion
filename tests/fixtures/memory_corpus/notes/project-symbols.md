# Minion Assist code identifiers

`AgentSession.send()` in `agents/session.py` is the entry point for a single
turn — it acquires a lock, builds the system prompt, and calls `run_turn()`.

`run_turn()` in `agents/runner.py` implements the Think-Act-Observe loop.

Config is loaded at import time in `config.py`; `_resolve_all()` builds the
`agents` mapping from `config.json`.
