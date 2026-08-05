# ADR 0006: Matrix rooms are the unit of session isolation, not threads

## Status

Accepted.

## Context

The Matrix channel originally computed a per-thread session key
(`MatrixThreadBindingManager.get_or_create_session_key()`, keyed by
`(thread_event_id, room_id, agent_id)`) intended to give each Matrix
*thread* its own isolated conversation. That key was computed in
`handler.py`'s `handle_room_message()` but never actually passed into
`_dispatch_and_reply()` — every message routed to an agent, from every
room, thread, and sender, was dispatched to the single shared
`sessions[agent_id]` `AgentSession`. In practice this meant an unrelated
message in one room could see (and add to) another room's conversation
history, and the thread-binding machinery had no effect at all — a defect
identified during a from-scratch gap analysis against OpenClaw's memory/
session architecture (`minion-assist-docs/improve/openclaw-memory-gap-analysis.md`,
MEM-GAP-001).

Fixing this requires deciding what the actual unit of conversation
isolation should be for this deployment. This deployment's operating model
(confirmed directly, not assumed) is:

- Each Matrix room is a deliberately created, persistent topic (e.g. a
  "Movie" room, a "Work" room) — analogous to a Slack channel, not a
  throwaway thread.
- Every room has exactly two members: the bot (Ada) and one person (Eric).
  Group chats (more than two members) are not used.
- Matrix's own threading feature (`m.thread` relations, Element's "Reply in
  thread") is not used at all.

Given that, per-thread isolation is solving the wrong problem: there are no
threads to isolate, and per-sender isolation within a room is moot since
only one real person ever sends in a given room. The room itself is already
the natural, deliberately-managed conversation boundary.

## Decision

Session isolation is scoped to `(room_id, agent_id)`, not
`(thread_event_id, room_id, agent_id)`. `matrix/room_sessions.py`'s
`MatrixRoomSessionManager` replaces `thread_bindings.py`'s
`MatrixThreadBindingManager`: it resolves a stable `session_id` for every
room unconditionally (not just when a message happens to arrive inside a
Matrix thread), persisted in SQLite (`matrix/room_sessions.db`) so a room's
session survives restarts. A room's first message always mints a brand new
`session_id` — it never inherits the old shared session's mixed-topic
history.

`MatrixMessageHandler` no longer selects `sessions[agent_id]` for a live
turn. Instead, `minion.py` builds a per-agent session factory
(`matrix_session_factories[agent_id]`) that shares that agent's provider,
tools, memory service, and compactor — the same expensive, stateful
resources `sessions[agent_id]` uses — but constructs a fresh `AgentSession`
per `session_id`. The handler lazily builds and caches one `AgentSession`
per `(agent_id, room_id)` for the life of the process
(`_get_or_build_session`), falling back to the shared `sessions[agent_id]`
(with a loud warning, not silently) only if no factory was wired for that
agent.

Matrix's thread-reply UI (`relates_to.rel_type == "m.thread"`) is
unaffected by this decision and is left as-is: if a message does happen to
arrive inside a thread, the bot's reply is still posted into that same
thread for display purposes. That's a cosmetic, per-message concern
(`_resolve_thread_id`/`thread_id`), entirely decoupled from session
selection now.

No configuration is exposed for this — every room is isolated
unconditionally. `MatrixThreadBindingsConfig` and the `threadBindings`
config key are removed rather than repurposed with new semantics, since
nothing meaningfully depended on the old idle/max-age eviction behavior
(rooms are persistent topics, not ephemeral threads, so bindings here never
expire).

## Consequences

- A room's conversation history is now genuinely private to that room, for
  a given agent — fixing MEM-GAP-001.
- `SessionSearchTool` (see `docs/adr` — MEM-GAP-002, agent-level isolation)
  and this room-scoping are independent controls: an agent's `session_search`
  tool can still search across that same agent's other rooms/sessions on
  purpose (that's the tool's job), it just can never cross into another
  *agent's* sessions.
- Restarting the process does not lose per-room continuity: `room_sessions.db`
  remembers each room's `session_id`.
- `thread_bindings.py`, `MatrixThreadBindingManager`, and
  `MatrixThreadBindingsConfig` are removed outright (not left
  disabled-by-default) — they never worked as shipped, and threads are out
  of scope for this deployment's design.
- Slash-command dispatch (`/agents`, `/session`, `/rename`, etc. issued from
  a Matrix room) is unchanged: it still operates against the shared
  `sessions` dict, since that's a separate, deliberately global
  administrative surface, not the per-turn chat history this ADR scopes.
- If this deployment's model changes later (e.g. a room is shared with more
  than one person, or Matrix threads are adopted), that's a new decision to
  revisit against this ADR, not a silent behavior change.
