# ADR 0002: PostgreSQL is the single operational store for derived memory state

## Status

Accepted (Stage One, Phase 0).

## Context

Minion Assist already has a configured PostgreSQL instance
(`config.json`'s `"database"` section) used today for session/message
persistence and full-text transcript search (`session/db.py`). The three
prior memory analyses independently considered several places to put new
machine state needed by the memory system: job queues, capture cursors,
recall telemetry, injection residency, entity links, and (optionally)
embeddings.

Mem0's SQLite side-store for mutation history was reviewed
(`mem0-memory.md`) and explicitly rejected for Minion: it would introduce a
second database with its own backup/migration story, splitting operational
state across two systems for no benefit, since PostgreSQL is already
deployed and already transactional.

## Decision

All *derived*, *machine* memory state lives in the already-configured
PostgreSQL database — one schema, one set of migrations, one set of
transactional guarantees. This includes (see the plan's schema section):
`message_mirrors`, `memory_sources`, `memory_chunks`, `memory_capture_jobs`,
`memory_proposals`, `memory_recalls`, `memory_injections`,
`memory_consolidation_runs`, and `memory_commitments`.

No hidden JSON sidecar files, no SQLite side-databases, and no separate
vector-database dependency (Qdrant) are introduced. The existing `pgvector`
extension is used for the optional embedding lane (Phase 4) rather than a
new dependency.

## Consequences

- Every phase that needs durable machine state adds a migration to the
  existing PostgreSQL schema, not a new storage backend.
- Degraded behavior when PostgreSQL is unreachable is a first-class,
  designed case (see ADR 0004), not an afterthought — because there is now
  exactly one database whose unavailability must be handled gracefully
  everywhere memory is read or written.
- The current fixed `message_embeddings vector(1536)` table must not be
  reused for semantic memory embeddings — a new, explicitly identified table
  (provider + model + dimensions + chunking settings) is required so index
  identity is never ambiguous.
