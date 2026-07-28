# ADR 0001: Markdown is canonical; every memory index is derived

## Status

Accepted (Stage One, Phase 0).

## Context

Minion Assist's memory implementation plan
(`minion-assist-docs/improve/memory-implementation-plan.md`) introduces a
PostgreSQL-backed lexical index (Phase 3), optional embeddings (Phase 4), an
entity index (Phase 1), and later a learned co-selection graph (Stage Two,
Phase 11). Each of these is a real, useful piece of infrastructure — but each
is also a second copy of information that already lives in a human-readable
Markdown file.

If any of these derived structures were treated as a second source of truth,
two problems follow: (1) a user editing a Markdown file by hand would not
know whether their edit "took" until some background reindex ran, and (2) a
corrupted or incompatible index (schema change, provider change, crash
mid-write) would have no well-defined recovery path other than restoring a
database backup — which may not exist, may be stale, or may not match the
current Markdown state at all.

## Decision

Markdown files under each agent's `workspaces/{agent_id}/` root (see ADR
0003) are the **canonical** representation of all durable memory. Every
other structure — PostgreSQL `tsvector`/GIN indexes, chunk tables, entity
links, embeddings, and any future learned graph — is **derived** and must be
fully reconstructable from Markdown plus transcript evidence via a `reindex`
operation, with no external database backup required.

Concretely, this means:

- Every derived table stores a `content_hash` of the Markdown it was built
  from, so staleness is detectable.
- `reindex --force` must produce byte-for-byte reproducible index content
  from the same Markdown input (Phase 3's shadow-table rebuild).
- A user can delete the entire PostgreSQL database and lose nothing but
  search speed and semantic recall — not knowledge itself.
- Direct human edits to a Markdown file are visible without a restart
  (Phase 1 acceptance criterion) once the file watcher / reconciliation
  (Phase 3) picks them up.

## Consequences

- Every write path (capture, consolidation, migration) must write Markdown
  first and treat index updates as a followup step, not a combined
  transaction with the file write.
- Index code must be written defensively: it is disposable, and tests should
  routinely delete it and rebuild from Markdown to catch drift.
- This rules out ever storing memory content only as a vector payload (the
  behavior rejected from Mem0 in `mem0-memory.md`'s decision table).
