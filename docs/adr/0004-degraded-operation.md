# ADR 0004: Every memory dependency has a defined degraded mode

## Status

Accepted (Stage One, Phase 0). Enforced incrementally as each dependency is
introduced (PostgreSQL indexing in Phase 3, embeddings in Phase 4).

## Context

The target memory system depends on PostgreSQL (Phase 3+) and, optionally,
an embedding provider (Phase 4). Both can be unavailable: PostgreSQL can be
down or unreachable; an embedding provider can be misconfigured, rate
limited, or simply not enabled. The three source analyses all flagged the
same risk: silently falling back to a degraded mode that looks identical to
the healthy mode is worse than an explicit, visible one, because a user
cannot tell why recall got worse.

## Decision

Every dependency gets an explicit, testable degraded mode, never a silent
fallback:

- **PostgreSQL unreachable:** capture remains durable in the append-only
  local journal (Phase 2); retrieval degrades to the existing bounded
  in-process lexical scan, with a visible warning in `memory status` and in
  any tool result that used the degraded path. This is documented as *the*
  fallback for an unreachable database — not a silent substitute for a
  *configured* database that is failing for another reason (e.g. schema
  drift), which must surface as an error, not a silent scan.
- **Embedding provider unavailable:** the last valid vector index is
  retained and lexical search continues with a visible warning (Phase 4).
  Embeddings are never a hard dependency for retrieval to function at all.
- **LLM provider unavailable** (for extraction/consolidation): capture jobs
  retry with backoff and remain inspectable (`memory jobs`, Phase 1/2); they
  do not block foreground turns, and a stalled job queue is visible in
  `memory status --deep`.

## Consequences

- `MemoryService.status()` must be able to report, per dependency, one of:
  healthy, degraded (with reason), or unavailable — and every degraded state
  must have a concrete repair command (`memory reindex`, `memory reconcile`,
  etc.).
- Tests for each phase must include the corresponding outage scenario (see
  the plan's "Testing and evaluation strategy" section) — a phase is not
  done until its degraded mode is covered, not only its happy path.
- No phase may introduce a dependency whose unavailability has undefined
  behavior; if a degraded mode cannot be defined for a proposed dependency,
  that is a signal the dependency should not be added yet.
