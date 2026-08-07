"""PostgreSQL-backed session and message store with FTS via tsvector.

Optional — disabled when DATABASE_URL is not set.  Call SessionDB(url) directly
or let minion.py create it from config; when unavailable the rest of the code
treats db=None and silently skips persistence.

Schema (created automatically on first connect):
  sessions(id, agent_id, source, started_at, last_active, turn_count, title, parent_id)
  messages(id BIGSERIAL, session_id, role, content, tool_name, timestamp,
           search_vector tsvector GENERATED ALWAYS AS ...)
  message_embeddings(message_id, model_identity, embedding vector(N)) —
           only when pgvector AND an embedding provider are configured
           (MEM-GAP-006). Keyed by (message_id, model_identity) so a
           model/dimension change adds new rows under a new identity
           rather than requiring a destructive migration — same shape as
           memory/postgres_index.py's memory_chunk_embeddings.
  message_embedding_jobs(id, agent_id, session_id, message_id,
           idempotency_key, state, attempts, run_after, last_error,
           created_at, updated_at) — durable embedding queue, one job per
           message (MEM-GAP-006), structurally identical to
           memory_capture_jobs/memory_commitment_jobs
  message_mirrors(session_id, event_id, message_id, mirrored_at) — idempotency
           ledger for mirroring (Stage One Phase 2, slice A)
  memory_capture_jobs(id, agent_id, session_id, source_from_message_id,
           source_to_message_id, idempotency_key, state, attempts, run_after,
           last_error, created_at, updated_at) — durable extraction queue
           (Stage One Phase 2, slice C)
  memory_proposals(id, job_id, session_id, agent_id, claim_text, created_at,
           status) — structured, unreviewed extraction output (Stage One
           Phase 2, slice C); status (Stage One Phase 5, slice B) defaults
           to "pending" until a later slice's review flow assigns
           "promoted"/"rejected"/"superseded". job_id is nullable and
           ON DELETE SET NULL (R2-GAP-001) — pruning the originating
           capture job detaches it but never deletes the proposal itself;
           session_id (added alongside that fix) is the reliable way to
           find a session's proposals regardless of whether its job row
           still exists.
  session_job_coverage_ranges(id, agent_id, session_id, lane, channel,
           from_id, to_id, created_at) — append-only record of every
           capture/commitment range ever successfully enqueued (lane in
           ('capture', 'commitment') — message_embedding coverage is
           instead checked directly against message_embeddings, see its
           method docstring). Added by R2-GAP-002, superseding R2-GAP-001's
           single-high-water-mark session_job_coverage table: exact ranges,
           not just a cutoff, let gap detection find a sparse hole below
           the overall maximum, not only a tail gap — and retention pruning
           the job tables still can't erase this, since nothing ever
           deletes from it.
  deletion_tombstones(agent_id, session_id, requested_at, jsonl_deleted,
           db_deleted, evidence_cleaned, proposal_ids, completed_at) —
           one row per ``/delete-session`` attempt (R2-GAP-007), tracking
           which of its three cross-store cleanup phases have completed so
           a partial failure is discoverable and resumable
           (``minion-assist memory verify-deletions``) instead of silently
           stuck. Never deleted once written, even after completion —
           itself the durable proof a session was fully, deliberately
           removed.
  memory_commitment_jobs(id, agent_id, session_id, channel,
           source_from_message_id, source_to_message_id, idempotency_key,
           state, attempts, run_after, last_error, created_at, updated_at)
           — durable commitment-extraction queue, same shape as
           memory_capture_jobs plus a channel column (Stage One Phase 6,
           slice B)
  commitments(id, agent_id, session_id, channel, kind, sensitivity,
           source, status, reason, suggested_text, dedupe_key, confidence,
           due_earliest, due_latest, source_job_id, created_at, updated_at,
           sent_at, dismissed_at) — inferred, short-lived social
           follow-ups (Stage One Phase 6, slice B)

Durable capture jobs (Stage One Phase 2, slice C)
----------------------------------------------------
Replaces the per-turn daemon-thread extractor (``memory/extractor.py``) when
a database is configured: rather than firing a new thread every turn,
``agents/session.py`` enqueues a ``memory_capture_jobs`` row referencing the
PostgreSQL message ID range to extract from (using the IDs
:meth:`mirror_message` already returns — slice A). A single long-running
:class:`~minion_assist.memory.capture_worker.CaptureWorker` (started once at
process startup, not per turn) polls for due jobs, extracts facts via the
provider, and records them as ``memory_proposals`` — structured, unreviewed
claims, deliberately *not* written into a note file directly (Stage One
Phase 5 will decide how/whether to promote them; see
``minion-assist-docs/improve/memory-implementation-plan.md``).

Idempotency: :meth:`enqueue_capture_job`'s ``idempotency_key`` includes the
agent, session, message ID range, extraction prompt version, and model
(``memory/extractor.py``'s ``_EXTRACTION_PROMPT_VERSION`` +
``AgentSession``'s configured ``model_id``) — re-enqueuing the same range
under the same prompt/model is a no-op; changing either produces a new key
and a fresh extraction.

Degraded mode: when no database is configured, ``agents/session.py`` keeps
firing the original per-turn daemon-thread extractor unchanged (including
its 50-entry rolling cap) — there is no durable queue to fall back to
without a database (see ``docs/adr/0004-degraded-operation.md``).

Idempotent mirroring (Stage One Phase 2, slice A)
---------------------------------------------------
``add_message()`` alone has no notion of "already mirrored" — calling it
twice with the same content creates two rows. :meth:`mirror_message` adds
that: every mirror attempt is keyed by ``(session_id, event_id)``, where
``event_id`` is a message's stable identity (``messages.py``'s
``EVENT_ID_KEY`` / ``ensure_event_id()`` — a UUID embedded in the message
dict, assigned once and persisted to JSONL from then on). Calling
``mirror_message()`` again with the same ``(session_id, event_id)`` is a
no-op, not a duplicate insert.

This replaces the old ``replay_jsonl()``, which skipped an entire session
the moment it found *any* row already mirrored for that session — meaning a
crash after mirroring 2 of 20 messages left the other 18 unmirrored
forever, silently. :meth:`reconcile_session` / :meth:`reconcile_all_sessions`
mirror exactly the messages still missing, every time they run, so a partial
mirror from any prior crash is always completed rather than left behind.

Concurrency note: :meth:`mirror_message` checks-then-inserts without wrapping
both steps in one transaction. This is safe under this codebase's actual
concurrency model — a session's messages are only ever mirrored from that
session's own ``AgentSession.send()``, which is serialized by
``AgentSession._lock`` — but would need a real transaction (or an
``INSERT ... ON CONFLICT`` upsert) if two processes ever mirrored the same
session concurrently.

Durable message-embedding jobs (MEM-GAP-006)
------------------------------------------------
Same durable-queue shape as capture/commitment jobs above, but one job per
message rather than one job per exchange range: ``agents/session.py``
enqueues a ``message_embedding_jobs`` row for each newly-mirrored
user/assistant message (only when ``config.embeddings`` is configured — see
``config.py``'s ``EmbeddingConfig``). A single
:class:`~minion_assist.memory.message_embedding_worker.MessageEmbeddingWorker`
polls for due jobs, embeds the message's content via the configured
:class:`~minion_assist.providers.embeddings.EmbeddingProvider`, and stores
the vector in ``message_embeddings`` keyed by ``(message_id, model_identity)``
— never per-turn synchronous embedding, for the same "don't block the turn on
a network call" reason ``memory/extractor.py``'s capture pipeline uses a
worker instead of an inline call.

Per-message rather than per-range idempotency key
(``"{agent}:{message_id}:{model_identity}"``): re-enqueuing the same message
under the same embedding model/endpoint is a no-op; switching models (a new
``model_identity``) produces a new key and a fresh embedding, without
disturbing any embedding already stored under the old identity — the same
multi-model-coexistence design ``memory_chunk_embeddings`` uses, which is
what makes changing the configured embedding model or its dimensions safe
rather than requiring a destructive migration.

Retrieval: :meth:`hybrid_search_messages` reciprocal-rank-fuses this vector
lane with :meth:`search_messages`'s existing FTS lexical lane (reusing
``memory/postgres_index.py``'s ``_reciprocal_rank_fusion``), and degrades to
FTS-only automatically when no embedding provider is configured — see
``tools/session_search.py``'s ``SessionSearchTool``.
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from ..messages import EVENT_ID_KEY, ensure_event_id
from ..schema_migrations import Migration, run_migrations

if TYPE_CHECKING:
    from ..providers.embeddings import EmbeddingProvider

# Thread-local connection cache — one psycopg connection per OS thread.
_local = threading.local()


def _migration_001_baseline(conn) -> None:
    """Baseline migration (MEM-GAP-010): every table/index SessionDB needs.

    This is exactly what ``_ensure_schema()`` unconditionally ran before
    versioned migrations existed — every statement here is idempotent, so
    replaying it against an existing database (retroactively marking it
    "at version 1" the first time this code runs) is always safe. Do not
    edit this function once it has shipped: add a new, higher-numbered
    migration instead — :func:`~minion_assist.schema_migrations.run_migrations`
    will refuse to start if this function's source ever changes after being
    recorded in the ``schema_migrations`` ledger.

    The pgvector-dependent ``message_embeddings`` table/index are wrapped in
    their own ``try``/``except`` (safe under this class's autocommit
    connections — one failed statement doesn't poison the others) rather
    than a pre-checked flag, since this function has no access to
    ``SessionDB._has_vector`` — it only ever receives a raw connection.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            agent_id    TEXT NOT NULL,
            source      TEXT NOT NULL DEFAULT 'cli',
            started_at  DOUBLE PRECISION NOT NULL,
            last_active DOUBLE PRECISION NOT NULL,
            turn_count  INTEGER NOT NULL DEFAULT 0,
            title       TEXT,
            parent_id   TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS sessions_agent_idx ON sessions (agent_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS sessions_last_active_idx ON sessions (last_active DESC)"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id           BIGSERIAL PRIMARY KEY,
            session_id   TEXT NOT NULL,
            role         TEXT NOT NULL,
            content      TEXT,
            tool_name    TEXT,
            timestamp    DOUBLE PRECISION NOT NULL,
            search_vector tsvector GENERATED ALWAYS AS (
                to_tsvector('english',
                    coalesce(content, '') || ' ' || coalesce(tool_name, ''))
            ) STORED
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS messages_fts_idx ON messages USING GIN (search_vector)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS messages_session_idx ON messages (session_id)"
    )
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS message_embeddings (
                message_id BIGINT PRIMARY KEY,
                embedding  vector(1536) NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS msg_emb_hnsw
            ON message_embeddings USING hnsw (embedding vector_cosine_ops)
        """)
    except Exception:
        pass

    # Idempotency ledger for mirroring — see the module docstring's
    # "Idempotent mirroring" section.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS message_mirrors (
            session_id  TEXT NOT NULL,
            event_id    TEXT NOT NULL,
            message_id  BIGINT NOT NULL,
            mirrored_at DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (session_id, event_id)
        )
    """)

    # Durable capture-job queue — see the module docstring's "Durable
    # capture jobs" section.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_capture_jobs (
            id                     BIGSERIAL PRIMARY KEY,
            agent_id               TEXT NOT NULL,
            session_id             TEXT NOT NULL,
            source_from_message_id BIGINT NOT NULL,
            source_to_message_id   BIGINT NOT NULL,
            idempotency_key        TEXT NOT NULL UNIQUE,
            state                  TEXT NOT NULL DEFAULT 'pending',
            attempts               INTEGER NOT NULL DEFAULT 0,
            run_after              DOUBLE PRECISION NOT NULL,
            last_error             TEXT,
            created_at             DOUBLE PRECISION NOT NULL,
            updated_at             DOUBLE PRECISION NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS capture_jobs_pending_idx
        ON memory_capture_jobs (state, run_after)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_proposals (
            id              BIGSERIAL PRIMARY KEY,
            job_id          BIGINT NOT NULL,
            agent_id        TEXT NOT NULL,
            claim_text      TEXT NOT NULL,
            created_at      DOUBLE PRECISION NOT NULL,
            status          TEXT NOT NULL DEFAULT 'pending',
            rejected_reason TEXT NOT NULL DEFAULT ''
        )
    """)
    # status added in Stage One Phase 5, slice B; rejected_reason in
    # slice D — both ADD COLUMN IF NOT EXISTS so a database that
    # already has this table from an earlier slice picks them up
    # without a manual migration. status: "pending" (not yet
    # reviewed), "promoted"/"rejected"/"superseded" (assigned by
    # MemoryConsolidator's approve()/reject()/rollback()).
    # rejected_reason: the human-readable reason passed to reject() —
    # cleared back to "" on any other status transition (approve,
    # rollback) so it can never linger as a stale explanation for a
    # decision that's since been reversed.
    conn.execute(
        "ALTER TABLE memory_proposals ADD COLUMN IF NOT EXISTS "
        "status TEXT NOT NULL DEFAULT 'pending'"
    )
    conn.execute(
        "ALTER TABLE memory_proposals ADD COLUMN IF NOT EXISTS "
        "rejected_reason TEXT NOT NULL DEFAULT ''"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS proposals_agent_idx ON memory_proposals (agent_id)"
    )

    # Durable commitment-extraction job queue — Stage One Phase 6, slice
    # B. Structurally identical to memory_capture_jobs above (same
    # claim/complete/fail lifecycle, same idempotency-key mechanism),
    # but a separate table/queue: commitment output (kind, sensitivity,
    # due window, confidence, dedupe key) is a completely different
    # shape than memory_proposals' plain claim strings, so the two
    # pipelines stay cleanly separated rather than overloading one
    # table/worker with two incompatible output schemas. The extra
    # `channel` column (e.g. a Matrix room id, or "cli" outside Matrix)
    # is what memory_capture_jobs never needed — commitments must be
    # scoped to "exact agent and channel context" (the plan's Task 4).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_commitment_jobs (
            id                     BIGSERIAL PRIMARY KEY,
            agent_id               TEXT NOT NULL,
            session_id             TEXT NOT NULL,
            channel                TEXT NOT NULL,
            source_from_message_id BIGINT NOT NULL,
            source_to_message_id   BIGINT NOT NULL,
            idempotency_key        TEXT NOT NULL UNIQUE,
            state                  TEXT NOT NULL DEFAULT 'pending',
            attempts               INTEGER NOT NULL DEFAULT 0,
            run_after              DOUBLE PRECISION NOT NULL,
            last_error             TEXT,
            created_at             DOUBLE PRECISION NOT NULL,
            updated_at             DOUBLE PRECISION NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS commitment_jobs_pending_idx
        ON memory_commitment_jobs (state, run_after)
    """)

    # Commitments — inferred, short-lived social follow-ups (Stage One
    # Phase 6, Task 3). Scaled down from OpenClaw's real
    # src/commitments/types.ts CommitmentRecord (verified against that
    # source, not just the plan doc): same kind/sensitivity/source/
    # status vocabulary and due-window shape, without OpenClaw's
    # richer accountId/to/threadId/senderId scope fields minion-assist
    # has no equivalent concept for yet.
    #
    # due_earliest/due_latest are epoch seconds, not a single instant —
    # a window, so "check in tomorrow afternoon" has somewhere to land
    # rather than needing to resolve to one exact second.
    # dedupe_key + (agent_id, channel) is what
    # SessionDB.complete_commitment_job() upserts against, so a
    # near-identical inferred follow-up extends an existing pending
    # commitment's window rather than creating a duplicate.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS commitments (
            id             BIGSERIAL PRIMARY KEY,
            agent_id       TEXT NOT NULL,
            session_id     TEXT NOT NULL,
            channel        TEXT NOT NULL,
            kind           TEXT NOT NULL,
            sensitivity    TEXT NOT NULL,
            source         TEXT NOT NULL,
            status         TEXT NOT NULL DEFAULT 'pending',
            reason         TEXT NOT NULL,
            suggested_text TEXT NOT NULL,
            dedupe_key     TEXT NOT NULL,
            confidence     DOUBLE PRECISION NOT NULL,
            due_earliest   DOUBLE PRECISION NOT NULL,
            due_latest     DOUBLE PRECISION NOT NULL,
            source_job_id  BIGINT NOT NULL,
            created_at     DOUBLE PRECISION NOT NULL,
            updated_at     DOUBLE PRECISION NOT NULL,
            sent_at        DOUBLE PRECISION,
            dismissed_at   DOUBLE PRECISION
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS commitments_scope_idx
        ON commitments (agent_id, channel, status)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS commitments_dedupe_idx
        ON commitments (agent_id, channel, dedupe_key)
    """)


def _migration_002_drop_legacy_message_embeddings(conn) -> None:
    """Drop the dead, hard-coded ``vector(1536)`` ``message_embeddings`` table (MEM-GAP-006).

    That table was created by ``_migration_001_baseline`` but never had a
    writer or reader anywhere in the codebase — the gap analysis confirmed
    this directly (``repository search finds no production writer/read
    lane``). Safe to drop unconditionally: there is no real data to lose.
    ``_ensure_schema()`` recreates it right after migrations run, this time
    parameterized by the configured embedding model's actual dimensions and
    keyed by ``(message_id, model_identity)`` — the same
    dimension-parameterized, self-contained-outside-the-ledger shape
    ``memory/postgres_index.py``'s ``memory_chunk_embeddings`` already uses,
    for the same reason: a ``vector(N)`` column's width can't be a bind
    parameter, so it can't live inside a migration whose whole point is
    frozen, checksummed source.

    Runs exactly once per database (recorded in the ``schema_migrations``
    ledger like any other migration) — a fresh database that never had the
    old table simply drops nothing, harmlessly.
    """
    conn.execute("DROP TABLE IF EXISTS message_embeddings")


def _migration_003_message_embedding_jobs(conn) -> None:
    """Durable one-job-per-message embedding queue (MEM-GAP-006).

    Structurally identical to ``memory_capture_jobs``/``memory_commitment_jobs``
    (same claim/complete/fail lifecycle via ``FOR UPDATE SKIP LOCKED``), but
    ``message_id`` instead of a ``(source_from, source_to)`` range — each job
    embeds exactly one message, so there is no range to express. See the
    module docstring's "Durable message-embedding jobs" section.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS message_embedding_jobs (
            id               BIGSERIAL PRIMARY KEY,
            agent_id         TEXT NOT NULL,
            session_id       TEXT NOT NULL,
            message_id       BIGINT NOT NULL,
            idempotency_key  TEXT NOT NULL UNIQUE,
            state            TEXT NOT NULL DEFAULT 'pending',
            attempts         INTEGER NOT NULL DEFAULT 0,
            run_after        DOUBLE PRECISION NOT NULL,
            last_error       TEXT,
            created_at       DOUBLE PRECISION NOT NULL,
            updated_at       DOUBLE PRECISION NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS message_embedding_jobs_pending_idx
        ON message_embedding_jobs (state, run_after)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS message_embedding_jobs_session_idx
        ON message_embedding_jobs (agent_id, session_id)
    """)


def _add_constraint_if_missing(conn, table: str, name: str, definition: str) -> None:
    """Run ``ALTER TABLE {table} ADD CONSTRAINT {name} {definition}``, tolerating "already exists".

    PostgreSQL has no ``ADD CONSTRAINT IF NOT EXISTS`` syntax (verified
    directly against this project's PostgreSQL 16 instance — it raises a
    plain ``SyntaxError``), unlike ``CREATE TABLE``/``CREATE INDEX``. This
    is the same try/except-tolerates-"already there" shape
    ``_migration_001_baseline`` already uses for the pgvector extension, so
    that re-running this migration's body (e.g. after a crash mid-migration,
    before the ledger row was written — see ``schema_migrations.py``) is
    always safe.
    """
    try:
        conn.execute(f"ALTER TABLE {table} ADD CONSTRAINT {name} {definition}")
    except Exception:
        pass


def _migration_004_referential_integrity(conn) -> None:
    """Foreign keys and enumerated-state CHECK constraints (MEM-GAP-011).

    Before this migration, every cross-table reference in this schema
    (``messages.session_id``, ``memory_capture_jobs.session_id``, etc.) was
    a plain unconstrained column — a session row could be deleted while its
    messages/jobs/mirrors survived, or an application bug could write an
    unrecognized ``state``/``status`` string with nothing to catch it. Every
    relationship added here stays *within* ``SessionDB``'s own tables
    (never reaching into ``memory/postgres_index.py``'s ``kb_claims``/
    ``memory_consolidation_previews`` etc.) — see the module docstring's
    referential-integrity note for why cross-module FKs and the
    ``kb_claims`` relationship table's intentionally-dangling references
    stay out of scope here.

    Order matters: known-orphaned rows (there should be none in a healthy
    deployment; a real one was found and is exactly what this cleans up —
    see the MEM-GAP-011 changelog entry) are deleted *before* adding each
    FK, since ``VALIDATE CONSTRAINT`` fails outright if any existing row
    violates it. ``NOT VALID`` on ``ADD CONSTRAINT`` means the constraint
    is enforced for all *new* writes immediately without an initial table
    scan; the separate ``VALIDATE CONSTRAINT`` step then confirms every
    existing row already satisfies it (a lighter lock than a plain
    ``ADD CONSTRAINT`` would take, since it doesn't need to block
    concurrent writes for the scan's duration).

    Every FK here is ``ON DELETE CASCADE`` — matching, and now structurally
    guaranteeing, ``SessionDB.delete_session()``'s existing manual
    same-order cleanup (MEM-GAP-003): deleting a session's row now cascades
    to its messages/mirrors/jobs/proposals/commitments even if application
    code ever fails to delete them explicitly first.
    """
    # --- Orphan cleanup (must run before the FKs below, or VALIDATE fails) ---
    conn.execute(
        "DELETE FROM message_mirrors mm WHERE NOT EXISTS "
        "(SELECT 1 FROM sessions s WHERE s.id = mm.session_id)"
    )
    conn.execute(
        "DELETE FROM message_mirrors mm WHERE NOT EXISTS "
        "(SELECT 1 FROM messages m WHERE m.id = mm.message_id)"
    )
    conn.execute(
        "DELETE FROM messages m WHERE NOT EXISTS "
        "(SELECT 1 FROM sessions s WHERE s.id = m.session_id)"
    )
    conn.execute(
        "DELETE FROM message_embedding_jobs j WHERE NOT EXISTS "
        "(SELECT 1 FROM sessions s WHERE s.id = j.session_id)"
    )
    conn.execute(
        "DELETE FROM message_embedding_jobs j WHERE NOT EXISTS "
        "(SELECT 1 FROM messages m WHERE m.id = j.message_id)"
    )
    conn.execute(
        "DELETE FROM memory_capture_jobs j WHERE NOT EXISTS "
        "(SELECT 1 FROM sessions s WHERE s.id = j.session_id)"
    )
    conn.execute(
        "DELETE FROM memory_proposals p WHERE NOT EXISTS "
        "(SELECT 1 FROM memory_capture_jobs j WHERE j.id = p.job_id)"
    )
    conn.execute(
        "DELETE FROM memory_commitment_jobs j WHERE NOT EXISTS "
        "(SELECT 1 FROM sessions s WHERE s.id = j.session_id)"
    )
    conn.execute(
        "DELETE FROM commitments c WHERE NOT EXISTS "
        "(SELECT 1 FROM sessions s WHERE s.id = c.session_id)"
    )
    conn.execute(
        "DELETE FROM commitments c WHERE NOT EXISTS "
        "(SELECT 1 FROM memory_commitment_jobs j WHERE j.id = c.source_job_id)"
    )

    # --- Foreign keys (NOT VALID, then validated) ---
    _fk_specs = [
        ("messages", "fk_messages_session",
         "FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE NOT VALID"),
        ("message_mirrors", "fk_message_mirrors_session",
         "FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE NOT VALID"),
        ("message_mirrors", "fk_message_mirrors_message",
         "FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE NOT VALID"),
        ("message_embedding_jobs", "fk_message_embedding_jobs_session",
         "FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE NOT VALID"),
        ("message_embedding_jobs", "fk_message_embedding_jobs_message",
         "FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE NOT VALID"),
        ("memory_capture_jobs", "fk_memory_capture_jobs_session",
         "FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE NOT VALID"),
        ("memory_proposals", "fk_memory_proposals_job",
         "FOREIGN KEY (job_id) REFERENCES memory_capture_jobs(id) ON DELETE CASCADE NOT VALID"),
        ("memory_commitment_jobs", "fk_memory_commitment_jobs_session",
         "FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE NOT VALID"),
        ("commitments", "fk_commitments_session",
         "FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE NOT VALID"),
        ("commitments", "fk_commitments_job",
         "FOREIGN KEY (source_job_id) REFERENCES memory_commitment_jobs(id) ON DELETE CASCADE NOT VALID"),
    ]
    for table, name, definition in _fk_specs:
        _add_constraint_if_missing(conn, table, name, definition)
    for table, name, _ in _fk_specs:
        conn.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT {name}")

    # --- Enumerated state/status CHECK constraints ---
    # Vocabularies confirmed against actual code (grep for every literal
    # assignment), not guessed — 'superseded' is included for
    # memory_proposals.status despite no current code path writing it: the
    # module docstring documents it as a legal value assigned by a future
    # MemoryConsolidator operation, and a CHECK constraint should never be
    # stricter than what's already documented as intended.
    _check_specs = [
        ("memory_capture_jobs", "chk_memory_capture_jobs_state",
         "CHECK (state IN ('pending', 'running', 'done', 'failed')) NOT VALID"),
        ("memory_commitment_jobs", "chk_memory_commitment_jobs_state",
         "CHECK (state IN ('pending', 'running', 'done', 'failed')) NOT VALID"),
        ("message_embedding_jobs", "chk_message_embedding_jobs_state",
         "CHECK (state IN ('pending', 'running', 'done', 'failed')) NOT VALID"),
        ("memory_proposals", "chk_memory_proposals_status",
         "CHECK (status IN ('pending', 'promoted', 'rejected', 'superseded')) NOT VALID"),
        ("commitments", "chk_commitments_status",
         "CHECK (status IN ('pending', 'sent', 'dismissed', 'expired')) NOT VALID"),
    ]
    for table, name, definition in _check_specs:
        _add_constraint_if_missing(conn, table, name, definition)
    for table, name, _ in _check_specs:
        conn.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT {name}")


def _migration_005_decouple_retention_from_outputs(conn) -> None:
    """Break retention's accidental power to delete outputs and erase coverage (R2-GAP-001).

    Migration 004 made ``memory_proposals.job_id`` and
    ``commitments.source_job_id`` ``ON DELETE CASCADE`` against the job
    tables that ``prune_operational_tables()`` (MEM-GAP-015) deletes old
    rows from. Combined, those two features meant enabling retention could
    silently delete a proposal or commitment that had already been produced
    (and possibly reviewed/promoted) — exactly the "durable output" this
    module's docstring promises retention never touches. Separately, every
    ``find_uncovered_*`` reconciliation method used ``MAX(job table column)``
    as its "already covered" cursor, so pruning old job rows also reset that
    cursor back to zero — enabling retention could make reconciliation
    re-enqueue a session's entire history for extraction/embedding.

    This migration fixes both, without touching migration 004's own body
    (never edit an already-applied migration — see this file's module
    docstring and ``schema_migrations.py``):

    1. Gives ``memory_proposals`` its own ``session_id`` column (backfilled
       from its job's ``session_id`` — always resolvable here since
       migration 004's CASCADE is still in effect at the moment this runs,
       so no proposal can yet have a dangling ``job_id``). This is what
       lets :meth:`delete_session` find a session's proposals directly
       instead of walking through its (now potentially-pruned) capture
       jobs.
    2. Adds ``session_job_coverage`` — one row per ``(session_id, lane)``
       recording the highest message/range id a job has ever been
       successfully enqueued for. ``enqueue_capture_job``/
       ``enqueue_commitment_job``/``enqueue_message_embedding_job`` advance
       it going forward; ``find_uncovered_*`` reads it instead of
       ``MAX(job table)``. Retention never deletes from this table, so
       pruning old job rows can no longer erase coverage. Backfilled once
       here from the job tables' current ``MAX`` values.
    3. Flips ``fk_memory_proposals_job`` and ``fk_commitments_job`` from
       ``ON DELETE CASCADE`` to ``ON DELETE SET NULL`` — pruning a job now
       merely detaches its (already-materialized) proposal/commitment from
       the deleted operational row instead of deleting it too. Both
       ``job_id``/``source_job_id`` columns lose their ``NOT NULL``
       constraint first, since a ``SET NULL`` target column must be
       nullable.

    ``prune_operational_tables()``'s docstring is corrected in the same
    change that adds this migration — see its current text for what it now
    accurately promises.
    """
    now = time.time()

    # --- 1. memory_proposals.session_id ---
    conn.execute("ALTER TABLE memory_proposals ADD COLUMN IF NOT EXISTS session_id TEXT")
    conn.execute(
        "UPDATE memory_proposals p SET session_id = j.session_id "
        "FROM memory_capture_jobs j WHERE p.job_id = j.id AND p.session_id IS NULL"
    )
    conn.execute("ALTER TABLE memory_proposals ALTER COLUMN session_id SET NOT NULL")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS proposals_session_idx ON memory_proposals (session_id)"
    )

    # --- 2. session_job_coverage, backfilled from current job-table state ---
    # `channel` is only meaningful for the 'commitment' lane (a session's
    # commitment jobs are additionally scoped to a Matrix room id / "cli" —
    # see find_uncovered_commitment_range's docstring); NULL for the other
    # two lanes, which have no such concept.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_job_coverage (
            agent_id            TEXT NOT NULL,
            session_id          TEXT NOT NULL,
            lane                TEXT NOT NULL,
            covered_through_id  BIGINT NOT NULL,
            channel             TEXT,
            updated_at          DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (session_id, lane)
        )
    """)
    conn.execute(
        """
        INSERT INTO session_job_coverage (agent_id, session_id, lane, covered_through_id, updated_at)
        SELECT agent_id, session_id, 'capture', MAX(source_to_message_id), %s
        FROM memory_capture_jobs GROUP BY agent_id, session_id
        ON CONFLICT (session_id, lane) DO NOTHING
        """,
        (now,),
    )
    # DISTINCT ON, not GROUP BY: the coverage row needs the *channel of the
    # single most-recently-covered job*, not an aggregate — matches
    # find_uncovered_commitment_range's pre-existing "most recently covered
    # job's channel" semantics.
    conn.execute(
        """
        INSERT INTO session_job_coverage (agent_id, session_id, lane, covered_through_id, channel, updated_at)
        SELECT DISTINCT ON (session_id) agent_id, session_id, 'commitment', source_to_message_id, channel, %s
        FROM memory_commitment_jobs
        ORDER BY session_id, source_to_message_id DESC
        ON CONFLICT (session_id, lane) DO NOTHING
        """,
        (now,),
    )
    conn.execute(
        """
        INSERT INTO session_job_coverage (agent_id, session_id, lane, covered_through_id, updated_at)
        SELECT agent_id, session_id, 'message_embedding', MAX(message_id), %s
        FROM message_embedding_jobs GROUP BY agent_id, session_id
        ON CONFLICT (session_id, lane) DO NOTHING
        """,
        (now,),
    )

    # --- 3. CASCADE -> SET NULL for the two output-parent FKs ---
    conn.execute("ALTER TABLE memory_proposals ALTER COLUMN job_id DROP NOT NULL")
    conn.execute("ALTER TABLE memory_proposals DROP CONSTRAINT IF EXISTS fk_memory_proposals_job")
    _add_constraint_if_missing(
        conn, "memory_proposals", "fk_memory_proposals_job",
        "FOREIGN KEY (job_id) REFERENCES memory_capture_jobs(id) ON DELETE SET NULL NOT VALID",
    )
    conn.execute("ALTER TABLE memory_proposals VALIDATE CONSTRAINT fk_memory_proposals_job")

    conn.execute("ALTER TABLE commitments ALTER COLUMN source_job_id DROP NOT NULL")
    conn.execute("ALTER TABLE commitments DROP CONSTRAINT IF EXISTS fk_commitments_job")
    _add_constraint_if_missing(
        conn, "commitments", "fk_commitments_job",
        "FOREIGN KEY (source_job_id) REFERENCES memory_commitment_jobs(id) ON DELETE SET NULL NOT VALID",
    )
    conn.execute("ALTER TABLE commitments VALIDATE CONSTRAINT fk_commitments_job")

    # --- 4. session_id FK for defense-in-depth, matching migration 004's
    #        "structurally guarantee the manual cleanup" precedent ---
    _add_constraint_if_missing(
        conn, "memory_proposals", "fk_memory_proposals_session",
        "FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE NOT VALID",
    )
    conn.execute("ALTER TABLE memory_proposals VALIDATE CONSTRAINT fk_memory_proposals_session")


def _migration_006_range_based_job_coverage(conn) -> None:
    """Exact per-range job coverage, replacing the single high-water mark (R2-GAP-002).

    Migration 005's ``session_job_coverage`` fixed retention erasing
    coverage, but it only ever stored one ``covered_through_id`` scalar per
    ``(session_id, lane)`` — a high-water mark can represent a *tail* gap
    (nothing enqueued since X) but not a *sparse* one (message 1's enqueue
    failed, message 2's later succeeded — 1 is hidden forever, since
    anything below the mark reads as "covered"). This migration replaces it
    with an append-only table of every range a job was ever successfully
    enqueued for, so gap detection can check each message individually
    against the exact ranges recorded, not just a single cutoff.

    Two lanes, two different fixes:

    - **capture / commitment** — no way to tell "attempted, zero results"
      from "never attempted" by looking at *outputs* (an extraction that
      legitimately found nothing looks identical to one that never ran), so
      these two lanes keep needing an explicit "this range was enqueued"
      record. ``session_job_coverage_ranges`` is that record, one row per
      successful enqueue, retention-proof because nothing ever deletes from
      it (unlike the job tables retention prunes).
    - **message_embedding** — moves to a *better* fix than a coverage table
      at all (R2-GAP-005): ``message_embeddings`` is already keyed by
      ``(message_id, model_identity)`` and is itself never pruned, so
      ``find_uncovered_message_ids_for_embedding`` can anti-join against it
      directly for the *currently active* model — simultaneously exact
      (no sparse-gap blind spot) and model-aware (a model switch now
      correctly re-surfaces every historical message as needing a new
      embedding, instead of looking permanently "covered" by the old
      model's job history).

    Backfill prefers the job tables themselves (still fully intact at
    migration time) over ``session_job_coverage``'s already-collapsed
    high-water mark — one row per existing job preserves whatever sparse
    structure hasn't been pruned yet, which is strictly more information
    than a single cutoff. A session whose job rows were *already* pruned by
    a production deployment that had retention enabled before this
    migration ran (not this project's own deployment — retention has never
    been applied here) falls back to a single synthetic ``(1,
    covered_through_id)`` range from the old table, so no session regresses
    to "never covered."

    ``session_job_coverage`` (migration 005) is dropped at the end — fully
    superseded on both lanes, one commit after it was introduced.
    """
    now = time.time()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_job_coverage_ranges (
            id          BIGSERIAL PRIMARY KEY,
            agent_id    TEXT NOT NULL,
            session_id  TEXT NOT NULL,
            lane        TEXT NOT NULL,
            channel     TEXT,
            from_id     BIGINT NOT NULL,
            to_id       BIGINT NOT NULL,
            created_at  DOUBLE PRECISION NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS session_job_coverage_ranges_lookup_idx "
        "ON session_job_coverage_ranges (session_id, lane)"
    )

    conn.execute(
        """
        INSERT INTO session_job_coverage_ranges
            (agent_id, session_id, lane, channel, from_id, to_id, created_at)
        SELECT agent_id, session_id, 'capture', NULL,
               source_from_message_id, source_to_message_id, %s
        FROM memory_capture_jobs
        """,
        (now,),
    )
    conn.execute(
        """
        INSERT INTO session_job_coverage_ranges
            (agent_id, session_id, lane, channel, from_id, to_id, created_at)
        SELECT agent_id, session_id, 'commitment', channel,
               source_from_message_id, source_to_message_id, %s
        FROM memory_commitment_jobs
        """,
        (now,),
    )
    # Fallback only for a (session, lane) with no surviving job rows at all
    # (already pruned) but a remembered high-water mark — everything else
    # was already backfilled above with better fidelity.
    conn.execute(
        """
        INSERT INTO session_job_coverage_ranges
            (agent_id, session_id, lane, channel, from_id, to_id, created_at)
        SELECT c.agent_id, c.session_id, c.lane, c.channel, 1, c.covered_through_id, %s
        FROM session_job_coverage c
        WHERE c.lane IN ('capture', 'commitment')
        AND NOT EXISTS (
            SELECT 1 FROM session_job_coverage_ranges r
            WHERE r.session_id = c.session_id AND r.lane = c.lane
        )
        """,
        (now,),
    )

    conn.execute("DROP TABLE IF EXISTS session_job_coverage")


def _migration_007_deletion_tombstones(conn) -> None:
    """Track ``/delete-session``'s multi-phase cross-store cleanup so it's resumable (R2-GAP-007).

    ``/delete-session`` deletes a session's JSONL file, then its
    ``SessionDB`` rows, then (if any proposals were found) their indexed
    evidence in ``PostgresMemoryIndex`` — three independent operations
    against three different stores, none wrapped in a shared transaction
    (can't be: a filesystem delete and two separate database connections).
    Before this migration, a failure partway through (the DB delete raises
    on a transient connection drop, say) left no record of which phases
    had actually completed — the only way to notice was reading a stack
    trace at the moment it happened, and the only way to "retry" was
    re-running ``/delete-session`` on the same target, which no longer
    works once the JSONL file is already gone (it can't be found by
    listing/index again).

    One row per deletion attempt, keyed by ``(agent_id, session_id)``,
    updated as each phase completes. ``minion-assist memory
    verify-deletions [--retry]`` (``memory/cli.py``) lists/finishes any
    tombstone whose ``completed_at`` is still ``NULL`` — the durable record
    is what makes an incomplete deletion discoverable and resumable without
    depending on the file that's the whole reason a retry might be needed.

    ``proposal_ids`` (an array, not a foreign key) is deliberately a plain
    snapshot of whatever :meth:`SessionDB.delete_session` returned the
    moment the DB phase completed — by the time evidence cleanup might
    need retrying, the ``memory_proposals`` rows those ids referred to are
    already gone (deleted in that same phase), so this is a record of
    *which evidence chunks to remove elsewhere*, not a live reference.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS deletion_tombstones (
            agent_id          TEXT NOT NULL,
            session_id        TEXT NOT NULL,
            requested_at      DOUBLE PRECISION NOT NULL,
            jsonl_deleted     BOOLEAN NOT NULL DEFAULT FALSE,
            db_deleted        BOOLEAN NOT NULL DEFAULT FALSE,
            evidence_cleaned  BOOLEAN NOT NULL DEFAULT FALSE,
            proposal_ids      BIGINT[] NOT NULL DEFAULT '{}',
            completed_at      DOUBLE PRECISION,
            PRIMARY KEY (agent_id, session_id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS deletion_tombstones_incomplete_idx "
        "ON deletion_tombstones (requested_at) WHERE completed_at IS NULL"
    )


def _migration_008_language_neutral_fts(conn) -> None:
    """Switch ``messages.search_vector`` from hardcoded 'english' to 'simple' (R2-GAP-012).

    Migration 001's baseline generated ``search_vector`` with PostgreSQL's
    ``'english'`` text-search configuration — English stemming (e.g.
    "running" matches "run") and English stopword removal, hardcoded with
    no way for a non-English deployment to get comparable lexical recall.
    ``'simple'`` is PostgreSQL's built-in language-neutral configuration:
    pure tokenization and lowercasing, no stemming, no stopword removal,
    for any language using whitespace/punctuation-delimited words — a
    genuine trade for English users (losing stemming) in exchange for
    every other language getting real, non-broken lexical search instead
    of none at all.

    A generated column's expression can't be altered in place
    (``ALTER COLUMN ... SET EXPRESSION`` doesn't exist in PostgreSQL) —
    dropping and recreating it is the only way, which also drops its GIN
    index (recreated here too). This is a full table rewrite; acceptable
    as a one-time migration cost, unlike an operation this project would
    ever want to repeat on a schedule.

    Application code's own ``to_tsvector('english', ...)``/
    ``websearch_to_tsquery('english', ...)``/``ts_headline('english', ...)``
    calls (``search_messages()``) are updated to ``'simple'`` in the same
    change that adds this migration — they must always match whatever
    config built ``search_vector``, or FTS matching breaks entirely
    (a stored-with-stemming column queried with a non-stemmed term, or
    vice versa, simply won't align).
    """
    conn.execute("ALTER TABLE messages DROP COLUMN search_vector")
    conn.execute("""
        ALTER TABLE messages ADD COLUMN search_vector tsvector GENERATED ALWAYS AS (
            to_tsvector('simple', coalesce(content, '') || ' ' || coalesce(tool_name, ''))
        ) STORED
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS messages_fts_idx ON messages USING GIN (search_vector)"
    )


# Every migration SessionDB knows about, in the order they were introduced.
# Append new migrations here — never edit an existing entry's `apply`
# function once it has shipped (see _migration_001_baseline's docstring).
_SESSION_DB_MIGRATIONS = [
    Migration(1, "baseline", _migration_001_baseline),
    Migration(2, "drop_legacy_message_embeddings", _migration_002_drop_legacy_message_embeddings),
    Migration(3, "message_embedding_jobs", _migration_003_message_embedding_jobs),
    Migration(4, "referential_integrity", _migration_004_referential_integrity),
    Migration(5, "decouple_retention_from_outputs", _migration_005_decouple_retention_from_outputs),
    Migration(6, "range_based_job_coverage", _migration_006_range_based_job_coverage),
    Migration(7, "deletion_tombstones", _migration_007_deletion_tombstones),
    Migration(8, "language_neutral_fts", _migration_008_language_neutral_fts),
]


def _record_job_coverage_range(
    conn, agent_id: str, session_id: str, lane: str, from_id: int, to_id: int,
    channel: str | None = None,
) -> None:
    """Append one covered range to ``session_job_coverage_ranges`` (R2-GAP-002).

    Called by ``enqueue_capture_job``/``enqueue_commitment_job`` right
    after a genuinely new job row is inserted — never on their
    ``ON CONFLICT ... DO NOTHING`` no-op path, since a job that already
    existed already recorded its range the first time it was enqueued.
    Deliberately append-only (unlike migration 005's superseded
    high-water-mark table, which upserted): every enqueue's exact range
    stays individually visible, which is what lets gap detection find a
    sparse hole below the overall maximum, not just a tail gap.

    Args:
        conn: An open connection (this class's autocommit connection).
        agent_id: The agent this session belongs to.
        session_id: The session this job's range belongs to.
        lane: ``"capture"`` or ``"commitment"`` — ``message_embedding``
            never calls this (see :meth:`SessionDB.find_uncovered_message_ids_for_embedding`'s
            docstring for why that lane doesn't need a coverage table).
        from_id: Lowest message id this job's range covers.
        to_id: Highest message id this job's range covers.
        channel: Only meaningful for the ``"commitment"`` lane (a Matrix
            room id, or ``"cli"``) — ``None`` for ``"capture"``.
    """
    conn.execute(
        "INSERT INTO session_job_coverage_ranges "
        "(agent_id, session_id, lane, channel, from_id, to_id, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (agent_id, session_id, lane, channel, from_id, to_id, time.time()),
    )


# Fixed message-count window for one backfilled/reconciled capture or
# commitment job (moved here from memory/consolidation.py so
# find_uncovered_capture_ranges/find_uncovered_commitment_ranges below can
# share it without a circular import — memory/consolidation.py already
# imports SessionDB from this module, so the reverse direction isn't
# available). Bounds how much history a single extract_facts() call ever
# receives — a live per-turn job only ever covers one exchange (2
# messages), so an unbounded gap window could dwarf that by orders of
# magnitude and risk exceeding the extraction model's context. Not
# configurable (yet) — same "no knob without evaluation data" reasoning as
# capture_worker.py's own hardcoded constants.
_BACKFILL_WINDOW_MESSAGES = 20

# R2-GAP-011: how many of a session's most-recent eligible messages
# (find_uncovered_capture_ranges/find_uncovered_commitment_ranges/
# find_uncovered_message_ids_for_embedding) or most-recent eligible
# candidate rows (find_uncovered_message_ids_for_embedding's anti-join)
# one reconciliation pass will scan. Without a bound, a lifelong session
# with a very large message history would have its *entire* history
# re-scanned every pass, purely to re-confirm coverage that essentially
# never changes for old messages. A real gap almost always appears near
# the tail (a turn that just happened failed to enqueue) and reconciliation
# runs frequently, so bounding to the most recent slice trades "detect an
# arbitrarily old gap immediately" for "keep every pass's cost bounded
# regardless of session size" — a deliberate choice, not a full fix for
# every theoretical case (an old gap from a reconciliation *outage* could
# still be missed; this is a known, accepted limitation, not claimed as
# complete). backfill_agent's manual one-off CLI backfill deliberately
# stays unbounded (list_message_ids, not this limit) since it's rare and
# explicit, not run automatically on a timer.
_RECONCILIATION_SCAN_LIMIT = 2000

# Sentinel distinguishing "caller didn't pass limit, use the current
# _RECONCILIATION_SCAN_LIMIT" from "caller explicitly passed None for
# unbounded" — see _eligible_message_ids's docstring.
_UNSET = object()


def _chunk_run(run: list[int]) -> list[tuple[int, int]]:
    """Split one contiguous run of uncovered message ids into bounded (from, to) windows."""
    return [
        (chunk[0], chunk[-1])
        for chunk in (
            run[i : i + _BACKFILL_WINDOW_MESSAGES]
            for i in range(0, len(run), _BACKFILL_WINDOW_MESSAGES)
        )
    ]


def compute_backfill_windows(
    message_ids: list[int], covered_ranges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Find bounded (from_id, to_id) windows covering every uncovered message in a session.

    Shared by :meth:`SessionDB.find_uncovered_capture_ranges`/
    :meth:`SessionDB.find_uncovered_commitment_ranges` (periodic
    reconciliation, R2-GAP-002) and ``memory/consolidation.py``'s
    ``backfill_agent`` (one-off manual backfill, Stage One Phase 5, slice
    D) — one gap-finding algorithm for both, promoted here from a private
    helper local to ``consolidation.py`` once reconciliation needed the
    exact same logic rather than a second, subtly different
    implementation.

    "Contiguous" here means adjacent *by position in message_ids* (this
    session's own message order), not adjacent integers — ``messages`` is
    one shared table with a single id sequence across every session, so
    two of one session's own messages are almost never numerically
    consecutive (other sessions' messages get ids in between). Walking
    ``message_ids`` in order and tracking runs of "not covered" sidesteps
    that entirely.

    Args:
        message_ids: The candidate message ids to check, ascending. Each
            caller decides what's eligible for its own purposes —
            ``backfill_agent`` passes every message id in a session
            (:meth:`SessionDB.list_message_ids`); the ``find_uncovered_*``
            reconciliation methods pass only user/assistant messages with
            non-empty content, so a run of pure tool-call messages with no
            actual dialogue never becomes an enqueued (and pointless) job.
        covered_ranges: Every ``(from_id, to_id)`` pair a job has ever been
            enqueued for, in this session and lane (as
            :meth:`SessionDB.list_capture_job_ranges`/
            :meth:`SessionDB.list_commitment_job_ranges` return).

    Returns:
        list[tuple[int, int]]: Windows to enqueue, each at most
            :data:`_BACKFILL_WINDOW_MESSAGES` messages wide, covering
            exactly the gaps — messages already covered by an existing
            job are never included in any window.
    """
    covered: set[int] = set()
    for from_id, to_id in covered_ranges:
        covered.update(range(from_id, to_id + 1))

    windows: list[tuple[int, int]] = []
    current_run: list[int] = []
    for mid in message_ids:
        if mid in covered:
            if current_run:
                windows.extend(_chunk_run(current_run))
                current_run = []
        else:
            current_run.append(mid)
    if current_run:
        windows.extend(_chunk_run(current_run))
    return windows


def _msg_text(msg: dict) -> str | None:
    """Extract plain text from an OpenAI-format message dict."""
    content = msg.get("content")
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(parts) or None
    return None


class SessionDB:
    """Thin wrapper around a PostgreSQL database for session + message storage.

    Args:
        url: libpq connection string, e.g. ``postgresql://user:pw@host/db``.
        embedding_dimensions: The configured embedding model's vector length
            (``config.embeddings.dimensions``), or ``None`` if no embedding
            backend is configured. Same reasoning as
            ``PostgresMemoryIndex.__init__`` — a pgvector column's width is
            fixed at ``CREATE TABLE`` time, so this must be known up front.
            When ``None`` (the default), ``message_embeddings`` is never
            created and the session vector lane simply doesn't exist.
        embedding_provider: The :class:`~minion_assist.providers.embeddings.EmbeddingProvider`
            :meth:`hybrid_search_messages` calls to embed a query. ``None``
            (the default) means the vector lane stays empty even if
            ``embedding_dimensions`` created the table — search degrades to
            FTS-only, exactly as if no embedding backend were configured.
        min_similarity: R2-GAP-012 — minimum cosine similarity a vector-
            search hit must clear to be returned by
            :meth:`_vector_search_messages` at all. ``0.0`` (the default)
            preserves the old always-return-``limit``-nearest-neighbors
            behavior; callers wanting a real relevance floor pass
            ``config.session_search.min_similarity``.
    """

    def __init__(
        self,
        url: str,
        embedding_dimensions: int | None = None,
        embedding_provider: "EmbeddingProvider | None" = None,
        min_similarity: float = 0.0,
    ) -> None:
        import psycopg  # noqa: PLC0415
        self._url = url
        self._psycopg = psycopg
        self._has_vector = False
        self._embedding_dimensions = embedding_dimensions
        self._embedding_provider = embedding_provider
        self._min_similarity = min_similarity
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self):
        """Return a thread-local autocommit connection, creating it if needed.

        Registers pgvector's psycopg adapter on every new connection when
        ``self._has_vector`` is set — without it, a ``vector`` column
        round-trips as a raw string instead of a usable list of floats. Same
        pattern as ``PostgresMemoryIndex._conn()``.
        """
        conn = getattr(_local, "conn", None)
        if conn is None or conn.closed:
            conn = self._psycopg.connect(self._url, autocommit=True)
            if self._has_vector:
                from pgvector.psycopg import register_vector  # noqa: PLC0415

                register_vector(conn)
            _local.conn = conn
        return conn

    @property
    def has_vector_lane(self) -> bool:
        """``True`` if pgvector is available AND an embedding model is configured.

        Mirrors ``PostgresMemoryIndex.has_vector_lane`` — used by callers
        (the message-embedding enqueue site, the worker, the vector search
        lane) to no-op cleanly rather than each re-deriving this condition.
        """
        return self._has_vector and bool(self._embedding_dimensions)

    @property
    def embedding_model_identity(self) -> str | None:
        """The configured embedding provider's identity string, or ``None`` if unconfigured.

        Exposes :attr:`EmbeddingProvider.model_identity` without callers
        (``agents/session.py``'s enqueue site, ``ReconciliationScheduler``)
        needing to reach into ``self._embedding_provider`` directly — used
        to build a message-embedding job's idempotency key.
        """
        return self._embedding_provider.model_identity if self._embedding_provider else None

    def _ensure_schema(self) -> None:
        conn = self._conn()
        # pgvector extension (optional — silently skipped if not available).
        # Kept here rather than inside a migration: this mutates self, which
        # a migration function (checksummed, self-less by design — see
        # schema_migrations.py) has no access to. Registered on *this*
        # connection immediately since _conn() only registers it for
        # connections created *after* self._has_vector becomes True.
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._has_vector = True
            from pgvector.psycopg import register_vector  # noqa: PLC0415

            register_vector(conn)
        except Exception:
            pass

        run_migrations(conn, "session_db", _SESSION_DB_MIGRATIONS)

        # message_embeddings — created only when pgvector is available AND
        # an embedding backend is configured (MEM-GAP-006). Parameterized by
        # the configured model's dimensions; kept outside the checksummed
        # migration ledger for the same reason memory_chunk_embeddings is —
        # see _migration_002_drop_legacy_message_embeddings's docstring.
        # Keyed by (message_id, model_identity), NOT message_id alone: a
        # model/dimension change adds new rows under a new identity instead
        # of requiring a destructive migration.
        if self._has_vector and self._embedding_dimensions:
            dims = int(self._embedding_dimensions)
            # Self-healing: if message_embeddings already exists with a
            # DIFFERENT vector width than currently configured, drop and
            # recreate rather than silently keep serving the old width
            # until some later real embed() call fails with a dimension
            # mismatch (a real production scenario if embeddings.dimensions
            # ever changes in config.json — and, found via R2-GAP-015's own
            # test-isolation work, also a real *test* scenario: two
            # SessionDB instances constructed with different
            # embedding_dimensions sharing one schema). Safe: this table is
            # a pure derived cache regenerable from `messages` plus the
            # embedding provider, never a source of truth — see this
            # table's own module-docstring entry.
            existing_dims = conn.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = to_regclass('message_embeddings') AND attname = 'embedding'"
            ).fetchone()
            if existing_dims is not None and existing_dims[0] not in (None, -1) and existing_dims[0] != dims:
                conn.execute("DROP TABLE message_embeddings")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS message_embeddings (
                    message_id     BIGINT NOT NULL,
                    model_identity TEXT NOT NULL,
                    embedding      vector({dims}) NOT NULL,
                    PRIMARY KEY (message_id, model_identity)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS message_embeddings_hnsw "
                "ON message_embeddings USING hnsw (embedding vector_cosine_ops)"
            )
            # MEM-GAP-011: same referential-integrity reasoning as the
            # checksummed migration below, applied here since this table
            # lives outside that ledger (dimension-parameterized, can't be
            # frozen source). Idempotent via _add_constraint_if_missing —
            # this whole block re-runs on every _ensure_schema() call, not
            # just once.
            _add_constraint_if_missing(
                conn, "message_embeddings", "fk_message_embeddings_message",
                "FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE NOT VALID",
            )
            conn.execute(
                "ALTER TABLE message_embeddings VALIDATE CONSTRAINT fk_message_embeddings_message"
            )

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    def upsert_session(
        self,
        session_id: str,
        agent_id: str,
        parent_id: str | None = None,
        source: str = "cli",
        started_at: float | None = None,
    ) -> None:
        """Insert a session row if it doesn't exist yet."""
        now = time.time()
        self._conn().execute(
            """
            INSERT INTO sessions (id, agent_id, source, started_at, last_active, parent_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (session_id, agent_id, source, started_at or now, now, parent_id),
        )

    def update_session(
        self,
        session_id: str,
        last_active: float | None = None,
        turn_count: int | None = None,
        title: str | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        if last_active is not None:
            sets.append("last_active = %s")
            params.append(last_active)
        if turn_count is not None:
            sets.append("turn_count = %s")
            params.append(turn_count)
        if title is not None:
            sets.append("title = %s")
            params.append(title)
        if not sets:
            return
        params.append(session_id)
        self._conn().execute(
            f"UPDATE sessions SET {', '.join(sets)} WHERE id = %s", params
        )

    def get_sessions_by_ids(self, session_ids: list[str], agent_id: str) -> dict[str, dict]:
        """Return session metadata for ``session_ids``, filtered to ``agent_id``.

        The ``agent_id`` filter (MEM-GAP-002) means a session belonging to a
        different agent is silently omitted from the result rather than
        raising — callers (e.g. the session-search tool) treat "not in the
        dict" the same as "doesn't exist", so a cross-agent id can't be used
        to read another agent's session metadata.
        """
        if not session_ids:
            return {}
        rows = self._conn().execute(
            """
            SELECT id, agent_id, title, started_at, last_active, turn_count
            FROM sessions WHERE id = ANY(%s) AND agent_id = %s
            """,
            (session_ids, agent_id),
        ).fetchall()
        return {
            r[0]: {
                "id": r[0], "agent_id": r[1], "title": r[2],
                "started_at": r[3], "last_active": r[4], "turn_count": r[5],
            }
            for r in rows
        }

    def list_sessions(self, agent_id: str, limit: int = 20) -> list[dict]:
        """Return this agent's most recent sessions, newest first.

        ``agent_id`` is mandatory (MEM-GAP-002) so one agent's browse/search
        tool can never enumerate another agent's session history — every
        agent's memory stays private to that agent, even from other agents
        in the same deployment.
        """
        rows = self._conn().execute(
            """
            SELECT s.id, s.agent_id, s.title, s.started_at, s.last_active, s.turn_count,
                (SELECT content FROM messages
                 WHERE session_id = s.id AND role = 'user'
                 ORDER BY id LIMIT 1) AS first_message
            FROM sessions s
            WHERE s.agent_id = %s
            ORDER BY s.last_active DESC
            LIMIT %s
            """,
            (agent_id, limit),
        ).fetchall()
        return [
            {
                "id": r[0], "agent_id": r[1], "title": r[2],
                "started_at": r[3], "last_active": r[4],
                "turn_count": r[5], "first_message": r[6],
            }
            for r in rows
        ]

    def delete_session(self, agent_id: str, session_id: str) -> dict | None:
        """Delete every row this database owns for one session (MEM-GAP-003).

        Removes, in dependency order: ``message_embeddings`` (keyed by
        message id, no ``session_id`` column of its own), ``message_mirrors``,
        ``messages``, this session's ``memory_proposals`` (found by their own
        ``session_id`` column — R2-GAP-001: no longer via
        ``memory_capture_jobs``, since a proposal's originating job may have
        already been pruned by retention by the time a session is deleted),
        ``memory_capture_jobs``, ``memory_commitment_jobs``,
        ``message_embedding_jobs``, ``session_job_coverage_ranges``,
        ``commitments``, and finally the ``sessions`` row itself.

        Deliberately does **not** touch anything outside this database, and
        does **not** touch any durable memory note a *promoted* proposal's
        content was already merged into (``memory/topics/*.md``) — that
        content became independent, reviewed memory the moment it was
        approved, and deleting the source conversation doesn't retroactively
        invalidate it. The proposal *bookkeeping row* is still deleted here
        regardless of its status (pending/rejected/promoted): it's pure
        session-derived tracking data, not the note itself. Cleaning up the
        matching indexed proposal chunk, draft previews, and knowledge-graph
        evidence citations in :class:`~minion_assist.memory.postgres_index.PostgresMemoryIndex`
        is the caller's job (:meth:`~minion_assist.memory.service.MemoryService.forget_proposals`)
        — a separate class/connection this method has no access to, using
        the ``proposal_ids`` this method returns.

        Args:
            agent_id: The agent this session must belong to.
            session_id: The session to delete.

        Returns:
            dict | None: ``None`` if ``session_id`` isn't owned by
                ``agent_id`` (fail closed — matches every other MEM-GAP-002
                scoped method's behavior; a missing/foreign session_id is
                treated as "nothing to delete," not an error). Otherwise
                ``{"messages": int, "proposal_ids": list[int]}`` — the
                message count deleted, and every deleted
                ``memory_proposals`` row's id, for the caller's PostgresMemoryIndex
                cleanup pass.
        """
        if not self._session_owned_by(session_id, agent_id):
            return None
        conn = self._conn()

        message_ids = [
            r[0] for r in conn.execute(
                "SELECT id FROM messages WHERE session_id = %s", (session_id,)
            ).fetchall()
        ]
        if message_ids and self.has_vector_lane:
            conn.execute(
                "DELETE FROM message_embeddings WHERE message_id = ANY(%s)", (message_ids,)
            )
        conn.execute("DELETE FROM message_embedding_jobs WHERE session_id = %s", (session_id,))
        conn.execute("DELETE FROM message_mirrors WHERE session_id = %s", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))

        proposal_ids = [
            r[0] for r in conn.execute(
                "SELECT id FROM memory_proposals WHERE session_id = %s", (session_id,)
            ).fetchall()
        ]
        if proposal_ids:
            conn.execute(
                "DELETE FROM memory_proposals WHERE id = ANY(%s)", (proposal_ids,)
            )
        conn.execute("DELETE FROM memory_capture_jobs WHERE session_id = %s", (session_id,))
        conn.execute("DELETE FROM memory_commitment_jobs WHERE session_id = %s", (session_id,))
        conn.execute("DELETE FROM session_job_coverage_ranges WHERE session_id = %s", (session_id,))
        conn.execute("DELETE FROM commitments WHERE session_id = %s", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = %s", (session_id,))

        return {"messages": len(message_ids), "proposal_ids": proposal_ids}

    # ------------------------------------------------------------------
    # Deletion tombstones (R2-GAP-007)
    # ------------------------------------------------------------------
    # See _migration_007_deletion_tombstones's docstring for why
    # /delete-session's three cross-store phases (JSONL, this class's own
    # delete_session, PostgresMemoryIndex evidence cleanup — orchestrated
    # by commands.py, not this class) need a durable, resumable record.

    def start_deletion_tombstone(self, agent_id: str, session_id: str) -> dict:
        """Create (or return the existing) tombstone row for one deletion attempt.

        ``ON CONFLICT ... DO NOTHING`` makes this safe to call even if a
        tombstone already exists for this exact ``(agent_id, session_id)``
        — returns whatever phases are already recorded as done rather than
        resetting them, so calling this again never un-does progress.

        Returns:
            dict: ``{"jsonl_deleted", "db_deleted", "evidence_cleaned",
                "proposal_ids", "completed_at"}`` — the tombstone's current
                state (freshly all-``False``/empty for a brand new one).
        """
        conn = self._conn()
        conn.execute(
            "INSERT INTO deletion_tombstones (agent_id, session_id, requested_at) "
            "VALUES (%s, %s, %s) ON CONFLICT (agent_id, session_id) DO NOTHING",
            (agent_id, session_id, time.time()),
        )
        return self.get_deletion_tombstone(agent_id, session_id)

    def get_deletion_tombstone(self, agent_id: str, session_id: str) -> dict | None:
        """Return one deletion attempt's current state, or ``None`` if none was ever started."""
        row = self._conn().execute(
            "SELECT jsonl_deleted, db_deleted, evidence_cleaned, proposal_ids, completed_at "
            "FROM deletion_tombstones WHERE agent_id = %s AND session_id = %s",
            (agent_id, session_id),
        ).fetchone()
        if row is None:
            return None
        return {
            "jsonl_deleted": row[0],
            "db_deleted": row[1],
            "evidence_cleaned": row[2],
            "proposal_ids": list(row[3]) if row[3] else [],
            "completed_at": row[4],
        }

    def mark_deletion_jsonl_done(self, agent_id: str, session_id: str) -> None:
        """Record that the JSONL-file phase of one deletion attempt has completed."""
        self._conn().execute(
            "UPDATE deletion_tombstones SET jsonl_deleted = TRUE "
            "WHERE agent_id = %s AND session_id = %s",
            (agent_id, session_id),
        )

    def mark_deletion_db_done(
        self, agent_id: str, session_id: str, proposal_ids: list[int]
    ) -> None:
        """Record that :meth:`delete_session`'s phase has completed.

        Args:
            proposal_ids: Exactly what that call returned — snapshotted
                here so a later evidence-cleanup retry (which needs these
                ids) never has to re-derive them from rows that no longer
                exist by the time it runs.
        """
        self._conn().execute(
            "UPDATE deletion_tombstones SET db_deleted = TRUE, proposal_ids = %s "
            "WHERE agent_id = %s AND session_id = %s",
            (proposal_ids, agent_id, session_id),
        )

    def mark_deletion_evidence_done(self, agent_id: str, session_id: str) -> None:
        """Record that the indexed-evidence cleanup phase has completed — the whole attempt is now done."""
        self._conn().execute(
            "UPDATE deletion_tombstones SET evidence_cleaned = TRUE, completed_at = %s "
            "WHERE agent_id = %s AND session_id = %s",
            (time.time(), agent_id, session_id),
        )

    def list_incomplete_deletion_tombstones(self) -> list[dict]:
        """Every deletion attempt that hasn't finished all three phases, oldest first.

        Used by ``minion-assist memory verify-deletions`` to surface (and,
        with ``--retry``, finish) any deletion a prior process crash or
        transient failure left stuck partway through.

        Returns:
            list[dict]: ``{"agent_id", "session_id", "jsonl_deleted",
                "db_deleted", "evidence_cleaned", "proposal_ids"}`` per
                incomplete attempt.
        """
        rows = self._conn().execute(
            "SELECT agent_id, session_id, jsonl_deleted, db_deleted, evidence_cleaned, proposal_ids "
            "FROM deletion_tombstones WHERE completed_at IS NULL ORDER BY requested_at"
        ).fetchall()
        return [
            {
                "agent_id": r[0],
                "session_id": r[1],
                "jsonl_deleted": r[2],
                "db_deleted": r[3],
                "evidence_cleaned": r[4],
                "proposal_ids": list(r[5]) if r[5] else [],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str | None,
        tool_name: str | None = None,
        timestamp: float | None = None,
    ) -> int:
        """Insert a message row unconditionally; returns the new message id.

        No idempotency check — calling this twice with the same content
        creates two rows. Use :meth:`mirror_message` for the idempotent
        version (mirroring live turns and reconciliation both go through
        that, not this).
        """
        row = self._conn().execute(
            """
            INSERT INTO messages (session_id, role, content, tool_name, timestamp)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (session_id, role, content, tool_name, timestamp or time.time()),
        ).fetchone()
        return row[0] if row else -1

    def is_mirrored(self, session_id: str, event_id: str) -> bool:
        """Return ``True`` if this exact message has already been mirrored."""
        row = self._conn().execute(
            "SELECT 1 FROM message_mirrors WHERE session_id = %s AND event_id = %s",
            (session_id, event_id),
        ).fetchone()
        return row is not None

    def mirror_message(
        self,
        session_id: str,
        event_id: str,
        role: str,
        content: str | None,
        tool_name: str | None = None,
        timestamp: float | None = None,
    ) -> int | None:
        """Idempotently insert a message, keyed by ``(session_id, event_id)``.

        Args:
            session_id: The session this message belongs to.
            event_id: The message's stable identity (``messages.py``'s
                ``ensure_event_id()``).
            role, content, tool_name, timestamp: Same as :meth:`add_message`.

        Returns:
            int | None: The new message row's id, or ``None`` if this exact
                ``(session_id, event_id)`` was already mirrored (a no-op,
                not an error — see the module docstring's concurrency note
                for why this check-then-insert is safe here).
        """
        if self.is_mirrored(session_id, event_id):
            return None
        message_id = self.add_message(
            session_id, role, content, tool_name=tool_name, timestamp=timestamp
        )
        self._conn().execute(
            """
            INSERT INTO message_mirrors (session_id, event_id, message_id, mirrored_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (session_id, event_id) DO NOTHING
            """,
            (session_id, event_id, message_id, time.time()),
        )
        return message_id

    def search_messages(self, query: str, agent_id: str, limit: int = 10) -> list[dict]:
        """FTS search across this agent's own message content only.

        Joins to ``sessions`` and filters on ``agent_id`` (MEM-GAP-002) so a
        full-text query can never surface another agent's conversation
        history — agents are meant to be completely isolated from each
        other's memory, even though they share one PostgreSQL database.

        R2-GAP-012: uses PostgreSQL's ``'simple'`` text-search
        configuration (tokenize + lowercase, no stemming/stopwords) rather
        than ``'english'`` — must always match whatever configuration
        ``messages.search_vector`` was generated with (migration 008), or
        matching breaks; see that migration's docstring for why.
        """
        rows = self._conn().execute(
            """
            SELECT
                m.id,
                m.session_id,
                m.role,
                m.content,
                m.tool_name,
                m.timestamp,
                ts_headline('simple', coalesce(m.content, ''),
                    websearch_to_tsquery('simple', %s),
                    'MaxWords=30, MinWords=10, StartSel=[, StopSel=]'
                ) AS snippet,
                ts_rank(m.search_vector, websearch_to_tsquery('simple', %s)) AS rank
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE s.agent_id = %s
              AND m.search_vector @@ websearch_to_tsquery('simple', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (query, query, agent_id, query, limit),
        ).fetchall()
        return [
            {
                "id": r[0], "session_id": r[1], "role": r[2],
                "content": r[3], "tool_name": r[4], "timestamp": r[5],
                "snippet": r[6], "rank": float(r[7]),
            }
            for r in rows
        ]

    def _vector_search_messages(self, query: str, agent_id: str, limit: int) -> list[dict]:
        """Cosine-similarity ranking against cached message embeddings (MEM-GAP-006).

        Mirrors ``PostgresMemoryIndex._vector_lane``'s shape and error
        handling: returns an empty lane (never raises) if no embedding
        provider is configured, or if embedding the query itself fails —
        the vector lane is optional by design, same as the memory index's.
        Scoped by ``agent_id`` through the same ``sessions`` join
        :meth:`search_messages` uses, so this can never surface another
        agent's message content (MEM-GAP-002).

        R2-GAP-012: filters out anything below ``self._min_similarity``
        (``config.session_search.min_similarity``) — without a floor, this
        always returned its ``limit`` nearest neighbors even when every one
        of them was actually unrelated to the query. "Nearest" only means
        "relevant" once real matches exist; for a novel query, the nearest
        vectors are still whatever happened to be closest, not a match.
        """
        if self._embedding_provider is None or not self.has_vector_lane:
            return []
        try:
            [query_vector] = self._embedding_provider.embed([query])
        except Exception:
            return []

        # %s::vector explicit casts: without them, psycopg sends a bare
        # Python list as a plain float8[] array and pgvector's <=> operator
        # has no overload for vector <=> float8[] — same reasoning as
        # PostgresMemoryIndex._vector_lane's identical cast.
        rows = self._conn().execute(
            """
            SELECT m.id, m.session_id, m.role, m.content, m.tool_name, m.timestamp,
                   1 - (me.embedding <=> %s::vector) AS similarity
            FROM messages m
            JOIN message_embeddings me ON me.message_id = m.id AND me.model_identity = %s
            JOIN sessions s ON s.id = m.session_id
            WHERE s.agent_id = %s
              AND 1 - (me.embedding <=> %s::vector) >= %s
            ORDER BY me.embedding <=> %s::vector
            LIMIT %s
            """,
            (
                query_vector, self._embedding_provider.model_identity, agent_id,
                query_vector, self._min_similarity, query_vector, limit,
            ),
        ).fetchall()
        return [
            {
                "id": r[0], "session_id": r[1], "role": r[2],
                "content": r[3], "tool_name": r[4], "timestamp": r[5],
                "snippet": None, "rank": float(r[6]),
            }
            for r in rows
        ]

    def hybrid_search_messages(self, query: str, agent_id: str, limit: int = 10) -> list[dict]:
        """Lexical + vector fused session-message search (MEM-GAP-006).

        Reciprocal-rank-fuses :meth:`search_messages`'s FTS lane with
        :meth:`_vector_search_messages`'s cosine-similarity lane, reusing
        ``memory/postgres_index.py``'s ``_reciprocal_rank_fusion`` — the
        same fusion algorithm ``PostgresMemoryIndex.hybrid_search`` uses for
        memory files, applied here to session messages instead. Degrades to
        FTS-only automatically when no embedding provider is configured
        (the vector lane returns ``[]``), so callers
        (``tools/session_search.py``) never need to branch on whether
        embeddings are available.

        Args:
            query: The search text — used verbatim for both lanes (FTS
                ``websearch_to_tsquery`` syntax and the embedding call).
            agent_id: Scopes both lanes (MEM-GAP-002).
            limit: Result count per lane before fusion, and of the final
                fused list.

        Returns:
            list[dict]: Same shape as :meth:`search_messages`'s rows
                (``id``, ``session_id``, ``role``, ``content``, ``tool_name``,
                ``timestamp``, ``snippet``, ``rank``), ordered by fused
                score descending. ``rank`` is the fused RRF score, not
                either lane's original rank/similarity.
        """
        lexical_hits = self.search_messages(query, agent_id, limit=limit)
        vector_hits = self._vector_search_messages(query, agent_id, limit)
        if not vector_hits:
            return lexical_hits

        from ..memory.postgres_index import _reciprocal_rank_fusion  # noqa: PLC0415

        # Lexical lane listed first: when both lanes agree on a message,
        # _reciprocal_rank_fusion's rows.setdefault() keeps the lexical
        # lane's dict (real ts_headline snippet) over the vector lane's
        # (snippet=None) — same lane ordering PostgresMemoryIndex.hybrid_search
        # uses so richer-field lanes win ties.
        scores, rows = _reciprocal_rank_fusion([lexical_hits, vector_hits])
        ranked_ids = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        fused = []
        for message_id, score in ranked_ids:
            row = dict(rows[message_id])
            row["rank"] = score
            fused.append(row)
        return fused

    def _session_owned_by(self, session_id: str, agent_id: str) -> bool:
        """Return ``True`` if ``session_id`` exists and belongs to ``agent_id``.

        Used to fail closed (MEM-GAP-002) before reading any message content
        keyed only by ``session_id`` — ``messages`` has no ``agent_id``
        column of its own, so this ownership check is what stops a caller
        from guessing another agent's session_id and scrolling through it.
        """
        row = self._conn().execute(
            "SELECT 1 FROM sessions WHERE id = %s AND agent_id = %s",
            (session_id, agent_id),
        ).fetchone()
        return row is not None

    def get_messages_around(
        self, session_id: str, agent_id: str, anchor_id: int, window: int = 5
    ) -> list[dict]:
        """Return messages in [anchor_id-window, anchor_id+window] for a session.

        Returns an empty list (rather than raising) if ``session_id`` isn't
        owned by ``agent_id`` — see :meth:`_session_owned_by`.
        """
        if not self._session_owned_by(session_id, agent_id):
            return []
        if anchor_id <= 0:
            row = self._conn().execute(
                "SELECT id FROM messages WHERE session_id = %s ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            anchor_id = row[0] if row else 0
        if anchor_id <= 0:
            return []
        rows = self._conn().execute(
            """
            SELECT id, role, content, tool_name, timestamp
            FROM messages
            WHERE session_id = %s AND id BETWEEN %s AND %s
            ORDER BY id
            """,
            (session_id, anchor_id - window, anchor_id + window),
        ).fetchall()
        return [
            {"id": r[0], "role": r[1], "content": r[2], "tool_name": r[3], "timestamp": r[4]}
            for r in rows
        ]

    def get_session_bookends(
        self, session_id: str, agent_id: str, n: int = 3
    ) -> tuple[list[dict], list[dict]]:
        """Return first n and last n user/assistant messages for a session.

        Returns ``([], [])`` if ``session_id`` isn't owned by ``agent_id``
        (MEM-GAP-002) — see :meth:`_session_owned_by`.
        """
        if not self._session_owned_by(session_id, agent_id):
            return [], []
        conn = self._conn()
        first = conn.execute(
            """
            SELECT id, role, content FROM messages
            WHERE session_id = %s AND role IN ('user', 'assistant')
            ORDER BY id LIMIT %s
            """,
            (session_id, n),
        ).fetchall()
        last = conn.execute(
            """
            SELECT id, role, content FROM (
                SELECT id, role, content FROM messages
                WHERE session_id = %s AND role IN ('user', 'assistant')
                ORDER BY id DESC LIMIT %s
            ) sub ORDER BY id
            """,
            (session_id, n),
        ).fetchall()
        to_dict = lambda r: {"id": r[0], "role": r[1], "content": r[2]}  # noqa: E731
        return [to_dict(r) for r in first], [to_dict(r) for r in last]

    def get_messages_in_range(self, session_id: str, from_id: int, to_id: int) -> list[dict]:
        """Return every message in ``[from_id, to_id]`` for a session, in order.

        Used by :class:`~minion_assist.memory.capture_worker.CaptureWorker`
        to fetch a capture job's source messages (Stage One Phase 2, slice C).

        Args:
            session_id: The session to read from.
            from_id: Lowest message id to include (inclusive).
            to_id: Highest message id to include (inclusive).

        Returns:
            list[dict]: ``{"id", "role", "content", "tool_name", "timestamp"}``
                dicts, ordered by id. These are raw DB rows, not
                provider-ready message dicts — callers must build a clean
                ``{"role", "content"}`` dict before passing anything to
                ``provider.chat()`` (extra keys here would otherwise leak
                into the API request the same way ``EVENT_ID_KEY`` almost
                did — see ``providers/openai_compatible.py``'s
                ``_prepare_messages_for_openai``).
        """
        rows = self._conn().execute(
            """
            SELECT id, role, content, tool_name, timestamp
            FROM messages
            WHERE session_id = %s AND id BETWEEN %s AND %s
            ORDER BY id
            """,
            (session_id, from_id, to_id),
        ).fetchall()
        return [
            {"id": r[0], "role": r[1], "content": r[2], "tool_name": r[3], "timestamp": r[4]}
            for r in rows
        ]

    def list_message_ids(self, session_id: str) -> list[int]:
        """Every message id in one session, ascending.

        Stage One Phase 5, slice D: ``memory/consolidation.py``'s
        ``backfill_agent`` diffs this against
        :meth:`list_capture_job_ranges` to find message ranges no capture
        job has ever covered.
        """
        rows = self._conn().execute(
            "SELECT id FROM messages WHERE session_id = %s ORDER BY id", (session_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def list_capture_job_ranges(self, session_id: str) -> list[tuple[int, int]]:
        """Every capture range ever successfully enqueued for one session.

        Reads ``session_job_coverage_ranges`` (R2-GAP-002), not
        ``memory_capture_jobs`` directly — a range is recorded once, at
        enqueue time, and never removed even after retention prunes the
        job row itself, so a session's full enqueue history stays visible
        regardless of how old its jobs are. Every job state originally
        counted as "covered" here, including ``failed`` — extraction
        having been *attempted* (successfully enqueued) is the bar, not
        having *succeeded*; retrying a failed attempt is
        ``CaptureWorker``'s own retry/backoff responsibility, up to
        ``_MAX_ATTEMPTS`` — and since a range is recorded unconditionally
        at enqueue time, that's still exactly what this returns.

        Stage One Phase 5, slice D: the other half of
        ``backfill_agent``'s gap computation, also used by
        :meth:`find_uncovered_capture_ranges` (R2-GAP-002).
        """
        rows = self._conn().execute(
            "SELECT from_id, to_id FROM session_job_coverage_ranges "
            "WHERE session_id = %s AND lane = 'capture'",
            (session_id,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def list_commitment_job_ranges(self, session_id: str) -> list[tuple[int, int]]:
        """Every commitment-extraction range ever successfully enqueued for one session.

        Same source and reasoning as :meth:`list_capture_job_ranges` — see
        its docstring. Channel isn't returned here (unlike
        :meth:`find_uncovered_commitment_ranges`'s result, which pairs
        each gap with a channel) since this is purely the "which message
        ids are covered" half of the gap computation; channel is a
        separate lookup for the one case (a genuinely uncovered gap) where
        it's actually needed.
        """
        rows = self._conn().execute(
            "SELECT from_id, to_id FROM session_job_coverage_ranges "
            "WHERE session_id = %s AND lane = 'commitment'",
            (session_id,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def _eligible_message_ids(
        self, session_id: str, limit: "int | None | object" = _UNSET
    ) -> list[int]:
        """User/assistant message ids with non-empty content, ascending.

        Shared by :meth:`find_uncovered_capture_ranges`/
        :meth:`find_uncovered_commitment_ranges` — deliberately narrower
        than :meth:`list_message_ids` (which returns *every* message,
        tool calls included, for :func:`compute_backfill_windows`'s other
        caller, ``backfill_agent``): reconciliation runs automatically and
        periodically, so a run of pure tool-call messages with no actual
        dialogue must never become its own pointless enqueued job the way
        it's harmless for a one-off manual backfill to occasionally do.

        Args:
            limit: R2-GAP-011 — when omitted (the default), returns only
                the *most recent* :data:`_RECONCILIATION_SCAN_LIMIT`
                eligible ids, not the session's entire history — see that
                constant's comment for why. The default is resolved
                inside the method body (a private ``_UNSET`` sentinel),
                not baked into the parameter list, specifically so tests
                can lower ``_RECONCILIATION_SCAN_LIMIT`` and see this
                method's behavior actually change. Pass ``None`` explicitly
                for the old fully-unbounded behavior (only meaningful for
                tests/tools that genuinely want the whole history); pass
                an ``int`` to override the bound directly.
        """
        if limit is _UNSET:
            limit = _RECONCILIATION_SCAN_LIMIT
        if limit is None:
            rows = self._conn().execute(
                "SELECT id FROM messages WHERE session_id = %s "
                "AND role IN ('user', 'assistant') AND content IS NOT NULL ORDER BY id",
                (session_id,),
            ).fetchall()
            return [r[0] for r in rows]
        # DESC + LIMIT to get the most-recent slice cheaply (an index scan
        # bounded by `limit`, not a full-table scan), then reverse in
        # Python back to the ascending order compute_backfill_windows needs.
        rows = self._conn().execute(
            "SELECT id FROM messages WHERE session_id = %s "
            "AND role IN ('user', 'assistant') AND content IS NOT NULL "
            "ORDER BY id DESC LIMIT %s",
            (session_id, limit),
        ).fetchall()
        return [r[0] for r in reversed(rows)]

    def list_session_ids_for_agent(self, agent_id: str) -> list[str]:
        """Every session id ever recorded for one agent — no recency limit.

        Unlike :meth:`list_sessions` (most recent N, for display), this is
        exhaustive — Stage One Phase 5, slice D's ``backfill_agent`` needs
        every session, including ones long past any "recent" window, to
        find historical gaps.
        """
        rows = self._conn().execute(
            "SELECT id FROM sessions WHERE agent_id = %s", (agent_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def set_proposal_status(self, proposal_id: int, status: str, reason: str = "") -> None:
        """Set one proposal's review status (and optional rejection reason).

        Stage One Phase 5, slice D: used by
        :class:`~minion_assist.memory.consolidation.MemoryConsolidator`'s
        ``approve()`` (→ ``"promoted"``), ``reject()`` (→ ``"rejected"``,
        with a reason), and ``rollback()`` (→ back to ``"pending"``).

        Args:
            proposal_id: Which proposal to update.
            status: ``"pending"``, ``"promoted"``, ``"rejected"``, or
                ``"superseded"`` — not validated against this list here
                (this is an internal primitive; validation, if any, is the
                caller's job) the same way ``fail_capture_job`` doesn't
                re-validate ``state`` either.
            reason: Stored as ``rejected_reason`` — always set (default
                ``""``), so any transition *other than* a fresh
                ``reject()`` call clears a stale reason from a previous
                decision rather than leaving it to linger.
        """
        self._conn().execute(
            "UPDATE memory_proposals SET status = %s, rejected_reason = %s WHERE id = %s",
            (status, reason, proposal_id),
        )

    # ------------------------------------------------------------------
    # Reconciliation (Stage One Phase 2, slice A)
    # ------------------------------------------------------------------

    def reconcile_session(
        self,
        session_id: str,
        agent_id: str,
        messages: list[dict],
        mtime: float,
    ) -> int:
        """Mirror every message in ``messages`` that isn't mirrored yet.

        Unlike the old ``replay_jsonl()``, this never skips a session just
        because it already has some rows — it checks each message
        individually via :meth:`mirror_message`'s idempotency, so a crash
        that left a session partially mirrored is completed here rather
        than left behind forever.

        Messages without an existing event ID are assigned one in place
        (``messages.py``'s ``ensure_event_id()``) — callers MUST persist
        ``messages`` back to JSONL if this mutated any message, or those
        IDs will be regenerated (and mirrored as "new") on the next call.
        :meth:`reconcile_all_sessions` handles this automatically.

        Args:
            session_id: The session being reconciled.
            agent_id: The owning agent — used to upsert the session row.
            messages: The session's full message list, as loaded from JSONL.
            mtime: Used as the session's ``started_at``/mirrored timestamp
                when a value isn't otherwise available.

        Returns:
            int: Number of messages newly mirrored by this call.
        """
        self.upsert_session(session_id, agent_id, started_at=mtime)

        mirrored = 0
        for msg in messages:
            role = msg.get("role", "")
            if not role:
                continue
            event_id = ensure_event_id(msg)
            content = _msg_text(msg)
            tool_name = msg.get("name") or msg.get("tool_name")
            result = self.mirror_message(
                session_id, event_id, role, content, tool_name=tool_name, timestamp=mtime
            )
            if result is not None:
                mirrored += 1
        return mirrored

    def reconcile_all_sessions(
        self,
        short_term: object,
        agent_ids: list[str],
    ) -> int:
        """Reconcile every JSONL session file for every agent against ``message_mirrors``.

        Runs at startup (replacing the old one-time ``replay_jsonl()``) and
        is safe to call repeatedly — already-mirrored messages are no-ops.
        Re-saves a session's JSONL file via ``short_term.save()`` whenever
        :meth:`reconcile_session` assigned any message a new event ID, so
        the ID is never lost to a subsequent restart.

        Args:
            short_term: The agent's ``ShortTermMemory`` instance (duck-typed
                — only ``list_sessions``/``load``/``save`` are used).
            agent_ids: Every configured agent ID to reconcile.

        Returns:
            int: Total number of messages newly mirrored across all sessions.
        """
        total = 0
        for agent_id in agent_ids:
            for path in short_term.list_sessions(agent_id):
                session_id = path.stem
                messages = short_term.load(agent_id, session_id)
                if not messages:
                    continue

                ids_before = [m.get(EVENT_ID_KEY) for m in messages]
                mtime = path.stat().st_mtime
                total += self.reconcile_session(session_id, agent_id, messages, mtime)

                if [m.get(EVENT_ID_KEY) for m in messages] != ids_before:
                    short_term.save(agent_id, session_id, messages)
        return total

    # ------------------------------------------------------------------
    # Durable capture jobs (Stage One Phase 2, slice C)
    # ------------------------------------------------------------------

    def enqueue_capture_job(
        self,
        agent_id: str,
        session_id: str,
        source_from_message_id: int,
        source_to_message_id: int,
        idempotency_key: str,
    ) -> int | None:
        """Enqueue a capture job, idempotently.

        Args:
            agent_id: The agent whose exchange should be extracted from.
            session_id: The session the source messages belong to.
            source_from_message_id: Lowest message id in the source range.
            source_to_message_id: Highest message id in the source range.
            idempotency_key: Typically ``"{agent}:{session}:{from}-{to}:{prompt_version}:{model}"``
                (see ``memory/extractor.py``). Enqueuing the same key twice
                is a no-op, not a duplicate job.

        Returns:
            int | None: The new job's id, or ``None`` if a job with this
                ``idempotency_key`` already exists.
        """
        now = time.time()
        conn = self._conn()
        row = conn.execute(
            """
            INSERT INTO memory_capture_jobs
                (agent_id, session_id, source_from_message_id, source_to_message_id,
                 idempotency_key, state, attempts, run_after, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, 'pending', 0, %s, %s, %s)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING id
            """,
            (
                agent_id, session_id, source_from_message_id, source_to_message_id,
                idempotency_key, now, now, now,
            ),
        ).fetchone()
        if row is None:
            return None
        _record_job_coverage_range(
            conn, agent_id, session_id, "capture", source_from_message_id, source_to_message_id,
        )
        return row[0]

    def claim_next_capture_job(self) -> dict | None:
        """Atomically claim one pending, due capture job for processing.

        Uses ``FOR UPDATE SKIP LOCKED`` so multiple workers (should more than
        one ever run) never claim the same job twice. The whole claim is one
        SQL statement, so it is atomic even under this class's ``autocommit``
        connections — no explicit transaction needed.

        Returns:
            dict | None: ``{"id", "agent_id", "session_id",
                "source_from_message_id", "source_to_message_id", "attempts"}``,
                or ``None`` if no job is currently due.
        """
        now = time.time()
        row = self._conn().execute(
            """
            UPDATE memory_capture_jobs
            SET state = 'running', updated_at = %s
            WHERE id = (
                SELECT id FROM memory_capture_jobs
                WHERE state = 'pending' AND run_after <= %s
                ORDER BY id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, agent_id, session_id, source_from_message_id,
                      source_to_message_id, attempts
            """,
            (now, now),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "agent_id": row[1],
            "session_id": row[2],
            "source_from_message_id": row[3],
            "source_to_message_id": row[4],
            "attempts": row[5],
        }

    def complete_capture_job(self, job_id: int, proposals: list[str]) -> list[dict]:
        """Mark a capture job done and record its extracted proposals.

        Args:
            job_id: The job to complete.
            proposals: 0 or more extracted claim strings. An empty list is a
                normal, successful outcome ("nothing worth remembering this
                time"), not a failure.

        Returns:
            list[dict]: One ``{"id", "agent_id", "claim_text"}`` per newly
                inserted proposal row, in the same order as ``proposals``.
                Stage One Phase 5, slice B: :class:`~minion_assist.memory.capture_worker.CaptureWorker`
                uses these ids to index each new proposal as searchable
                right after this call returns, without a second query.
        """
        now = time.time()
        conn = self._conn()
        job_row = conn.execute(
            "SELECT agent_id, session_id FROM memory_capture_jobs WHERE id = %s", (job_id,)
        ).fetchone()
        agent_id = job_row[0] if job_row else ""
        session_id = job_row[1] if job_row else ""
        new_proposals = []
        for claim in proposals:
            row = conn.execute(
                """
                INSERT INTO memory_proposals (job_id, session_id, agent_id, claim_text, created_at)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (job_id, session_id, agent_id, claim, now),
            ).fetchone()
            new_proposals.append({"id": row[0], "agent_id": agent_id, "claim_text": claim})
        conn.execute(
            "UPDATE memory_capture_jobs SET state = 'done', updated_at = %s WHERE id = %s",
            (now, job_id),
        )
        return new_proposals

    def fail_capture_job(
        self, job_id: int, error: str, backoff_seconds: float, max_attempts: int = 5
    ) -> None:
        """Record a failed capture-job attempt — retry with backoff, or give up.

        Args:
            job_id: The job that failed.
            error: Short description of the failure, stored as ``last_error``.
            backoff_seconds: How long to wait before this job is eligible to
                be claimed again (``run_after = now + backoff_seconds``).
            max_attempts: Once ``attempts`` reaches this count, the job is
                marked ``'failed'`` instead of ``'pending'`` — it stops being
                retried automatically but remains inspectable in the table.
        """
        now = time.time()
        conn = self._conn()
        row = conn.execute(
            "SELECT attempts FROM memory_capture_jobs WHERE id = %s", (job_id,)
        ).fetchone()
        attempts = (row[0] if row else 0) + 1
        state = "failed" if attempts >= max_attempts else "pending"
        conn.execute(
            """
            UPDATE memory_capture_jobs
            SET state = %s, attempts = %s, run_after = %s, last_error = %s, updated_at = %s
            WHERE id = %s
            """,
            (state, attempts, now + backoff_seconds, error, now, job_id),
        )

    # ------------------------------------------------------------------
    # Durable commitment-extraction jobs (Stage One Phase 6, slice B)
    # ------------------------------------------------------------------
    #
    # Structurally identical to the capture-job methods above — see the
    # module docstring's commitments schema note for why this is a
    # separate table/queue rather than an extension of memory_capture_jobs.

    def enqueue_commitment_job(
        self,
        agent_id: str,
        session_id: str,
        channel: str,
        source_from_message_id: int,
        source_to_message_id: int,
        idempotency_key: str,
    ) -> int | None:
        """Enqueue a commitment-extraction job, idempotently.

        Args mirror :meth:`enqueue_capture_job` with one addition:
        ``channel`` — a Matrix room id, or ``"cli"`` outside Matrix. This
        is what scopes a later-extracted commitment to "exact agent and
        channel context" (the plan's Task 4).

        Returns:
            int | None: The new job's id, or ``None`` if a job with this
                ``idempotency_key`` already exists.
        """
        now = time.time()
        conn = self._conn()
        row = conn.execute(
            """
            INSERT INTO memory_commitment_jobs
                (agent_id, session_id, channel, source_from_message_id,
                 source_to_message_id, idempotency_key, state, attempts,
                 run_after, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, 'pending', 0, %s, %s, %s)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING id
            """,
            (
                agent_id, session_id, channel, source_from_message_id,
                source_to_message_id, idempotency_key, now, now, now,
            ),
        ).fetchone()
        if row is None:
            return None
        _record_job_coverage_range(
            conn, agent_id, session_id, "commitment", source_from_message_id, source_to_message_id,
            channel=channel,
        )
        return row[0]

    def claim_next_commitment_job(self) -> dict | None:
        """Atomically claim one pending, due commitment-extraction job.

        Same ``FOR UPDATE SKIP LOCKED`` mechanics as
        :meth:`claim_next_capture_job` — see its docstring.

        Returns:
            dict | None: ``{"id", "agent_id", "session_id", "channel",
                "source_from_message_id", "source_to_message_id",
                "attempts"}``, or ``None`` if no job is currently due.
        """
        now = time.time()
        row = self._conn().execute(
            """
            UPDATE memory_commitment_jobs
            SET state = 'running', updated_at = %s
            WHERE id = (
                SELECT id FROM memory_commitment_jobs
                WHERE state = 'pending' AND run_after <= %s
                ORDER BY id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, agent_id, session_id, channel, source_from_message_id,
                      source_to_message_id, attempts
            """,
            (now, now),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "agent_id": row[1],
            "session_id": row[2],
            "channel": row[3],
            "source_from_message_id": row[4],
            "source_to_message_id": row[5],
            "attempts": row[6],
        }

    def list_pending_commitments_for_scope(
        self, agent_id: str, channel: str, limit: int = 8
    ) -> list[dict]:
        """List pending commitments already tracked for one (agent, channel) scope.

        Fed into the extraction prompt as ``existing_pending`` context
        (``memory/commitments.py``'s ``build_commitment_extraction_prompt``)
        so the model can extend an already-tracked follow-up instead of
        proposing a near-duplicate — the same anti-duplication input
        OpenClaw's real extractor uses (verified against
        ``ref-repos/openclaw/src/commitments/extraction.ts``'s
        ``hydrateCommitmentExtractionItem``).

        Returns:
            list[dict]: ``{"kind", "reason", "dedupe_key", "due_earliest",
                "due_latest"}`` dicts, most recently created first, capped
                at ``limit``.
        """
        rows = self._conn().execute(
            """
            SELECT kind, reason, dedupe_key, due_earliest, due_latest
            FROM commitments
            WHERE agent_id = %s AND channel = %s AND status = 'pending'
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (agent_id, channel, limit),
        ).fetchall()
        return [
            {
                "kind": r[0], "reason": r[1], "dedupe_key": r[2],
                "due_earliest": r[3], "due_latest": r[4],
            }
            for r in rows
        ]

    def complete_commitment_job(self, job_id: int, candidates: list[dict]) -> list[dict]:
        """Mark a commitment job done and record its extracted candidates.

        A candidate whose ``dedupe_key`` matches an existing *pending*
        commitment in the same ``(agent_id, channel)`` scope is upserted
        into that row (widening the due window to
        ``min(earliest)``/``max(latest)``, keeping the higher confidence)
        rather than inserted as a duplicate — mirrors OpenClaw's real
        ``upsertInferredCommitments`` (verified against
        ``ref-repos/openclaw/src/commitments/store.ts``), scaled down to
        this codebase's simpler needs.

        Args:
            job_id: The job to complete.
            candidates: 0 or more validated candidate dicts (see
                ``memory/commitments.py``'s ``extract_commitments`` —
                already confidence-gated and due-window-clamped by the
                time they reach here). Each needs ``kind``, ``sensitivity``,
                ``source``, ``reason``, ``suggested_text``, ``dedupe_key``,
                ``confidence``, ``due_earliest``, ``due_latest`` (all as
                produced by ``extract_commitments``).

        Returns:
            list[dict]: ``{"id", "created"}`` per candidate — ``created``
                is ``False`` when the candidate was merged into an
                existing pending commitment rather than inserted fresh.
        """
        now = time.time()
        conn = self._conn()
        job_row = conn.execute(
            "SELECT agent_id, session_id, channel FROM memory_commitment_jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
        agent_id, session_id, channel = job_row if job_row else ("", "", "")

        results = []
        for candidate in candidates:
            existing = conn.execute(
                """
                SELECT id, due_earliest, due_latest, confidence FROM commitments
                WHERE agent_id = %s AND channel = %s AND dedupe_key = %s AND status = 'pending'
                """,
                (agent_id, channel, candidate["dedupe_key"]),
            ).fetchone()
            if existing is not None:
                existing_id, existing_earliest, existing_latest, existing_confidence = existing
                conn.execute(
                    """
                    UPDATE commitments
                    SET due_earliest = %s, due_latest = %s, confidence = %s,
                        reason = %s, suggested_text = %s, updated_at = %s
                    WHERE id = %s
                    """,
                    (
                        min(existing_earliest, candidate["due_earliest"]),
                        max(existing_latest, candidate["due_latest"]),
                        max(existing_confidence, candidate["confidence"]),
                        candidate["reason"], candidate["suggested_text"], now,
                        existing_id,
                    ),
                )
                results.append({"id": existing_id, "created": False})
                continue
            row = conn.execute(
                """
                INSERT INTO commitments
                    (agent_id, session_id, channel, kind, sensitivity, source, reason,
                     suggested_text, dedupe_key, confidence, due_earliest, due_latest,
                     source_job_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    agent_id, session_id, channel, candidate["kind"], candidate["sensitivity"],
                    candidate["source"], candidate["reason"], candidate["suggested_text"],
                    candidate["dedupe_key"], candidate["confidence"], candidate["due_earliest"],
                    candidate["due_latest"], job_id, now, now,
                ),
            ).fetchone()
            results.append({"id": row[0], "created": True})

        conn.execute(
            "UPDATE memory_commitment_jobs SET state = 'done', updated_at = %s WHERE id = %s",
            (now, job_id),
        )
        return results

    def fail_commitment_job(
        self, job_id: int, error: str, backoff_seconds: float, max_attempts: int = 5
    ) -> None:
        """Record a failed commitment-extraction attempt — retry with backoff, or give up.

        Identical mechanics to :meth:`fail_capture_job` — see its
        docstring.
        """
        now = time.time()
        conn = self._conn()
        row = conn.execute(
            "SELECT attempts FROM memory_commitment_jobs WHERE id = %s", (job_id,)
        ).fetchone()
        attempts = (row[0] if row else 0) + 1
        state = "failed" if attempts >= max_attempts else "pending"
        conn.execute(
            """
            UPDATE memory_commitment_jobs
            SET state = %s, attempts = %s, run_after = %s, last_error = %s, updated_at = %s
            WHERE id = %s
            """,
            (state, attempts, now + backoff_seconds, error, now, job_id),
        )

    # ------------------------------------------------------------------
    # Message-embedding job queue (MEM-GAP-006)
    # ------------------------------------------------------------------
    # Same claim/complete/fail lifecycle as the capture/commitment queues
    # above (FOR UPDATE SKIP LOCKED, exponential backoff, ON CONFLICT DO
    # NOTHING idempotency), but one job embeds exactly one message rather
    # than extracting from a range — see the module docstring's "Durable
    # message-embedding jobs" section.

    def enqueue_message_embedding_job(
        self, agent_id: str, session_id: str, message_id: int, idempotency_key: str,
    ) -> int | None:
        """Enqueue a message-embedding job, idempotently.

        Args:
            agent_id: The agent this message belongs to.
            session_id: The session this message belongs to.
            message_id: The message to embed.
            idempotency_key: Typically ``"{agent}:{message_id}:{model_identity}"``
                — enqueuing the same message under the same embedding
                model/endpoint twice is a no-op, not a duplicate job.

        Returns:
            int | None: The new job's id, or ``None`` if a job with this
                ``idempotency_key`` already exists.
        """
        now = time.time()
        conn = self._conn()
        row = conn.execute(
            """
            INSERT INTO message_embedding_jobs
                (agent_id, session_id, message_id, idempotency_key,
                 state, attempts, run_after, created_at, updated_at)
            VALUES (%s, %s, %s, %s, 'pending', 0, %s, %s, %s)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING id
            """,
            (agent_id, session_id, message_id, idempotency_key, now, now, now),
        ).fetchone()
        # No coverage-table bookkeeping here, unlike enqueue_capture_job/
        # enqueue_commitment_job — this lane's coverage is checked directly
        # against message_embeddings instead (see
        # find_uncovered_message_ids_for_embedding's docstring, R2-GAP-005).
        return row[0] if row is not None else None

    def claim_next_message_embedding_job(self) -> dict | None:
        """Atomically claim one pending, due message-embedding job for processing.

        Identical mechanics to :meth:`claim_next_capture_job` — see its
        docstring.

        Returns:
            dict | None: ``{"id", "agent_id", "session_id", "message_id",
                "attempts"}``, or ``None`` if no job is currently due.
        """
        now = time.time()
        row = self._conn().execute(
            """
            UPDATE message_embedding_jobs
            SET state = 'running', updated_at = %s
            WHERE id = (
                SELECT id FROM message_embedding_jobs
                WHERE state = 'pending' AND run_after <= %s
                ORDER BY id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, agent_id, session_id, message_id, attempts
            """,
            (now, now),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "agent_id": row[1], "session_id": row[2],
            "message_id": row[3], "attempts": row[4],
        }

    def complete_message_embedding_job(
        self, job_id: int, message_id: int, model_identity: str, embedding: list[float],
    ) -> None:
        """Store the embedded vector and mark a message-embedding job done.

        Upserts into ``message_embeddings`` (``ON CONFLICT ... DO UPDATE``)
        rather than a plain insert, so retrying a job that partially
        completed (vector written, job marked done failed before this
        method returned) is safe to run again.

        Args:
            job_id: The job to complete.
            message_id: The message that was embedded — passed separately
                rather than re-queried, since the caller (the worker) has
                it already from :meth:`claim_next_message_embedding_job`.
            model_identity: Identifies which model/endpoint produced this
                vector (:attr:`EmbeddingProvider.model_identity`).
            embedding: The embedding vector itself.
        """
        now = time.time()
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO message_embeddings (message_id, model_identity, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (message_id, model_identity) DO UPDATE SET embedding = EXCLUDED.embedding
            """,
            (message_id, model_identity, embedding),
        )
        conn.execute(
            "UPDATE message_embedding_jobs SET state = 'done', updated_at = %s WHERE id = %s",
            (now, job_id),
        )

    def fail_message_embedding_job(
        self, job_id: int, error: str, backoff_seconds: float, max_attempts: int = 5,
    ) -> None:
        """Record a failed embedding attempt — retry with backoff, or give up.

        Identical mechanics to :meth:`fail_capture_job` — see its
        docstring.
        """
        now = time.time()
        conn = self._conn()
        row = conn.execute(
            "SELECT attempts FROM message_embedding_jobs WHERE id = %s", (job_id,)
        ).fetchone()
        attempts = (row[0] if row else 0) + 1
        state = "failed" if attempts >= max_attempts else "pending"
        conn.execute(
            """
            UPDATE message_embedding_jobs
            SET state = %s, attempts = %s, run_after = %s, last_error = %s, updated_at = %s
            WHERE id = %s
            """,
            (state, attempts, now + backoff_seconds, error, now, job_id),
        )

    def find_uncovered_message_ids_for_embedding(
        self, agent_id: str, session_id: str
    ) -> list[int]:
        """Return mirrored user/assistant message ids with no embedding under the active model.

        R2-GAP-005/R2-GAP-002: anti-joins directly against
        ``message_embeddings`` for :attr:`embedding_model_identity` —
        never a cursor derived from ``message_embedding_jobs``. This fixes
        two things at once: it's *model-aware* (switching the configured
        embedding model makes every historical message re-appear as
        uncovered for the new model, since ``message_embeddings`` is keyed
        by ``(message_id, model_identity)`` and an old model's rows simply
        don't match the new one — no separate "rebuild" operation needed,
        periodic reconciliation naturally backfills it), and it's *exact*
        (a genuine per-message check, not a high-water mark, so a sparse
        gap is found exactly like a tail one). Unlike the capture/
        commitment lanes, this needs no coverage-ranges table at all:
        ``message_embeddings`` is itself the durable, never-pruned record
        of what's actually been done, and "attempted but produced zero
        results" isn't a possible outcome here (unlike extraction) — a
        message either has a vector for the active model or it doesn't.

        Args:
            agent_id: The agent this session must belong to.
            session_id: The session to check.

        Returns:
            list[int]: Message ids missing an embedding under the active
                model, ascending, or ``[]`` if ``session_id`` isn't owned
                by ``agent_id``, no embedding model is configured, or
                there's nothing to catch up on.
        """
        if not self._session_owned_by(session_id, agent_id):
            return []
        # has_vector_lane guards against querying message_embeddings before
        # it exists — that table is only created when pgvector AND an
        # embedding provider are both configured (see _ensure_schema()).
        if not self.has_vector_lane:
            return []
        model_identity = self.embedding_model_identity
        if model_identity is None:
            return []
        # R2-GAP-011: bounded to the most recent _RECONCILIATION_SCAN_LIMIT
        # eligible messages, same reasoning as _eligible_message_ids' own
        # limit — DESC + LIMIT for a cheap bounded scan, reversed back to
        # ascending order in Python.
        rows = self._conn().execute(
            """
            SELECT m.id FROM messages m
            WHERE m.session_id = %s AND m.role IN ('user', 'assistant') AND m.content IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM message_embeddings me
                WHERE me.message_id = m.id AND me.model_identity = %s
            )
            ORDER BY m.id DESC LIMIT %s
            """,
            (session_id, model_identity, _RECONCILIATION_SCAN_LIMIT),
        ).fetchall()
        return [r[0] for r in reversed(rows)]

    def embedding_coverage_summary(self, agent_id: str) -> dict | None:
        """Count how many of one agent's messages are missing an embedding under the active model (R2-GAP-014).

        Unlike :meth:`find_uncovered_message_ids_for_embedding` (bounded to
        :data:`_RECONCILIATION_SCAN_LIMIT`, and scoped to one session — it
        exists to hand reconciliation a boundedly-sized list of ids to
        enqueue), this is a plain ``count(*)`` across *every* session this
        agent owns — no id list to build, so no memory-bound reason to
        limit it, and unlike the automatic periodic reconciliation pass,
        this only runs when something (``/status deep``) actually asks for
        it. Answers a different question than reconciliation's own
        progress: not "what should I enqueue right now" but "how far
        behind is this agent's embedding coverage overall, across its
        entire history" — visible operational context reconciliation
        itself has no reason to expose.

        Args:
            agent_id: Which agent's message history to check.

        Returns:
            dict | None: ``{"missing_count", "model_identity"}``, or
                ``None`` if no vector lane is configured (no pgvector, or
                no embedding provider) — "coverage" isn't a meaningful
                concept without an active model to measure it against.
        """
        if not self.has_vector_lane:
            return None
        model_identity = self.embedding_model_identity
        if model_identity is None:
            return None
        row = self._conn().execute(
            """
            SELECT count(*) FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE s.agent_id = %s AND m.role IN ('user', 'assistant') AND m.content IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM message_embeddings me
                WHERE me.message_id = m.id AND me.model_identity = %s
            )
            """,
            (agent_id, model_identity),
        ).fetchone()
        return {"missing_count": row[0] or 0, "model_identity": model_identity}

    # ------------------------------------------------------------------
    # Queue health (MEM-GAP-016)
    # ------------------------------------------------------------------

    def queue_lag_summary(self, agent_id: str) -> dict:
        """Report how far behind the capture/commitment job queues are for one agent.

        Unlike :class:`~minion_assist.worker_health.WorkerHealth` (in-process
        liveness, only visible to the same running process), this is a plain
        SQL aggregate over ``memory_capture_jobs``/``memory_commitment_jobs``
        — a fact anyone with a database connection can observe, including a
        separate ``minion-assist memory status --deep`` CLI invocation.
        "Pending" here means ``state = 'pending'``. ``running_count``/
        ``oldest_running_age_s`` (R2-GAP-003) separately surface jobs stuck
        in ``'running'`` — normally that state is transient (a worker holds
        it only for the duration of one claim-to-complete cycle), so a
        running job that's been sitting for a long time means its worker
        crashed mid-job; :meth:`reclaim_stale_running_jobs` is what recovers
        those, this method is only what makes them visible before that
        happens. ``failed_count`` (R2-GAP-014) surfaces jobs that gave up
        retrying entirely (``attempts`` reached ``max_attempts`` — see
        ``fail_capture_job``/``fail_commitment_job``/``fail_message_embedding_job``);
        those are the ones ``memory_retention`` will eventually prune, but
        until then they're real, permanently-lost work with nothing else
        making that visible.

        Args:
            agent_id: Which agent's queues to summarize.

        Returns:
            dict: ``{"capture": {...}, "commitment": {...},
                "message_embedding": {...}}``, each a dict with
                ``pending_count``, ``oldest_pending_age_s``,
                ``running_count``, ``oldest_running_age_s``,
                ``failed_count``.
                ``oldest_*_age_s`` is seconds since the oldest matching job
                was last updated, or ``None`` if that state is empty.
        """
        conn = self._conn()
        now = time.time()

        def _lane(table: str) -> dict:
            row = conn.execute(
                f"SELECT count(*), MIN(created_at) FROM {table} "
                "WHERE agent_id = %s AND state = 'pending'",
                (agent_id,),
            ).fetchone()
            pending_count = row[0] or 0
            oldest_created_at = row[1]
            oldest_pending_age_s = (
                now - oldest_created_at if oldest_created_at is not None else None
            )
            running_row = conn.execute(
                f"SELECT count(*), MIN(updated_at) FROM {table} "
                "WHERE agent_id = %s AND state = 'running'",
                (agent_id,),
            ).fetchone()
            running_count = running_row[0] or 0
            oldest_running_updated_at = running_row[1]
            oldest_running_age_s = (
                now - oldest_running_updated_at if oldest_running_updated_at is not None else None
            )
            failed_row = conn.execute(
                f"SELECT count(*) FROM {table} WHERE agent_id = %s AND state = 'failed'",
                (agent_id,),
            ).fetchone()
            failed_count = failed_row[0] or 0
            return {
                "pending_count": pending_count,
                "oldest_pending_age_s": oldest_pending_age_s,
                "running_count": running_count,
                "oldest_running_age_s": oldest_running_age_s,
                "failed_count": failed_count,
            }

        return {
            "capture": _lane("memory_capture_jobs"),
            "commitment": _lane("memory_commitment_jobs"),
            "message_embedding": _lane("message_embedding_jobs"),
        }

    def reclaim_stale_running_jobs(self, stale_after_seconds: float = 3600.0) -> dict:
        """Reset jobs stuck in ``'running'`` back to ``'pending'`` (R2-GAP-003).

        None of the three job queues use a claim lease/owner — a worker
        crash (or process kill) between claiming a job and completing it
        leaves that row in ``'running'`` forever, invisible to the normal
        pending-queue claim query and therefore never retried. Since this
        codebase runs exactly one worker per lane (never multiple
        concurrent claimants racing each other), a full owner/lease system
        would be more machinery than the actual failure mode warrants —
        the minimum fix that closes the gap is resetting anything that's
        been ``'running'`` for implausibly long back to ``'pending'`` once,
        at process startup, before any worker starts claiming.

        A job legitimately in ``'running'`` for the *current* process is
        never at risk of being reclaimed by this: workers claim and
        complete a job within a single method call (seconds, not the
        default one-hour threshold), and this is only ever called once at
        startup before workers begin claiming — never while workers are
        already running.

        Args:
            stale_after_seconds: How long a row may sit in ``'running'``
                (measured from its ``updated_at``, set at claim time)
                before it's considered abandoned and reset. Defaults to one
                hour — far longer than any real capture/commitment/
                embedding call is expected to take, so this only ever
                fires for a genuinely crashed worker.

        Returns:
            dict: ``{"capture_jobs", "commitment_jobs", "message_embedding_jobs"}``
                — number of rows reset in each table.
        """
        conn = self._conn()
        cutoff = time.time() - stale_after_seconds

        def _reclaim(table: str) -> int:
            cur = conn.execute(
                f"UPDATE {table} SET state = 'pending' "
                "WHERE state = 'running' AND updated_at < %s",
                (cutoff,),
            )
            return cur.rowcount

        return {
            "capture_jobs": _reclaim("memory_capture_jobs"),
            "commitment_jobs": _reclaim("memory_commitment_jobs"),
            "message_embedding_jobs": _reclaim("message_embedding_jobs"),
        }

    def prune_operational_tables(self, retention_days: float, dry_run: bool = False) -> dict:
        """Delete terminal-state job rows older than ``retention_days`` (MEM-GAP-015).

        Only ever deletes rows whose ``state`` is ``'done'`` or ``'failed'``
        (never ``'pending'``/``'running'``) and whose ``updated_at`` is
        older than the cutoff — a job still in flight or waiting to be
        retried is never touched regardless of age. This is pure
        operational bookkeeping (queue history, not memory content):
        deleting an old completed/failed job row does not delete the
        capture/commitment output it already produced (``memory_proposals``/
        ``commitments`` survive via ``ON DELETE SET NULL``, not ``CASCADE``
        — R2-GAP-001) and does not reset reconciliation's notion of what's
        already been covered (tracked separately in
        ``session_job_coverage_ranges``, which this method never touches —
        R2-GAP-002). Prior to R2-GAP-001 this docstring claimed outputs
        were simply "untouched" by this method, which was false: migration
        004's original ``ON DELETE CASCADE`` meant pruning a job *did*
        delete its proposal/commitment children, and the coverage cursor
        *was* derived from these same job rows. Both are now fixed at the
        schema/query level (migrations 005/006), which is what makes this docstring's claim
        accurate rather than aspirational.

        Args:
            retention_days: How many days a terminal-state row may age
                before it's deleted.
            dry_run: When ``True``, counts matching rows via ``SELECT
                count(*)`` instead of deleting them — same filter, same
                return shape, nothing removed. Used by
                ``minion-assist memory retention``'s dry-run report.

        Returns:
            dict: ``{"capture_jobs", "commitment_jobs", "message_embedding_jobs"}``
                — number of rows deleted (or, in dry-run mode, matching)
                in each table.
        """
        conn = self._conn()
        cutoff = time.time() - (retention_days * 86400)
        verb = "SELECT count(*) FROM" if dry_run else "DELETE FROM"

        def _prune(table: str) -> int:
            cur = conn.execute(
                f"{verb} {table} WHERE state IN ('done', 'failed') AND updated_at < %s",
                (cutoff,),
            )
            return cur.fetchone()[0] if dry_run else cur.rowcount

        return {
            "capture_jobs": _prune("memory_capture_jobs"),
            "commitment_jobs": _prune("memory_commitment_jobs"),
            "message_embedding_jobs": _prune("message_embedding_jobs"),
        }

    def find_uncovered_capture_ranges(
        self, agent_id: str, session_id: str
    ) -> list[tuple[int, int]]:
        """Return every gap of mirrored messages no capture job has ever covered.

        MEM-GAP-007: ``AgentSession._send_locked()``'s own
        ``enqueue_capture_job()`` call can fail (e.g. a transient
        connection drop) and, before this method existed, that turn's
        capture job would simply never exist — nothing ever retried it.
        :class:`~minion_assist.memory.reconciliation_scheduler.ReconciliationScheduler`
        calls this periodically per session and enqueues one catch-up job
        per gap it finds.

        R2-GAP-002: earlier versions of this method only ever looked at
        the single highest-covered id (a high-water mark), so a *sparse*
        gap — message 1's enqueue failed, message 2's later succeeded —
        was permanently invisible: 1 sits below the mark set by 2, so it
        read as "covered" forever. Using :func:`compute_backfill_windows`
        against every range :meth:`list_capture_job_ranges` has ever
        recorded finds every such gap exactly, tail or sparse, the same
        algorithm ``backfill_agent``'s one-off manual backfill already
        used — see that function's docstring for why "contiguous" means
        adjacent in this session's own eligible-message order, not
        adjacent integers.

        A job counts as "covering" its range regardless of its current
        ``state`` (pending/running/done/failed) — a failed extraction
        attempt still means a job was successfully *enqueued* for that
        range; re-enqueuing here would create a duplicate under a new
        idempotency key, not retry the original failed attempt.

        Args:
            agent_id: The agent this session must belong to.
            session_id: The session to check.

        Returns:
            list[tuple[int, int]]: ``(from_id, to_id)`` for every
                uncovered gap, each at most
                :data:`~minion_assist.session.db._BACKFILL_WINDOW_MESSAGES`
                messages wide, or ``[]`` if ``session_id`` isn't owned by
                ``agent_id`` or there's nothing to catch up on.
        """
        if not self._session_owned_by(session_id, agent_id):
            return []
        message_ids = self._eligible_message_ids(session_id)
        if not message_ids:
            return []
        covered = self.list_capture_job_ranges(session_id)
        return compute_backfill_windows(message_ids, covered)

    def find_uncovered_commitment_ranges(
        self, agent_id: str, session_id: str
    ) -> list[tuple[str, int, int]]:
        """Return every gap of mirrored messages no commitment job has ever covered.

        Same reasoning as :meth:`find_uncovered_capture_ranges` — see its
        docstring, including R2-GAP-002's sparse-gap fix.

        Commitment jobs are additionally channel-scoped (a Matrix room id,
        or ``"cli"``), so this only reconciles a session that has *at
        least one* prior commitment job to infer a channel from. A
        session with zero commitment jobs ever most likely means
        commitments were disabled (``config.json``'s
        ``commitments.enabled``) when its messages were created, not a
        missed enqueue — left alone rather than guessing at a channel.
        Every gap in one pass shares the single most-recently-enqueued
        channel — this method has no per-gap channel to fall back on for
        a range nothing was ever enqueued for, the same limitation the
        single-gap version of this method always had.

        Args:
            agent_id: The agent this session must belong to.
            session_id: The session to check.

        Returns:
            list[tuple[str, int, int]]: ``(channel, from_id, to_id)`` for
                every uncovered gap, or ``[]`` if ``session_id`` isn't
                owned by ``agent_id``, this session has no prior
                commitment job, or there's nothing to catch up on.
        """
        if not self._session_owned_by(session_id, agent_id):
            return []
        channel_row = self._conn().execute(
            "SELECT channel FROM session_job_coverage_ranges "
            "WHERE session_id = %s AND lane = 'commitment' "
            "ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if channel_row is None:
            return []
        channel = channel_row[0]
        message_ids = self._eligible_message_ids(session_id)
        if not message_ids:
            return []
        covered = self.list_commitment_job_ranges(session_id)
        windows = compute_backfill_windows(message_ids, covered)
        return [(channel, from_id, to_id) for from_id, to_id in windows]

    # ------------------------------------------------------------------
    # Commitment lifecycle — Stage One Phase 6, slice C
    # ------------------------------------------------------------------

    # How long past its own due_latest a still-pending commitment survives
    # before being auto-expired. OpenClaw's own shipped default
    # (ref-repos/openclaw/src/commitments/config.ts's
    # DEFAULT_COMMITMENT_EXPIRE_AFTER_HOURS), reused for the same
    # "no evaluation data of our own yet" reason
    # memory/commitments.py's confidence thresholds already document.
    _COMMITMENT_EXPIRE_AFTER_SECONDS = 72 * 3600.0

    def expire_stale_commitments(self, agent_id: str, now: float) -> int:
        """Mark any ``pending`` commitment whose window closed long ago as ``"expired"``.

        Called at the top of :meth:`list_due_commitments_for_agent` (lazy
        sweep on every read, mirroring OpenClaw's own
        ``expireStaleCommitments`` — verified against
        ``ref-repos/openclaw/src/commitments/store.ts``) rather than
        needing a dedicated expiry scheduler.

        Args:
            agent_id: Which agent's commitments to sweep.
            now: Epoch seconds "now" is evaluated at.

        Returns:
            int: How many commitments were just expired.
        """
        cutoff = now - self._COMMITMENT_EXPIRE_AFTER_SECONDS
        result = self._conn().execute(
            """
            UPDATE commitments
            SET status = 'expired', updated_at = %s
            WHERE agent_id = %s AND status = 'pending' AND due_latest < %s
            """,
            (now, agent_id, cutoff),
        )
        return result.rowcount

    def list_due_commitments_for_agent(
        self, agent_id: str, now: float, max_per_day: int = 3, max_per_heartbeat: int = 3
    ) -> list[dict]:
        """List commitments currently due for one agent, across every channel, rate-limited.

        "Due" means ``status = 'pending'`` and ``due_earliest <= now``.
        Unlike :meth:`list_pending_commitments_for_scope` (a single
        ``(agent_id, channel)`` scope, used for extraction-time
        deduplication), this spans every channel the agent has commitments
        in — the delivery side of "multi-room-aware": one heartbeat pass
        can see commitments from several different rooms at once, each
        later resolved to its own room at send time (see
        ``heartbeat.py``'s ``_deliver_to_channel``), rather than needing a
        separate heartbeat run per room.

        Rate-limited the same way OpenClaw's own
        ``listDueCommitmentsForSession`` is (verified against
        ``ref-repos/openclaw/src/commitments/store.ts``): capped at
        ``max_per_heartbeat`` per call *and* at ``max_per_day`` total
        ``"sent"`` commitments in the trailing 24 hours — whichever is
        smaller wins.

        Args:
            agent_id: Which agent's commitments to list.
            now: Epoch seconds "now" is evaluated at.
            max_per_day: Maximum commitments *sent* per rolling 24 hours.
            max_per_heartbeat: Maximum commitments returned by one call.

        Returns:
            list[dict]: ``{"id", "agent_id", "session_id", "channel",
                "kind", "sensitivity", "source", "reason",
                "suggested_text", "dedupe_key", "confidence",
                "due_earliest", "due_latest"}`` dicts, earliest-due first
                (ties broken by ``created_at`` then ``id`` for
                deterministic ordering) — empty once the day's quota is
                used up.
        """
        self.expire_stale_commitments(agent_id, now)
        conn = self._conn()
        sent_today = conn.execute(
            """
            SELECT count(*) FROM commitments
            WHERE agent_id = %s AND status = 'sent' AND sent_at >= %s
            """,
            (agent_id, now - 86400.0),
        ).fetchone()[0]
        remaining_today = max_per_day - sent_today
        if remaining_today <= 0:
            return []
        limit = min(max_per_heartbeat, remaining_today)

        rows = conn.execute(
            """
            SELECT id, agent_id, session_id, channel, kind, sensitivity, source, reason,
                   suggested_text, dedupe_key, confidence, due_earliest, due_latest
            FROM commitments
            WHERE agent_id = %s AND status = 'pending' AND due_earliest <= %s
            ORDER BY due_earliest ASC, created_at ASC, id ASC
            LIMIT %s
            """,
            (agent_id, now, limit),
        ).fetchall()
        return [
            {
                "id": r[0], "agent_id": r[1], "session_id": r[2], "channel": r[3],
                "kind": r[4], "sensitivity": r[5], "source": r[6], "reason": r[7],
                "suggested_text": r[8], "dedupe_key": r[9], "confidence": r[10],
                "due_earliest": r[11], "due_latest": r[12],
            }
            for r in rows
        ]

    def get_commitment(self, commitment_id: int) -> dict | None:
        """Look up one commitment by id.

        Returns:
            dict | None: ``{"id", "agent_id", "session_id", "channel",
                "kind", "sensitivity", "source", "status", "reason",
                "suggested_text", "dedupe_key", "confidence",
                "due_earliest", "due_latest"}``, or ``None`` if it doesn't
                exist.
        """
        row = self._conn().execute(
            """
            SELECT id, agent_id, session_id, channel, kind, sensitivity, source, status,
                   reason, suggested_text, dedupe_key, confidence, due_earliest, due_latest
            FROM commitments WHERE id = %s
            """,
            (commitment_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "agent_id": row[1], "session_id": row[2], "channel": row[3],
            "kind": row[4], "sensitivity": row[5], "source": row[6], "status": row[7],
            "reason": row[8], "suggested_text": row[9], "dedupe_key": row[10],
            "confidence": row[11], "due_earliest": row[12], "due_latest": row[13],
        }

    def mark_commitment_sent(self, commitment_id: int, now: float | None = None) -> None:
        """Mark a commitment ``"sent"`` — a check-in was actually delivered for it.

        Called by ``tools/commitment_response.py``'s
        ``RespondToCommitmentTool`` right after delivery succeeds.
        """
        now = time.time() if now is None else now
        self._conn().execute(
            "UPDATE commitments SET status = 'sent', sent_at = %s, updated_at = %s WHERE id = %s",
            (now, now, commitment_id),
        )

    def mark_commitment_dismissed(self, commitment_id: int, now: float | None = None) -> None:
        """Mark a commitment ``"dismissed"`` — reviewed and deliberately not sent.

        Called by ``tools/commitment_response.py``'s
        ``DismissCommitmentTool``.
        """
        now = time.time() if now is None else now
        self._conn().execute(
            "UPDATE commitments SET status = 'dismissed', dismissed_at = %s, updated_at = %s "
            "WHERE id = %s",
            (now, now, commitment_id),
        )

    def list_commitments(
        self, agent_id: str, status: str | None = None, channel: str | None = None
    ) -> list[dict]:
        """List commitments for one agent, most recently created first — the CLI's primitive.

        Args:
            agent_id: Which agent's commitments to list.
            status: Restrict to one status (``"pending"``/``"sent"``/
                ``"dismissed"``/``"snoozed"``/``"expired"``), or ``None``
                for every status.
            channel: Restrict to one channel, or ``None`` for every
                channel.

        Returns:
            list[dict]: Same shape as :meth:`get_commitment`'s return
                value, newest first.
        """
        clauses = ["agent_id = %s"]
        params: list = [agent_id]
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        if channel is not None:
            clauses.append("channel = %s")
            params.append(channel)
        where_sql = " AND ".join(clauses)
        rows = self._conn().execute(
            f"""
            SELECT id, agent_id, session_id, channel, kind, sensitivity, source, status,
                   reason, suggested_text, dedupe_key, confidence, due_earliest, due_latest
            FROM commitments
            WHERE {where_sql}
            ORDER BY created_at DESC
            """,
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "agent_id": r[1], "session_id": r[2], "channel": r[3],
                "kind": r[4], "sensitivity": r[5], "source": r[6], "status": r[7],
                "reason": r[8], "suggested_text": r[9], "dedupe_key": r[10],
                "confidence": r[11], "due_earliest": r[12], "due_latest": r[13],
            }
            for r in rows
        ]

    def delete_commitment(self, agent_id: str, commitment_id: int) -> bool:
        """Permanently delete one commitment — Task 7's "complete scoped deletion."

        Scoped to ``agent_id`` (not just the raw id) so an operator can't
        accidentally delete a different agent's commitment by guessing an
        id — the same scoping discipline ``memory/service.py``'s
        ``pin()``/``unpin()`` already applies.

        Args:
            agent_id: The commitment's owning agent — must match, or
                nothing is deleted.
            commitment_id: Which commitment to delete.

        Returns:
            bool: ``True`` if a row was actually deleted.
        """
        result = self._conn().execute(
            "DELETE FROM commitments WHERE id = %s AND agent_id = %s",
            (commitment_id, agent_id),
        )
        return result.rowcount > 0

    def list_pending_proposals(self, agent_id: str) -> list[dict]:
        """List every not-yet-reviewed proposal for one agent, oldest first.

        Stage One Phase 5, slice C: the source list
        ``memory/consolidation.py``'s ``rank_proposals`` scores and orders
        into a review queue. Only ``status = 'pending'`` rows are returned —
        a proposal already ``promoted``/``rejected``/``superseded`` (a later
        slice's review flow assigns these) has already been decided and
        shouldn't be re-offered for review.

        Args:
            agent_id: Which agent's proposals to list.

        Returns:
            list[dict]: ``{"id", "job_id", "agent_id", "claim_text",
                "created_at"}`` dicts, ordered by ``id`` (oldest first).
        """
        rows = self._conn().execute(
            """
            SELECT id, job_id, agent_id, claim_text, created_at
            FROM memory_proposals
            WHERE agent_id = %s AND status = 'pending'
            ORDER BY id
            """,
            (agent_id,),
        ).fetchall()
        return [
            {"id": r[0], "job_id": r[1], "agent_id": r[2], "claim_text": r[3], "created_at": r[4]}
            for r in rows
        ]

    def get_proposal(self, proposal_id: int) -> dict | None:
        """Look up one proposal by id.

        Stage One Phase 5, slice C: ``MemoryConsolidator.preview()`` needs a
        single proposal's claim text and status without listing every
        pending proposal first.

        Returns:
            dict | None: ``{"id", "job_id", "agent_id", "claim_text",
                "created_at", "status", "rejected_reason"}``, or ``None``
                if no proposal with this id exists.
        """
        row = self._conn().execute(
            """
            SELECT id, job_id, agent_id, claim_text, created_at, status, rejected_reason
            FROM memory_proposals
            WHERE id = %s
            """,
            (proposal_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "job_id": row[1], "agent_id": row[2], "claim_text": row[3],
            "created_at": row[4], "status": row[5], "rejected_reason": row[6],
        }
