"""``PostgresMemoryIndex`` — rebuildable lexical index over memory files.

Stage One Phase 3, slice A.

What this is, and why it's separate from ``session/db.py``
------------------------------------------------------------
``session/db.py``'s ``SessionDB`` already runs PostgreSQL full-text search —
but over *chat messages* (``messages.search_vector``), for the
``session_search`` tool. This module indexes something different: the
curated Markdown *memory files* a ``MemoryFileRepository`` owns (``MEMORY.md``,
topic notes, daily notes, imports). Those are a different corpus with a
different lifecycle (edited occasionally, not appended every turn), so this
gets its own tables and its own class rather than growing ``SessionDB``
further — matching the module split the plan lays out
(``minion-assist-docs/improve/memory-implementation-plan.md``'s "Stage One
proposed module changes").

Why chunk instead of indexing whole files?
--------------------------------------------
See ``memory/chunking.py``'s module docstring — the short version is that a
citation needs to point at a specific section once a file (typically
``MEMORY.md``) grows long, not just "somewhere in this file."

What this slice does *not* yet do
------------------------------------
This class only builds and rebuilds the index — nothing in the running app
calls it yet. Wiring (write-path sync, startup reconciliation, a live
filesystem watcher) is Stage One Phase 3 slice B; crash-safe rebuild-and-swap,
corpus-filtered search, and citations wired into ``MemoryService.search()``
are slice C. Building this in isolation first means the chunking and schema
logic can be tested against a real PostgreSQL instance without needing the
rest of the machinery to exist yet.

Talks to
--------
- ``memory/chunking.py`` — :func:`chunk_markdown`, this module's only
  chunking dependency.
- ``memory/files.py`` — :meth:`MemoryFileRepository.list_indexable_files`
  supplies the ``(source_kind, rel_path, content)`` triples
  :meth:`PostgresMemoryIndex.rebuild_agent` indexes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from datetime import date as _date
from typing import TYPE_CHECKING

from .boundaries import parse_frontmatter
from .chunking import Chunk, chunk_markdown
from .knowledge import parse_claims, parse_time_epoch

if TYPE_CHECKING:
    from ..providers.embeddings import EmbeddingProvider

_log = logging.getLogger("minion_assist.memory_index")

# Thread-local connection cache, kept separate from session/db.py's own
# thread-local so a psycopg connection meant for one module is never
# accidentally reused by the other.
_local = threading.local()

# ---------------------------------------------------------------------------
# Fusion helpers — Stage One Phase 4, slice C
# ---------------------------------------------------------------------------
# Free functions (not methods) since none of them touch the database — they
# operate purely on the lane results hybrid_search() already fetched.

# Standard reciprocal-rank-fusion constant. 60 is the value from the
# original RRF paper (Cormack et al.) and is not sensitive to tuning for
# typical result-list lengths — it just controls how quickly a lower rank's
# contribution falls off, not the fusion's correctness.
_RRF_K = 60

# Half-life for daily-note score decay: a daily note's fused score is
# multiplied by 0.5 for every _DECAY_HALF_LIFE_DAYS days since the date in
# its filename. Only source_kind "daily" decays — "durable" content
# (MEMORY.md, topic notes) does not, per Task 6's "evergreen MEMORY.md and
# topic pages do not decay merely because their file mtime is old."
_DECAY_HALF_LIFE_DAYS = 30

_DAILY_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.md$")


def _reciprocal_rank_fusion(lanes: list[list[dict]]) -> tuple[dict[int, float], dict[int, dict]]:
    """Combine several independently-ranked lanes into one score per chunk id.

    Each lane is a list of chunk dicts already ranked best-first (by
    whatever criterion that lane uses — ts_rank, cosine similarity, recency,
    exact-path match). A chunk's fused score is the sum, across every lane
    it appears in, of ``1 / (_RRF_K + rank)`` — so a chunk ranked highly by
    even one lane scores well, and a chunk multiple lanes agree on scores
    higher still. This is why every lane must be able to surface a
    candidate the others missed (the mem0 "reject semantic-seeded fusion"
    decision the plan cites): a chunk absent from a lane simply doesn't
    collect that lane's contribution, rather than being penalized for it.

    Args:
        lanes: One ranked list per lane. A lane may be empty (e.g. the
            vector lane with no embedding provider configured).

    Returns:
        tuple[dict[int, float], dict[int, dict]]: ``(scores, rows)`` —
            ``scores`` maps chunk id to its fused score, ``rows`` maps
            chunk id to the first lane's dict that mentioned it (all lanes
            describe the same underlying chunk, so any one's fields work).
    """
    scores: dict[int, float] = {}
    rows: dict[int, dict] = {}
    for lane in lanes:
        for rank, row in enumerate(lane, start=1):
            chunk_id = row["id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
            rows.setdefault(chunk_id, row)
    return scores, rows


def _decay_factor(rel_path: str, source_kind: str) -> float:
    """Score multiplier for temporal decay — 1.0 (no decay) for anything but daily notes.

    Args:
        rel_path: The chunk's file path, e.g. ``"memory/2026-07-20.md"``.
        source_kind: ``"durable"``, ``"daily"``, or ``"import"``.

    Returns:
        float: 1.0 for non-daily content, or ``0.5 ** (days_old /
            _DECAY_HALF_LIFE_DAYS)`` for a daily note — halving every
            ``_DECAY_HALF_LIFE_DAYS`` days since the date in its filename.
            Also 1.0 (no decay applied) if the filename doesn't parse as a
            date or the note is dated today/in the future — decay should
            never fail loudly over an unexpected filename shape.
    """
    if source_kind != "daily":
        return 1.0
    match = _DAILY_DATE_RE.search(rel_path)
    if not match:
        return 1.0
    try:
        note_date = _date.fromisoformat(match.group(1))
    except ValueError:
        return 1.0
    days_old = (_date.today() - note_date).days
    if days_old <= 0:
        return 1.0
    return 0.5 ** (days_old / _DECAY_HALF_LIFE_DAYS)


# Half-life for a claim's freshness (Stage One Phase 7, slice C's "stale
# claims" dashboard) — longer than daily-note decay above (90 vs 30 days):
# a knowledge claim is meant to be a more durable fact than a daily-note
# scratch entry, so it should take longer to be considered stale. A
# documented placeholder, not evaluated data, the same "collect data
# before choosing thresholds" posture the rest of Stage One has taken
# throughout.
_FRESHNESS_HALF_LIFE_DAYS = 90.0


def _freshness(observed_at: float, now: float) -> float:
    """A claim's freshness — 1.0 when just observed, halving every ``_FRESHNESS_HALF_LIFE_DAYS`` days since.

    Mirrors :func:`_decay_factor`'s shape but is unconditional (every
    claim has an ``observed_at``, unlike a chunk's ``source_kind``-gated
    decay) and query-time only — never stored (see
    ``memory/knowledge.py``'s module docstring for why ``freshness`` is
    deliberately not a marker field).

    Args:
        observed_at: The claim's ``observed_at`` (epoch seconds).
        now: Epoch seconds "now" is evaluated at.

    Returns:
        float: 1.0 for a claim observed at or after ``now`` (never
            negative "days old"), otherwise ``0.5 ** (days_old /
            _FRESHNESS_HALF_LIFE_DAYS)``.
    """
    days_old = (now - observed_at) / 86400.0
    if days_old <= 0:
        return 1.0
    return 0.5 ** (days_old / _FRESHNESS_HALF_LIFE_DAYS)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors, in [-1, 1] (0.0 if either is zero)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _hash_text(text: str) -> str:
    """SHA256 hex digest of a file's content — used to detect real changes.

    Mirrors ``memory/migration.py``'s ``_hash_bytes`` (same algorithm, same
    reasoning: a cheap, reliable way to tell "this file's content actually
    changed" apart from "this file's mtime changed for some other reason").
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_query(query: str) -> str:
    """Normalize and hash a search query — the recall-telemetry correlation key.

    Stage One Phase 5, slice A, Task 2: "hash normalized queries rather
    than storing unnecessary raw query text in promotion telemetry."
    Normalizing (lowercase, collapsed whitespace) before hashing means two
    queries that differ only in case or spacing count as the "same"
    query for recall-diversity purposes — what Task 3's later ranking
    calls "query diversity" cares about distinct *intents*, not
    incidental text formatting.

    Public (not a leading-underscore helper) because both
    :meth:`PostgresMemoryIndex.hybrid_search` (recording a surfaced
    result) and :meth:`~minion_assist.memory.service.MemoryService.mark_injected`
    (recording which of those were actually injected, later in the same
    turn) must compute the exact same hash for a given query to correlate
    with each other.
    """
    normalized = " ".join(query.lower().split())
    return _hash_text(normalized)


class PostgresMemoryIndex:
    """Rebuildable PostgreSQL lexical index over one or more agents' memory files.

    Args:
        url: libpq connection string — the same ``database.url`` config
            value :class:`~minion_assist.session.db.SessionDB` uses. Opening
            a second connection to the same database (rather than sharing
            ``SessionDB``'s) keeps this module fully independent, at the
            cost of one extra idle connection per thread — a deliberate,
            simple trade given how rarely indexing operations happen
            relative to message traffic.
        embedding_dimensions: The configured embedding model's vector
            length (``config.embeddings.dimensions``), or ``None`` if no
            embedding backend is configured. Needed up front because a
            pgvector column's width is fixed at ``CREATE TABLE`` time — it
            can't be discovered later from the data itself. When ``None``
            (the default — matches every call site before Stage One
            Phase 4), :attr:`memory_chunk_embeddings` is never created and
            the vector lane simply doesn't exist for this instance.
        embedding_provider: The :class:`EmbeddingProvider` to call when
            indexing a chunk with no cached embedding yet (Stage One
            Phase 4, slice C). ``None`` (the default) means chunks are
            never embedded — the vector lane stays empty even if
            ``embedding_dimensions`` created the table, exactly as if no
            embedding backend were configured at all.
    """

    def __init__(
        self,
        url: str,
        embedding_dimensions: int | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        import psycopg  # noqa: PLC0415

        self._url = url
        self._psycopg = psycopg
        self._embedding_dimensions = embedding_dimensions
        self._embedding_provider = embedding_provider
        self._has_vector = False
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self):
        """Return a thread-local autocommit connection, creating it if needed.

        When ``self._has_vector`` is set, registers pgvector's psycopg
        adapter on every new connection — without it, a ``vector`` column
        round-trips as a raw ``"[0.1,0.2,0.3]"`` string on read instead of a
        usable :class:`pgvector.vector.Vector`, and (less critically) an
        outgoing Python list wouldn't reliably be recognized as a vector
        value on write either.
        """
        conn = getattr(_local, "conn", None)
        if conn is None or conn.closed:
            conn = self._psycopg.connect(self._url, autocommit=True)
            if self._has_vector:
                from pgvector.psycopg import register_vector  # noqa: PLC0415

                register_vector(conn)
            _local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        conn = self._conn()
        # pgvector extension (optional — silently skipped if not available,
        # same pattern as session/db.py's SessionDB._has_vector). Registered
        # on *this* connection immediately since _conn() only registers it
        # for connections created *after* self._has_vector becomes True —
        # this first connection was already open before that happened.
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._has_vector = True
            from pgvector.psycopg import register_vector  # noqa: PLC0415

            register_vector(conn)
        except Exception:
            pass

        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_chunks (
                id            BIGSERIAL PRIMARY KEY,
                agent_id      TEXT NOT NULL,
                source_kind   TEXT NOT NULL,
                rel_path      TEXT NOT NULL,
                chunk_index   INTEGER NOT NULL,
                heading_path  TEXT NOT NULL DEFAULT '',
                content       TEXT NOT NULL,
                start_line    INTEGER NOT NULL,
                end_line      INTEGER NOT NULL,
                chunk_hash    TEXT NOT NULL,
                search_vector tsvector GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', coalesce(heading_path, '')), 'A') ||
                    setweight(to_tsvector('english', coalesce(content, '')), 'B')
                ) STORED
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_chunks_fts_idx "
            "ON memory_chunks USING GIN (search_vector)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_chunks_file_idx "
            "ON memory_chunks (agent_id, rel_path)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_chunks_corpus_idx "
            "ON memory_chunks (agent_id, source_kind)"
        )

        # Per-file reconciliation ledger — same role as session/db.py's
        # message_mirrors: lets a later slice B diff "what's on disk" against
        # "what's indexed" by content hash rather than reindexing everything
        # unconditionally on every startup.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_files (
                agent_id     TEXT NOT NULL,
                rel_path     TEXT NOT NULL,
                source_kind  TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at   DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (agent_id, rel_path)
            )
        """)
        # boundary_metadata added in Stage One Phase 6, slice A — ADD COLUMN
        # IF NOT EXISTS so a database that already has this table from an
        # earlier phase picks it up without a manual migration. JSON-encoded
        # dict from memory/boundaries.py's parse_frontmatter(); '' means the
        # file has no action-boundary frontmatter. The file itself is the
        # source of truth (a human can hand-edit the frontmatter block
        # directly) — this column is a cache refreshed on every reindex, the
        # same relationship content_hash already has to the file's raw text.
        conn.execute(
            "ALTER TABLE memory_files ADD COLUMN IF NOT EXISTS "
            "boundary_metadata TEXT NOT NULL DEFAULT ''"
        )

        # Pinned files — Stage One Phase 4, slice B. Backs the "pinned" fusion
        # lane (slice C): a pinned file's chunks are always surfaced,
        # regardless of query match. Scoped to topic notes only (see
        # memory/service.py's pin()/unpin() docstrings for why).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_pins (
                agent_id   TEXT NOT NULL,
                rel_path   TEXT NOT NULL,
                pinned_at  DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (agent_id, rel_path)
            )
        """)

        # Recall telemetry — Stage One Phase 5, slice A. Keyed by rel_path,
        # not a memory_chunks row id: chunk ids aren't stable across a
        # reindex (the same lesson learned fixing the embedding cache in
        # Phase 4), and promotion decisions (Phase 5 slice C) operate at
        # the file/note level anyway. query_hash is a normalized-and-hashed
        # query (Task 2: "hash normalized queries rather than storing
        # unnecessary raw query text") — see hash_query().
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_recall_events (
                id           BIGSERIAL PRIMARY KEY,
                agent_id     TEXT NOT NULL,
                rel_path     TEXT NOT NULL,
                query_hash   TEXT NOT NULL,
                surfaced_at  DOUBLE PRECISION NOT NULL,
                was_injected BOOLEAN NOT NULL DEFAULT FALSE
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_recall_events_file_idx "
            "ON memory_recall_events (agent_id, rel_path)"
        )

        # Consolidation previews — Stage One Phase 5, slice C. A row is a
        # draft, never an applied change: MemoryConsolidator.preview() never
        # writes to disk, it only records what it *would* write, for a human
        # to review (a later slice adds apply/reject/rollback on top of
        # these rows). No FOREIGN KEY to memory_proposals — this codebase
        # never declares cross-table FKs (see memory_proposals.job_id for
        # the same plain-BIGINT precedent), partly because memory_proposals
        # is owned by session/db.py's SessionDB, a separate connection/class
        # entirely. based_on_content_hash captures the target file's content
        # hash *at preview time* — Task 6's "detect human edits before
        # applying a stale proposal" needs to compare this against the
        # file's hash again at apply time, which only works if it was
        # captured when the preview was drafted, not reconstructed later.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_consolidation_previews (
                id                    BIGSERIAL PRIMARY KEY,
                agent_id              TEXT NOT NULL,
                proposal_id           BIGINT NOT NULL,
                target_kind           TEXT NOT NULL,
                target_key            TEXT NOT NULL,
                based_on_content_hash TEXT NOT NULL,
                drafted_content       TEXT NOT NULL,
                rationale             TEXT NOT NULL,
                created_at            DOUBLE PRECISION NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_consolidation_previews_proposal_idx "
            "ON memory_consolidation_previews (agent_id, proposal_id)"
        )

        # Topic revision history — Stage One Phase 5, slice D. A row is a
        # snapshot of a topic note's content taken right BEFORE
        # MemoryConsolidator.approve() overwrites it, so rollback() can
        # restore exactly what was there (an empty string means the topic
        # didn't exist before this apply — rollback() deletes the file
        # rather than writing "" back). Acts as a one-entry-per-apply undo
        # stack: rollback() consumes (deletes) the row it restores from,
        # so repeated rollbacks step back through history one apply at a
        # time, the same shape as a normal undo stack.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_topic_revisions (
                id            BIGSERIAL PRIMARY KEY,
                agent_id      TEXT NOT NULL,
                target_key    TEXT NOT NULL,
                proposal_id   BIGINT NOT NULL,
                prior_content TEXT NOT NULL,
                created_at    DOUBLE PRECISION NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_topic_revisions_key_idx "
            "ON memory_topic_revisions (agent_id, target_key)"
        )

        # Import review previews — Stage One Phase 7, slice E. Structurally
        # identical to memory_consolidation_previews above, but deliberately
        # a SEPARATE table rather than a shared one: an import is identified
        # by a TEXT key (e.g. "_auto_extracted"), not the BIGINT proposal_id
        # memory_consolidation_previews/memory_topic_revisions are typed
        # around, and ImportReviewer.approve()/reject() delete the reviewed
        # import file outright rather than flipping a memory_proposals
        # status — there is no rollback for import review in this slice
        # (see memory/import_review.py's module docstring), so no
        # import-side revision-history table exists either.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_import_previews (
                id                    BIGSERIAL PRIMARY KEY,
                agent_id              TEXT NOT NULL,
                import_key            TEXT NOT NULL,
                target_kind           TEXT NOT NULL,
                target_key            TEXT NOT NULL,
                based_on_content_hash TEXT NOT NULL,
                drafted_content       TEXT NOT NULL,
                rationale             TEXT NOT NULL,
                created_at            DOUBLE PRECISION NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_import_previews_key_idx "
            "ON memory_import_previews (agent_id, import_key)"
        )

        # Shadow copies of both tables above, same shape — scratch space for
        # force_rebuild_agent()'s crash-safe rebuild-and-swap (Stage One
        # Phase 3, slice C). See that method's docstring for the swap
        # mechanics. Never queried by search()/chunk_count()/etc. — only
        # force_rebuild_agent() ever touches these.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_chunks_shadow (
                id            BIGSERIAL PRIMARY KEY,
                agent_id      TEXT NOT NULL,
                source_kind   TEXT NOT NULL,
                rel_path      TEXT NOT NULL,
                chunk_index   INTEGER NOT NULL,
                heading_path  TEXT NOT NULL DEFAULT '',
                content       TEXT NOT NULL,
                start_line    INTEGER NOT NULL,
                end_line      INTEGER NOT NULL,
                chunk_hash    TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_chunks_shadow_agent_idx "
            "ON memory_chunks_shadow (agent_id)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_files_shadow (
                agent_id     TEXT NOT NULL,
                rel_path     TEXT NOT NULL,
                source_kind  TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at   DOUBLE PRECISION NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_files_shadow_agent_idx "
            "ON memory_files_shadow (agent_id)"
        )
        # Same self-healing ADD COLUMN IF NOT EXISTS as memory_files above —
        # this scratch table's CREATE TABLE IF NOT EXISTS is a no-op on an
        # existing database, so the column needs adding explicitly too.
        conn.execute(
            "ALTER TABLE memory_files_shadow ADD COLUMN IF NOT EXISTS "
            "boundary_metadata TEXT NOT NULL DEFAULT ''"
        )

        # Embedding cache — only created when pgvector is available AND an
        # embedding backend is actually configured (Stage One Phase 4,
        # slice A). The vector width is fixed at creation time from the
        # configured model's dimensions; f-string interpolation is required
        # here because PostgreSQL's vector(N) type modifier can't be a bind
        # parameter, but embedding_dimensions is a validated int from
        # config.py, never external input, so this is safe.
        #
        # Keyed by (content_hash, model_identity), NOT chunk_id or agent_id.
        # reindex_file() deletes and reinserts every chunk on every call
        # (even when only one changed), so a chunk_id-keyed cache could
        # never hit — Task 3's "cache embeddings by model identity and
        # content hash" literally means content, not row identity.
        # memory_chunks.chunk_hash already stores this same hash per row,
        # so the vector lane joins through that column rather than needing
        # its own chunk_id reference. No agent_id either: identical chunk
        # text embeds identically regardless of which agent's note it came
        # from, so a cache hit is shared across agents for free.
        if self._has_vector and self._embedding_dimensions:
            # Self-healing one-time migration: an earlier build of this
            # table (before this content-hash-keyed shape existed) used
            # chunk_id as its primary key. Nothing ever wrote real
            # embeddings under that shape — it was never wired into any
            # write path until now — so it's safe to detect and drop it
            # here rather than requiring a manual fix on any machine that
            # already ran the earlier code.
            old_pk_cols = conn.execute("""
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = 'memory_chunk_embeddings'
                  AND tc.constraint_type = 'PRIMARY KEY'
            """).fetchall()
            if old_pk_cols and {r[0] for r in old_pk_cols} == {"chunk_id"}:
                conn.execute("DROP TABLE memory_chunk_embeddings")

            dims = int(self._embedding_dimensions)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS memory_chunk_embeddings (
                    content_hash   TEXT NOT NULL,
                    model_identity TEXT NOT NULL,
                    embedding      vector({dims}) NOT NULL,
                    PRIMARY KEY (content_hash, model_identity)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS memory_chunk_embeddings_hnsw "
                "ON memory_chunk_embeddings USING hnsw (embedding vector_cosine_ops)"
            )

        # Knowledge layer — Stage One Phase 7, slice A. No kb_pages table:
        # a topic note's (agent_id, rel_path) is already the stable "page"
        # identifier every other part of this project uses (memory_files,
        # memory_pins, ...) — see memory/knowledge.py's module docstring
        # for why a separate pages table would just duplicate that.
        #
        # Entity ids are system-assigned (get_or_create_entity), not
        # human-authored like claim ids — a claim marker only names an
        # entity by (free-text, case-insensitive) name; name_normalized is
        # the actual dedup key, name preserves whichever casing first
        # created the row.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kb_entities (
                id              TEXT PRIMARY KEY,
                agent_id        TEXT NOT NULL,
                name            TEXT NOT NULL,
                name_normalized TEXT NOT NULL,
                created_at      DOUBLE PRECISION NOT NULL,
                UNIQUE (agent_id, name_normalized)
            )
        """)

        # Claim ids are human/model-authored, embedded in the canonical
        # Markdown page itself (memory/knowledge.py's parse_claims()) —
        # this table is a synced cache of what the pages currently say,
        # never the other way around. entity_id has no FOREIGN KEY (this
        # project never declares cross-table FKs — see
        # memory_proposals.job_id for the established precedent).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kb_claims (
                id           TEXT PRIMARY KEY,
                agent_id     TEXT NOT NULL,
                rel_path     TEXT NOT NULL,
                entity_id    TEXT,
                text         TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'unknown',
                confidence   DOUBLE PRECISION,
                observed_at  DOUBLE PRECISION,
                valid_from   DOUBLE PRECISION,
                valid_to     DOUBLE PRECISION,
                privacy_tier TEXT NOT NULL DEFAULT '',
                line_number  INTEGER NOT NULL,
                created_at   DOUBLE PRECISION NOT NULL,
                updated_at   DOUBLE PRECISION NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS kb_claims_page_idx ON kb_claims (agent_id, rel_path)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS kb_claims_status_idx ON kb_claims (agent_id, status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS kb_claims_entity_idx ON kb_claims (agent_id, entity_id)"
        )

        # A claim's evidence= field, one row per (kind, ref) pair — e.g.
        # ("proposal", "42"). A claim with zero evidence rows is exactly
        # what Task 3's provenance-gap dashboard (a later slice) reports.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kb_evidence (
                id          BIGSERIAL PRIMARY KEY,
                agent_id    TEXT NOT NULL,
                claim_id    TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                source_ref  TEXT NOT NULL,
                created_at  DOUBLE PRECISION NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS kb_evidence_claim_idx ON kb_evidence (agent_id, claim_id)"
        )

        # Claim relationships — Stage One Phase 7, slice B. Deliberately
        # only two kinds ("supersedes"/"contradicts") — see
        # memory/knowledge.py's module docstring for why this isn't an
        # open-ended relationship-type system. No FOREIGN KEY on
        # from_claim_id/to_claim_id (this project never declares
        # cross-table FKs); to_claim_id may reference a claim that
        # doesn't (or no longer) exists — e.g. a hand-written
        # contradicts= referencing a typo'd id — which is itself useful
        # information for a later slice's dashboards to surface, not
        # something to silently reject at sync time.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kb_relationships (
                id            BIGSERIAL PRIMARY KEY,
                agent_id      TEXT NOT NULL,
                from_claim_id TEXT NOT NULL,
                to_claim_id   TEXT NOT NULL,
                kind          TEXT NOT NULL,
                created_at    DOUBLE PRECISION NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS kb_relationships_from_idx "
            "ON kb_relationships (agent_id, from_claim_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS kb_relationships_to_idx "
            "ON kb_relationships (agent_id, to_claim_id)"
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def reindex_file(self, agent_id: str, rel_path: str, source_kind: str, content: str) -> int:
        """(Re)index one file: replace its chunks and update the reconciliation ledger.

        Always deletes any existing chunks for this ``(agent_id, rel_path)``
        first, then inserts fresh ones — simpler and safer than trying to
        diff old vs. new chunk boundaries, and cheap enough given these are
        curated notes, not a high-frequency write path.

        Args:
            agent_id: The owning agent.
            rel_path: Path relative to the agent's workspace root (as
                returned by :meth:`MemoryFileRepository.list_indexable_files`).
            source_kind: ``"durable"``, ``"daily"``, or ``"import"``.
            content: The file's current full text.

        Returns:
            int: How many chunks were written (0 for empty/whitespace-only
                content — the file's ledger row is still updated so a
                since-emptied file doesn't linger as stale chunks).
        """
        conn = self._conn()
        conn.execute(
            "DELETE FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        )

        # Stage One Phase 6, slice A: strip any action-boundary frontmatter
        # before chunking so it never becomes searchable/embeddable body
        # text — only chunk_markdown()'s citations need adjusting, via
        # line_offset, so they still point at the right line in the
        # *original* file on disk (the frontmatter block still occupies
        # real lines there).
        boundary_metadata, body, line_offset = parse_frontmatter(content)
        chunks = chunk_markdown(body)
        for i, chunk in enumerate(chunks):
            conn.execute(
                """
                INSERT INTO memory_chunks
                    (agent_id, source_kind, rel_path, chunk_index, heading_path,
                     content, start_line, end_line, chunk_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    agent_id,
                    source_kind,
                    rel_path,
                    i,
                    " > ".join(chunk.heading_path),
                    chunk.content,
                    chunk.start_line + line_offset,
                    chunk.end_line + line_offset,
                    _hash_text(chunk.content),
                ),
            )

        conn.execute(
            """
            INSERT INTO memory_files
                (agent_id, rel_path, source_kind, content_hash, indexed_at, boundary_metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (agent_id, rel_path)
            DO UPDATE SET source_kind = EXCLUDED.source_kind,
                          content_hash = EXCLUDED.content_hash,
                          indexed_at = EXCLUDED.indexed_at,
                          boundary_metadata = EXCLUDED.boundary_metadata
            """,
            (
                agent_id, rel_path, source_kind, _hash_text(content), time.time(),
                json.dumps(boundary_metadata) if boundary_metadata else "",
            ),
        )
        self._maybe_embed_chunks(chunks)
        if source_kind == "durable":
            # Stage One Phase 7, slice A: claim markers are a curated-
            # knowledge concept — daily notes and unreviewed imports are
            # deliberately excluded (an import only gets claims once a
            # later slice's review flow promotes it into a durable page).
            self._sync_claims(agent_id, rel_path, body, line_offset)
        return len(chunks)

    def get_or_create_entity(self, agent_id: str, name: str) -> str:
        """Resolve a claim marker's ``entity=`` name to a stable entity id, creating one if needed.

        Matched by exact, case-insensitive name within the agent's scope
        — deliberately no fuzzy entity resolution/merging (see
        ``memory/knowledge.py``'s module docstring for why). Race-safe:
        two concurrent callers creating the same new entity name both end
        up returning the same id (``ON CONFLICT DO NOTHING`` + re-select).

        Args:
            agent_id: The agent this entity belongs to.
            name: The entity's name, as written in a claim marker's
                ``entity=`` field. Whitespace-trimmed; matched
                case-insensitively.

        Returns:
            str: The entity's stable id (``"e-" + 8 hex chars``) — newly
                generated if this exact name (case-insensitively) has
                never been seen for this agent before.
        """
        normalized = name.strip().lower()
        conn = self._conn()
        row = conn.execute(
            "SELECT id FROM kb_entities WHERE agent_id = %s AND name_normalized = %s",
            (agent_id, normalized),
        ).fetchone()
        if row is not None:
            return row[0]
        new_id = f"e-{uuid.uuid4().hex[:8]}"
        conn.execute(
            """
            INSERT INTO kb_entities (id, agent_id, name, name_normalized, created_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (agent_id, name_normalized) DO NOTHING
            """,
            (new_id, agent_id, name.strip(), normalized, time.time()),
        )
        row = conn.execute(
            "SELECT id FROM kb_entities WHERE agent_id = %s AND name_normalized = %s",
            (agent_id, normalized),
        ).fetchone()
        return row[0]

    def _sync_claims(self, agent_id: str, rel_path: str, body: str, line_offset: int) -> int:
        """Sync one page's claim markers into ``kb_claims``/``kb_evidence``/``kb_relationships``.

        Stage One Phase 7, slices A-B. Read-only relative to the page
        itself — never writes a marker back into the file (see
        ``memory/knowledge.py``'s module docstring: the file is always
        the source of truth, this is a derived cache refreshed on every
        reindex, the same relationship ``memory_chunks``/
        ``boundary_metadata`` already have to a file's raw text). Claims
        whose markers have been removed from the page since the last
        sync are deleted here too (and their evidence/relationships with
        them) — the same "diff and remove" shape ``remove_file()``/
        ``reconcile_agent()`` already use for whole files.

        Args:
            agent_id: The owning agent.
            rel_path: The page this content came from.
            body: The page's content with any action-boundary frontmatter
                already stripped (Stage One Phase 6, slice A) — the same
                text ``chunk_markdown()`` chunks.
            line_offset: Lines to add back onto each claim's reported line
                number so it points at the right line in the *original*
                file on disk, mirroring how chunk citations already
                account for a stripped frontmatter block.

        Returns:
            int: How many claims are now recorded for this page.
        """
        parsed = parse_claims(body)
        conn = self._conn()
        now = time.time()

        existing_ids = {
            r[0]
            for r in conn.execute(
                "SELECT id FROM kb_claims WHERE agent_id = %s AND rel_path = %s",
                (agent_id, rel_path),
            ).fetchall()
        }
        current_ids = {c.id for c in parsed}
        for removed_id in existing_ids - current_ids:
            conn.execute(
                "DELETE FROM kb_evidence WHERE agent_id = %s AND claim_id = %s",
                (agent_id, removed_id),
            )
            conn.execute(
                "DELETE FROM kb_relationships WHERE agent_id = %s AND from_claim_id = %s",
                (agent_id, removed_id),
            )
            conn.execute(
                "DELETE FROM kb_claims WHERE agent_id = %s AND id = %s", (agent_id, removed_id)
            )

        for claim in parsed:
            entity_id = self.get_or_create_entity(agent_id, claim.entity) if claim.entity else None
            observed_at = parse_time_epoch(claim.observed)
            if observed_at is None:
                observed_at = now
            conn.execute(
                """
                INSERT INTO kb_claims
                    (id, agent_id, rel_path, entity_id, text, status, confidence,
                     observed_at, valid_from, valid_to, privacy_tier, line_number,
                     created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    rel_path = EXCLUDED.rel_path,
                    entity_id = EXCLUDED.entity_id,
                    text = EXCLUDED.text,
                    status = EXCLUDED.status,
                    confidence = EXCLUDED.confidence,
                    observed_at = EXCLUDED.observed_at,
                    valid_from = EXCLUDED.valid_from,
                    valid_to = EXCLUDED.valid_to,
                    privacy_tier = EXCLUDED.privacy_tier,
                    line_number = EXCLUDED.line_number,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    claim.id, agent_id, rel_path, entity_id, claim.text, claim.status,
                    claim.confidence, observed_at, parse_time_epoch(claim.valid_from),
                    parse_time_epoch(claim.valid_to), claim.privacy, claim.line + line_offset,
                    now, now,
                ),
            )
            conn.execute(
                "DELETE FROM kb_evidence WHERE agent_id = %s AND claim_id = %s",
                (agent_id, claim.id),
            )
            for source_kind, source_ref in claim.evidence:
                conn.execute(
                    """
                    INSERT INTO kb_evidence (agent_id, claim_id, source_kind, source_ref, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (agent_id, claim.id, source_kind, source_ref, now),
                )
            conn.execute(
                "DELETE FROM kb_relationships WHERE agent_id = %s AND from_claim_id = %s",
                (agent_id, claim.id),
            )
            for to_claim_id in claim.supersedes:
                conn.execute(
                    """
                    INSERT INTO kb_relationships (agent_id, from_claim_id, to_claim_id, kind, created_at)
                    VALUES (%s, %s, %s, 'supersedes', %s)
                    """,
                    (agent_id, claim.id, to_claim_id, now),
                )
            for to_claim_id in claim.contradicts:
                conn.execute(
                    """
                    INSERT INTO kb_relationships (agent_id, from_claim_id, to_claim_id, kind, created_at)
                    VALUES (%s, %s, %s, 'contradicts', %s)
                    """,
                    (agent_id, claim.id, to_claim_id, now),
                )
        return len(parsed)

    def get_claim(self, agent_id: str, claim_id: str) -> dict | None:
        """Look up one claim by id, including its evidence and outgoing relationships.

        Returns:
            dict | None: ``{"id", "agent_id", "rel_path", "entity_id",
                "text", "status", "confidence", "observed_at",
                "valid_from", "valid_to", "privacy_tier", "line_number",
                "evidence", "supersedes", "contradicts"}`` — ``evidence``
                is a list of ``{"source_kind", "source_ref"}`` dicts;
                ``supersedes``/``contradicts`` are lists of claim ids
                this claim points *to* (Stage One Phase 7, slice B — not
                claims pointing at this one; use
                :meth:`list_relationships_to` for the reverse direction).
                ``None`` if no claim with this id exists for this agent.
        """
        row = self._conn().execute(
            """
            SELECT id, agent_id, rel_path, entity_id, text, status, confidence,
                   observed_at, valid_from, valid_to, privacy_tier, line_number
            FROM kb_claims WHERE agent_id = %s AND id = %s
            """,
            (agent_id, claim_id),
        ).fetchone()
        if row is None:
            return None
        evidence_rows = self._conn().execute(
            "SELECT source_kind, source_ref FROM kb_evidence WHERE agent_id = %s AND claim_id = %s",
            (agent_id, claim_id),
        ).fetchall()
        relationship_rows = self._conn().execute(
            "SELECT kind, to_claim_id FROM kb_relationships WHERE agent_id = %s AND from_claim_id = %s",
            (agent_id, claim_id),
        ).fetchall()
        return {
            "id": row[0], "agent_id": row[1], "rel_path": row[2], "entity_id": row[3],
            "text": row[4], "status": row[5], "confidence": row[6], "observed_at": row[7],
            "valid_from": row[8], "valid_to": row[9], "privacy_tier": row[10],
            "line_number": row[11],
            "evidence": [{"source_kind": r[0], "source_ref": r[1]} for r in evidence_rows],
            "supersedes": [r[1] for r in relationship_rows if r[0] == "supersedes"],
            "contradicts": [r[1] for r in relationship_rows if r[0] == "contradicts"],
        }

    def list_relationships_to(self, agent_id: str, claim_id: str) -> list[dict]:
        """Find every claim that points *at* ``claim_id`` (Stage One Phase 7, slice B).

        The reverse of :meth:`get_claim`'s ``supersedes``/``contradicts``
        fields — e.g. "which claims contradict claim X," not "what does
        claim X contradict." A later slice's contradiction dashboard
        needs both directions to show a conflict from either claim's
        perspective.

        Returns:
            list[dict]: ``{"from_claim_id", "kind"}`` dicts.
        """
        rows = self._conn().execute(
            "SELECT from_claim_id, kind FROM kb_relationships WHERE agent_id = %s AND to_claim_id = %s",
            (agent_id, claim_id),
        ).fetchall()
        return [{"from_claim_id": r[0], "kind": r[1]} for r in rows]

    def list_claims(
        self, agent_id: str, rel_path: str | None = None, status: str | None = None
    ) -> list[dict]:
        """List claims for one agent, optionally scoped to one page and/or status.

        Does not include evidence (use :meth:`get_claim` per-claim for
        that) — this is the lightweight listing primitive later slices'
        dashboards build on.

        Returns:
            list[dict]: Same shape as :meth:`get_claim`'s return value
                minus ``evidence``, ordered by ``rel_path`` then
                ``line_number``.
        """
        clauses = ["agent_id = %s"]
        params: list = [agent_id]
        if rel_path is not None:
            clauses.append("rel_path = %s")
            params.append(rel_path)
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        where_sql = " AND ".join(clauses)
        rows = self._conn().execute(
            f"""
            SELECT id, agent_id, rel_path, entity_id, text, status, confidence,
                   observed_at, valid_from, valid_to, privacy_tier, line_number
            FROM kb_claims
            WHERE {where_sql}
            ORDER BY rel_path, line_number
            """,
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "agent_id": r[1], "rel_path": r[2], "entity_id": r[3],
                "text": r[4], "status": r[5], "confidence": r[6], "observed_at": r[7],
                "valid_from": r[8], "valid_to": r[9], "privacy_tier": r[10],
                "line_number": r[11],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Knowledge dashboards — Stage One Phase 7, slices C & F (Task 3)
    # ------------------------------------------------------------------
    #
    # "Open questions" (status="unknown") needs no dedicated method —
    # list_claims(agent_id, status="unknown") already covers it.
    # "Deletion coverage" (list_claims_needing_reevaluation, added slice
    # F below) was deferred from slice C to here — there was nothing to
    # report on before memory/forgetting.py's forget_source() existed to
    # produce claims in this state.

    def list_contradictions(self, agent_id: str) -> list[dict]:
        """List every recorded ``contradicts`` relationship, with both claims' current text/status.

        Not deduplicated by pair — a hand-authored note could record
        both directions (A contradicts B *and* B contradicts A), which
        would show as two rows here rather than being silently merged
        into one; a human reviewing this report can tell they're the
        same conflict from the claim ids shown.

        Returns:
            list[dict]: ``{"from_claim_id", "from_text", "from_status",
                "to_claim_id", "to_text", "to_status"}``. ``to_text``/
                ``to_status`` are ``None`` when ``to_claim_id`` doesn't
                resolve to a real claim — a dangling reference (e.g. a
                typo in a hand-written ``contradicts=``) surfaced here
                rather than silently dropped by an inner join.
        """
        rows = self._conn().execute(
            """
            SELECT r.from_claim_id, a.text, a.status, r.to_claim_id, b.text, b.status
            FROM kb_relationships r
            JOIN kb_claims a ON a.agent_id = r.agent_id AND a.id = r.from_claim_id
            LEFT JOIN kb_claims b ON b.agent_id = r.agent_id AND b.id = r.to_claim_id
            WHERE r.agent_id = %s AND r.kind = 'contradicts'
            ORDER BY r.created_at
            """,
            (agent_id,),
        ).fetchall()
        return [
            {
                "from_claim_id": r[0], "from_text": r[1], "from_status": r[2],
                "to_claim_id": r[3], "to_text": r[4], "to_status": r[5],
            }
            for r in rows
        ]

    def list_stale_claims(self, agent_id: str, now: float, threshold: float = 0.5) -> list[dict]:
        """List claims whose :func:`_freshness` has decayed below ``threshold``.

        Args:
            agent_id: Which agent's claims to check.
            now: Epoch seconds "now" is evaluated at.
            threshold: Freshness cutoff (0.0-1.0). Default 0.5 — one
                half-life old.

        Returns:
            list[dict]: ``{"id", "rel_path", "text", "status",
                "observed_at", "freshness"}``, stalest (lowest
                freshness) first.
        """
        rows = self._conn().execute(
            "SELECT id, rel_path, text, status, observed_at FROM kb_claims WHERE agent_id = %s",
            (agent_id,),
        ).fetchall()
        stale = []
        for r in rows:
            freshness = _freshness(r[4], now)
            if freshness < threshold:
                stale.append({
                    "id": r[0], "rel_path": r[1], "text": r[2], "status": r[3],
                    "observed_at": r[4], "freshness": freshness,
                })
        stale.sort(key=lambda c: c["freshness"])
        return stale

    def list_low_confidence_claims(self, agent_id: str, threshold: float = 0.5) -> list[dict]:
        """List claims with no confidence estimate, or one below ``threshold``.

        A ``NULL`` confidence (never rated) and a low rated confidence
        are different situations, but both mean "a human shouldn't trust
        this without a closer look" — the reason this dashboard groups
        them together rather than reporting them separately.

        Returns:
            list[dict]: ``{"id", "rel_path", "text", "status",
                "confidence"}`` (``confidence`` is ``None`` for an
                unrated claim), unrated claims first.
        """
        rows = self._conn().execute(
            """
            SELECT id, rel_path, text, status, confidence FROM kb_claims
            WHERE agent_id = %s AND (confidence IS NULL OR confidence < %s)
            ORDER BY confidence NULLS FIRST
            """,
            (agent_id, threshold),
        ).fetchall()
        return [
            {"id": r[0], "rel_path": r[1], "text": r[2], "status": r[3], "confidence": r[4]}
            for r in rows
        ]

    def list_claims_missing_evidence(self, agent_id: str) -> list[dict]:
        """List claims with zero ``kb_evidence`` rows — the provenance-gap report.

        A marker-less claim is allowed to exist (hand-authored content
        can't be retroactively forced to cite a source — see
        ``memory/knowledge.py``'s module docstring); this is where it
        surfaces instead.

        Returns:
            list[dict]: ``{"id", "rel_path", "text", "status"}``.
        """
        rows = self._conn().execute(
            """
            SELECT id, rel_path, text, status FROM kb_claims
            WHERE agent_id = %s AND id NOT IN (
                SELECT DISTINCT claim_id FROM kb_evidence WHERE agent_id = %s
            )
            ORDER BY rel_path, line_number
            """,
            (agent_id, agent_id),
        ).fetchall()
        return [{"id": r[0], "rel_path": r[1], "text": r[2], "status": r[3]} for r in rows]

    def list_claims_needing_privacy_review(self, agent_id: str) -> list[dict]:
        """List claims with no ``privacy_tier`` assigned yet.

        Returns:
            list[dict]: ``{"id", "rel_path", "text", "status"}``.
        """
        rows = self._conn().execute(
            """
            SELECT id, rel_path, text, status FROM kb_claims
            WHERE agent_id = %s AND privacy_tier = ''
            ORDER BY rel_path, line_number
            """,
            (agent_id,),
        ).fetchall()
        return [{"id": r[0], "rel_path": r[1], "text": r[2], "status": r[3]} for r in rows]

    def list_claims_needing_reevaluation(self, agent_id: str) -> list[dict]:
        """List claims left with ``status="unknown"`` and zero evidence — the "deletion coverage" report.

        Stage One Phase 7, slice F. This is exactly the signature
        ``memory/forgetting.py``'s ``forget_source()`` leaves behind on a
        claim once its last evidence citation is removed: the marker and
        its text stay in the note (never silently deleted), but its
        status flips to ``"unknown"`` so it surfaces here as something a
        human should look at again — the audit trail a human reviewing
        "did forgetting correctly leave things in a re-evaluate-me
        state, and is there anything left to act on" wants to see.

        Overlaps in principle with ``list_claims(status="unknown")``
        (open questions) and :meth:`list_claims_missing_evidence`
        (provenance gaps), but the *intersection* of both is a distinct,
        meaningful signal on its own — a claim that was never classified
        yet is not the same situation as one that lost its only
        grounding.

        Returns:
            list[dict]: ``{"id", "rel_path", "text", "status"}``.
        """
        rows = self._conn().execute(
            """
            SELECT id, rel_path, text, status FROM kb_claims
            WHERE agent_id = %s AND status = 'unknown' AND id NOT IN (
                SELECT DISTINCT claim_id FROM kb_evidence WHERE agent_id = %s
            )
            ORDER BY rel_path, line_number
            """,
            (agent_id, agent_id),
        ).fetchall()
        return [{"id": r[0], "rel_path": r[1], "text": r[2], "status": r[3]} for r in rows]

    def list_claims_citing_evidence(self, agent_id: str, source_kind: str, source_ref: str) -> list[dict]:
        """List every claim whose evidence includes ``(source_kind, source_ref)``.

        Stage One Phase 7, slice F: the lookup behind
        ``memory/forgetting.py``'s ``forget_source()`` — a purely
        read-only query (this method never mutates anything; the caller
        edits the affected pages' claim markers directly and reindexes,
        the same "files are the source of truth" discipline every other
        write in this project follows).

        Args:
            agent_id: Which agent's claims to search.
            source_kind: The evidence kind to match, e.g. ``"proposal"``,
                ``"import"``, or ``"message"``.
            source_ref: The evidence reference to match, e.g. a proposal
                id or import key, as a string.

        Returns:
            list[dict]: ``{"id", "rel_path", "text", "status"}`` for
                every matching claim, ordered by page.
        """
        rows = self._conn().execute(
            """
            SELECT c.id, c.rel_path, c.text, c.status
            FROM kb_claims c
            JOIN kb_evidence e ON e.agent_id = c.agent_id AND e.claim_id = c.id
            WHERE c.agent_id = %s AND e.source_kind = %s AND e.source_ref = %s
            ORDER BY c.rel_path, c.line_number
            """,
            (agent_id, source_kind, source_ref),
        ).fetchall()
        return [{"id": r[0], "rel_path": r[1], "text": r[2], "status": r[3]} for r in rows]

    def get_boundary(self, agent_id: str, rel_path: str) -> dict[str, str] | None:
        """Look up one file's cached action-boundary metadata (Stage One Phase 6, slice A).

        Reads the ``boundary_metadata`` cached on ``memory_files`` by the
        most recent :meth:`reindex_file`/:meth:`force_rebuild_agent` — never
        re-parses the file from disk. Called by
        ``memory/service.py``'s ``MemoryService._apply_boundaries`` once per
        unique ``rel_path`` in a result set.

        Returns:
            dict[str, str] | None: The parsed frontmatter fields (see
                ``memory/boundaries.py``'s ``parse_frontmatter``), or
                ``None`` if the file was never indexed or has no
                frontmatter block.
        """
        row = self._conn().execute(
            "SELECT boundary_metadata FROM memory_files WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        ).fetchone()
        if row is None or not row[0]:
            return None
        return json.loads(row[0])

    def remove_file(self, agent_id: str, rel_path: str) -> None:
        """Remove one file's chunks, ledger row, pin, and claims (e.g. after on-disk deletion).

        A no-op if this file was never indexed/pinned — safe to call
        unconditionally. Also clears any pin (Stage One Phase 4, slice B)
        and any claims/evidence/relationships synced from this page
        (Stage One Phase 7, slices A-B) so a deleted note can never
        linger as an orphaned pin or orphaned claims pointing at content
        that no longer exists.
        """
        conn = self._conn()
        conn.execute(
            "DELETE FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        )
        conn.execute(
            "DELETE FROM memory_files WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        )
        conn.execute(
            "DELETE FROM memory_pins WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        )
        conn.execute(
            "DELETE FROM kb_evidence WHERE agent_id = %s AND claim_id IN "
            "(SELECT id FROM kb_claims WHERE agent_id = %s AND rel_path = %s)",
            (agent_id, agent_id, rel_path),
        )
        conn.execute(
            "DELETE FROM kb_relationships WHERE agent_id = %s AND from_claim_id IN "
            "(SELECT id FROM kb_claims WHERE agent_id = %s AND rel_path = %s)",
            (agent_id, agent_id, rel_path),
        )
        conn.execute(
            "DELETE FROM kb_claims WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        )

    # ------------------------------------------------------------------
    # Proposal indexing — Stage One Phase 5, slice B
    # ------------------------------------------------------------------

    def reindex_proposal(self, agent_id: str, proposal_id: int, claim_text: str) -> int:
        """Index one unreviewed capture-job proposal as a searchable chunk.

        Makes ``memory_proposals`` rows (``session/db.py``'s ``SessionDB``)
        reachable through the same lexical/hybrid search as real memory
        files, so Phase 5 slice C's consolidation ranking can use recall
        telemetry (:func:`hash_query`, :meth:`recall_stats`) on proposals
        the same way it does on notes.

        A proposal has no file on disk, so it gets a synthetic
        ``rel_path`` (``"proposals/{proposal_id}"``) instead of a real
        one, and deliberately gets no ``memory_files`` ledger row: that
        ledger exists to reconcile indexed chunks against on-disk content
        (see :meth:`reconcile_agent`), which doesn't apply here — a
        proposal is written once, by
        :class:`~minion_assist.memory.capture_worker.CaptureWorker`, and
        never edited in place. Because it has no ledger row,
        :meth:`rebuild_agent`/:meth:`reconcile_agent` (which only ever
        look at ``memory_files`` rows) never touch it; :meth:`force_rebuild_agent`'s
        live-swap DELETE explicitly excludes ``source_kind = 'proposal'``
        rows for the same reason.

        Args:
            agent_id: The agent the proposal belongs to.
            proposal_id: ``memory_proposals.id`` (returned by
                :meth:`~minion_assist.session.db.SessionDB.complete_capture_job`).
            claim_text: The proposal's claim text.

        Returns:
            int: How many chunks were written (a proposal's claim text is
                short — almost always 1 — but :func:`chunk_markdown` is
                reused unconditionally rather than special-cased).
        """
        conn = self._conn()
        rel_path = f"proposals/{proposal_id}"
        conn.execute(
            "DELETE FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        )
        chunks = chunk_markdown(claim_text)
        for i, chunk in enumerate(chunks):
            conn.execute(
                """
                INSERT INTO memory_chunks
                    (agent_id, source_kind, rel_path, chunk_index, heading_path,
                     content, start_line, end_line, chunk_hash)
                VALUES (%s, 'proposal', %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    agent_id,
                    rel_path,
                    i,
                    " > ".join(chunk.heading_path),
                    chunk.content,
                    chunk.start_line,
                    chunk.end_line,
                    _hash_text(chunk.content),
                ),
            )
        self._maybe_embed_chunks(chunks)
        return len(chunks)

    def remove_proposal(self, agent_id: str, proposal_id: int) -> None:
        """Remove one proposal's indexed chunks (e.g. once reviewed, no longer 'pending').

        A no-op if this proposal was never indexed — safe to call
        unconditionally. Phase 5 slice D (review/apply/reject) is the
        expected caller once a proposal's status leaves ``"pending"``.
        """
        conn = self._conn()
        conn.execute(
            "DELETE FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
            (agent_id, f"proposals/{proposal_id}"),
        )

    def remove_consolidation_previews_for_proposal(self, agent_id: str, proposal_id: int) -> None:
        """Delete every drafted-but-never-applied preview for one proposal (MEM-GAP-003).

        A preview only ever records what :class:`~minion_assist.memory.consolidation.MemoryConsolidator`
        *would* write (see :meth:`record_consolidation_preview`) — once its
        source proposal no longer exists (e.g. the session it came from was
        deleted), there's nothing left to apply or reject it against, so the
        draft becomes meaningless clutter rather than a real audit record.
        Safe to call unconditionally — a no-op if none exist. Does not touch
        ``memory_topic_revisions``: that table records rollback history for
        an *already-applied* (promoted) proposal's resulting note, which is
        independent memory that survives deletion of its source session —
        see :meth:`~minion_assist.memory.service.MemoryService.forget_proposals`.
        """
        self._conn().execute(
            "DELETE FROM memory_consolidation_previews WHERE agent_id = %s AND proposal_id = %s",
            (agent_id, proposal_id),
        )

    def rebuild_agent(self, agent_id: str, indexable_files: list[tuple[str, str, str]]) -> int:
        """Rebuild one agent's entire index from a fresh file listing.

        Removes ledger/chunk rows for any file no longer present in
        ``indexable_files`` (i.e. deleted from disk since the last build),
        then reindexes every file in the listing.

        Args:
            agent_id: The agent to rebuild.
            indexable_files: ``(source_kind, rel_path, content)`` triples —
                typically :meth:`MemoryFileRepository.list_indexable_files`'s
                return value.

        Returns:
            int: Total chunk count written across all files.
        """
        conn = self._conn()
        current_paths = {rel_path for _kind, rel_path, _content in indexable_files}

        existing_rows = conn.execute(
            "SELECT rel_path FROM memory_files WHERE agent_id = %s", (agent_id,)
        ).fetchall()
        for (rel_path,) in existing_rows:
            if rel_path not in current_paths:
                self.remove_file(agent_id, rel_path)

        total = 0
        for source_kind, rel_path, content in indexable_files:
            total += self.reindex_file(agent_id, rel_path, source_kind, content)
        return total

    def reconcile_agent(self, agent_id: str, indexable_files: list[tuple[str, str, str]]) -> int:
        """Bring one agent's index up to date by content hash, touching only what changed.

        Unlike :meth:`rebuild_agent` (which unconditionally reindexes every
        file in place) or :meth:`force_rebuild_agent` (which does the same
        but crash-safely, via a shadow-table swap), this only reindexes a
        file whose content hash differs from what's already in the
        ``memory_files`` ledger, and only
        removes ledger/chunk rows for files no longer present. A file
        that's unchanged since the last reconciliation costs one hash
        comparison, not a full rechunk-and-reinsert.

        This is what startup catch-up, ``MemoryService``'s write-path sync,
        and the live filesystem watcher (Phase 3 slice B) all call — the
        same "diff by hash, heal exactly what's missing or stale" shape as
        ``session/db.py``'s ``reconcile_session``/``reconcile_all_sessions``
        for message mirrors.

        Args:
            agent_id: The agent to reconcile.
            indexable_files: ``(source_kind, rel_path, content)`` triples —
                typically :meth:`MemoryFileRepository.list_indexable_files`'s
                current return value.

        Returns:
            int: How many files were actually reindexed or removed (0 means
                the index was already fully up to date).
        """
        conn = self._conn()
        current = {
            rel_path: (source_kind, content) for source_kind, rel_path, content in indexable_files
        }

        existing_rows = conn.execute(
            "SELECT rel_path, content_hash FROM memory_files WHERE agent_id = %s", (agent_id,)
        ).fetchall()
        existing_hashes = {rel_path: content_hash for rel_path, content_hash in existing_rows}

        touched = 0
        for rel_path in existing_hashes:
            if rel_path not in current:
                self.remove_file(agent_id, rel_path)
                touched += 1

        for rel_path, (source_kind, content) in current.items():
            if existing_hashes.get(rel_path) != _hash_text(content):
                self.reindex_file(agent_id, rel_path, source_kind, content)
                touched += 1

        return touched

    def force_rebuild_agent(
        self, agent_id: str, indexable_files: list[tuple[str, str, str]]
    ) -> int:
        """Crash-safely rebuild one agent's entire index, atomically swapping it into service.

        Unlike :meth:`rebuild_agent` (which deletes and reinserts each
        file's chunks one at a time, so a crash partway through can leave
        an agent with a genuinely incomplete index — some files reindexed,
        some not), this builds the *entire* new index into a scratch area
        first and only replaces the live tables in a single atomic
        transaction:

        1. Clear this agent's rows from ``memory_chunks_shadow`` /
           ``memory_files_shadow`` (leftover scratch from a previous
           *interrupted* force-rebuild, if any — safe to discard, it was
           never live).
        2. Chunk and write every file in ``indexable_files`` into the
           shadow tables. If this raises partway through (or the process
           crashes), the live ``memory_chunks``/``memory_files`` tables
           are never touched at all — search results are completely
           unaffected by an interrupted rebuild.
        3. Validate that every file was actually processed (a simple
           count check — see the "processed" counter below) before
           proceeding to the swap.
        4. Swap: in one PostgreSQL transaction, delete this agent's live
           rows and copy the shadow rows into their place, then clear the
           shadow rows. PostgreSQL's own transactional guarantees mean
           this is atomic even if the process crashes mid-swap — either
           the whole transaction commits (search immediately sees the new
           index) or none of it does (search keeps seeing the old index,
           unchanged) - there is never a moment where search sees a
           half-old-half-new or empty result set.

        Args:
            agent_id: The agent to rebuild.
            indexable_files: ``(source_kind, rel_path, content)`` triples —
                typically :meth:`MemoryFileRepository.list_indexable_files`'s
                current return value.

        Returns:
            int: Total chunk count written across all files (post-swap).

        Note:
            Single-writer assumption: running two ``force_rebuild_agent``
            calls for the *same* agent concurrently is not safe (both would
            share the same shadow scratch rows). This is a rare, manual
            maintenance operation (``minion-assist memory reindex --force``),
            not something the running app ever triggers concurrently with
            itself, so this is an accepted, documented limitation rather
            than something guarded against with extra locking — the same
            kind of concurrency assumption ``session/db.py``'s
            ``mirror_message`` documents for its own check-then-insert.
        """
        conn = self._conn()
        conn.execute("DELETE FROM memory_chunks_shadow WHERE agent_id = %s", (agent_id,))
        conn.execute("DELETE FROM memory_files_shadow WHERE agent_id = %s", (agent_id,))

        processed = 0
        all_chunks = []
        for source_kind, rel_path, content in indexable_files:
            # Stage One Phase 6, slice A: same frontmatter stripping as
            # reindex_file() — see its comment for why line_offset is added
            # back onto every chunk's start_line/end_line.
            boundary_metadata, body, line_offset = parse_frontmatter(content)
            chunks = chunk_markdown(body)
            all_chunks.extend(chunks)
            for i, chunk in enumerate(chunks):
                conn.execute(
                    """
                    INSERT INTO memory_chunks_shadow
                        (agent_id, source_kind, rel_path, chunk_index, heading_path,
                         content, start_line, end_line, chunk_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        agent_id, source_kind, rel_path, i, " > ".join(chunk.heading_path),
                        chunk.content, chunk.start_line + line_offset, chunk.end_line + line_offset,
                        _hash_text(chunk.content),
                    ),
                )
            conn.execute(
                """
                INSERT INTO memory_files_shadow
                    (agent_id, rel_path, source_kind, content_hash, indexed_at, boundary_metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    agent_id, rel_path, source_kind, _hash_text(content), time.time(),
                    json.dumps(boundary_metadata) if boundary_metadata else "",
                ),
            )
            if source_kind == "durable":
                # Stage One Phase 7, slice A. Best-effort and outside the
                # shadow/swap transaction below, same treatment
                # _maybe_embed_chunks() already gets — claim tracking is
                # an additive layer on top of the chunk index's own
                # atomicity guarantee, not part of what that guarantee
                # protects. An interrupted force-rebuild leaves claims
                # merely stale until the next reconcile/rebuild, never
                # incorrect in a way that corrupts search.
                self._sync_claims(agent_id, rel_path, body, line_offset)
            processed += 1

        if processed != len(indexable_files):
            raise RuntimeError(
                f"force_rebuild_agent: processed {processed} of {len(indexable_files)} "
                f"files for agent {agent_id!r} — aborting swap, live index left unchanged."
            )

        with conn.transaction():
            # Proposal chunks (Stage One Phase 5, slice B) live in this same
            # table but have no memory_files ledger row and aren't part of
            # indexable_files — excluded here so a force-rebuild (a
            # files-only operation) can't wipe them out.
            conn.execute(
                "DELETE FROM memory_chunks WHERE agent_id = %s AND source_kind != 'proposal'",
                (agent_id,),
            )
            conn.execute(
                """
                INSERT INTO memory_chunks
                    (agent_id, source_kind, rel_path, chunk_index, heading_path,
                     content, start_line, end_line, chunk_hash)
                SELECT agent_id, source_kind, rel_path, chunk_index, heading_path,
                       content, start_line, end_line, chunk_hash
                FROM memory_chunks_shadow WHERE agent_id = %s
                """,
                (agent_id,),
            )
            conn.execute("DELETE FROM memory_files WHERE agent_id = %s", (agent_id,))
            conn.execute(
                """
                INSERT INTO memory_files
                    (agent_id, rel_path, source_kind, content_hash, indexed_at, boundary_metadata)
                SELECT agent_id, rel_path, source_kind, content_hash, indexed_at, boundary_metadata
                FROM memory_files_shadow WHERE agent_id = %s
                """,
                (agent_id,),
            )

        conn.execute("DELETE FROM memory_chunks_shadow WHERE agent_id = %s", (agent_id,))
        conn.execute("DELETE FROM memory_files_shadow WHERE agent_id = %s", (agent_id,))

        self._maybe_embed_chunks(all_chunks)
        return self.chunk_count(agent_id)

    # ------------------------------------------------------------------
    # Search — Stage One Phase 3, slice C
    # ------------------------------------------------------------------

    def search(
        self,
        agent_id: str,
        query: str,
        *,
        corpus: str | None = None,
        max_results: int = 20,
    ) -> list[dict]:
        """Full-text search across one agent's indexed chunks, ranked by relevance.

        Args:
            agent_id: Which agent's chunks to search.
            query: Free-text query, passed through PostgreSQL's
                ``websearch_to_tsquery`` — quoted phrases, ``AND``/``OR``,
                and ``-exclude`` all work, the same query language
                ``session/db.py``'s ``search_messages`` already uses for
                message search.
            corpus: Restrict results to one ``source_kind`` (``"durable"``,
                ``"daily"``, ``"import"``, or ``"proposal"`` — Stage One
                Phase 5, slice B), or ``None`` to search every corpus. The
                plan's fourth corpus, "sessions", is
                deliberately not offered here — that's already the
                separate ``session_search`` tool's job; duplicating it into
                this index would mean two divergent ways to search the same
                message data.
            max_results: Maximum chunks to return.

        Returns:
            list[dict]: Best matches first. Each has ``id`` (the
                ``memory_chunks`` row id — used internally by
                :meth:`hybrid_search`'s fusion, not part of
                :class:`~minion_assist.memory.models.MemoryHit`),
                ``rel_path``, ``source_kind``, ``chunk_index``,
                ``heading_path``, ``content``, ``start_line``, ``end_line``,
                and ``score`` (the ``ts_rank`` value).
        """
        conn = self._conn()
        # Proposals (unreviewed capture-job output) AND imports (unreviewed
        # quarantined notes — MEM-GAP-004) are never surfaced by a
        # corpus-agnostic search (which is what per-turn injection and
        # search_memory both do by default) unless a caller explicitly asks
        # for corpus="proposal" or corpus="import". Without this, unreviewed
        # or externally-sourced text could be injected straight into the
        # system prompt (or a search result) as if it were reviewed,
        # trusted memory — a prompt-injection path. Every lane below follows
        # this same rule.
        if corpus:
            corpus_sql = " AND source_kind = %s"
        else:
            corpus_sql = " AND source_kind NOT IN ('proposal', 'import')"
        params: list = [query, agent_id, query]
        if corpus:
            params.append(corpus)
        params.append(max_results)

        rows = conn.execute(
            f"""
            SELECT id, rel_path, source_kind, chunk_index, heading_path, content,
                   start_line, end_line,
                   ts_rank(search_vector, websearch_to_tsquery('english', %s)) AS score
            FROM memory_chunks
            WHERE agent_id = %s
              AND search_vector @@ websearch_to_tsquery('english', %s)
              {corpus_sql}
            ORDER BY score DESC
            LIMIT %s
            """,
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "rel_path": r[1], "source_kind": r[2], "chunk_index": r[3],
                "heading_path": r[4], "content": r[5], "start_line": r[6],
                "end_line": r[7], "score": float(r[8]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Hybrid retrieval — Stage One Phase 4, slice C
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        agent_id: str,
        query: str,
        *,
        corpus: str | None = None,
        max_results: int = 20,
    ) -> list[dict]:
        """Fuse path, lexical, vector, pinned, and recent lanes into one ranked result list.

        Each lane ranks candidates by a completely different signal (exact
        path match, ``ts_rank``, cosine similarity, pin status, recency), so
        no single lane's score is comparable to another's. Reciprocal-rank
        fusion (:func:`_reciprocal_rank_fusion`) sidesteps that by summing
        each candidate's ``1/(k+rank)`` contribution across every lane it
        appears in, rather than trying to average incompatible scores.

        Order of operations:

        1. Run each lane (each already ranked, best first).
        2. Fuse via RRF.
        3. Apply temporal decay to daily-note chunks (:func:`_decay_factor`).
        4. Re-rank by the decayed fused score.
        5. If an embedding provider is configured, run MMR
           (:meth:`_mmr`) to reduce near-duplicate snippets. Skipped
           without embeddings — MMR's whole purpose is catching *semantic*
           near-duplicates, which requires vectors to measure; there is no
           equivalent lexical-only substitute implemented here.
        6. Guarantee every currently pinned chunk (matching ``corpus``) is
           included, prepended ahead of the ranked results — "pinned"
           means always surfaced, not just boosted.
        7. Truncate to ``max_results``.

        Args:
            agent_id: Which agent's chunks to search.
            query: Free-text query.
            corpus: Restrict to one ``source_kind``, or ``None`` for every
                *reviewed* corpus. Applies to every lane, including pinned
                (in practice only "durable" pins exist, since pinning is
                scoped to topic notes — see ``memory/service.py``'s
                ``pin()``). ``None`` does *not* mean literally every
                chunk: ``"proposal"`` chunks (Stage One Phase 5, slice B —
                unreviewed capture-job output) are excluded unless
                ``corpus="proposal"`` is given explicitly, so a normal
                per-turn search or ``search_memory`` call never surfaces
                an unreviewed claim as if it were a reviewed note.
            max_results: Maximum chunks to return.

        Returns:
            list[dict]: Same shape as :meth:`search`'s results (``id``,
                ``rel_path``, ``source_kind``, ``chunk_index``,
                ``heading_path``, ``content``, ``start_line``, ``end_line``,
                ``score``) — ``score`` here is the fused, decayed RRF score
                (0.0 for a chunk included only because it's pinned and
                wasn't otherwise ranked by any lane).
        """
        lane_limit = max(max_results * 2, 20)  # overfetch each lane so fusion has real signal

        lexical_hits = self.search(agent_id, query, corpus=corpus, max_results=lane_limit)
        path_hits = self._path_lane(agent_id, query, corpus, lane_limit)
        vector_hits = self._vector_lane(agent_id, query, corpus, lane_limit)
        recent_hits = self._recent_lane(agent_id, corpus, lane_limit)

        scores, rows = _reciprocal_rank_fusion([path_hits, lexical_hits, vector_hits, recent_hits])
        for chunk_id, row in rows.items():
            scores[chunk_id] *= _decay_factor(row["rel_path"], row["source_kind"])

        ranked = sorted(rows.values(), key=lambda r: -scores[r["id"]])

        if self._embedding_provider is not None and self.has_vector_lane:
            final = self._mmr(ranked, scores, max_results)
        else:
            final = ranked[:max_results]

        pinned_hits = self._pinned_lane(agent_id, corpus)
        pinned_ids = {r["id"] for r in pinned_hits}
        non_pinned = [r for r in final if r["id"] not in pinned_ids]
        combined = pinned_hits + non_pinned

        results = []
        for row in combined[:max_results]:
            results.append({**row, "score": scores.get(row["id"], 0.0)})

        # Recall telemetry (Stage One Phase 5, slice A) — record every
        # result actually returned, regardless of caller (an explicit
        # search_memory tool call surfaces results just as much as
        # proactive per-turn injection does; mark_injected() is what later
        # distinguishes the subset that was actually injected). Never
        # allowed to turn a working search into a failed one.
        try:
            query_hash = hash_query(query)
            for row in results:
                self.record_recall(agent_id, row["rel_path"], query_hash)
        except Exception as exc:
            _log.debug(
                "Recording recall telemetry failed for agent %s: %s: %s",
                agent_id, type(exc).__name__, exc,
            )

        return results

    def _path_lane(self, agent_id: str, query: str, corpus: str | None, limit: int) -> list[dict]:
        """Exact/substring identifier match — catches a query naming a file directly.

        Complements the lexical lane's content-based ranking: a query like
        "project-goals" should surface a file at that path even if the
        word "project-goals" never appears in the file's *body* text.
        """
        terms = [t.lower() for t in query.split() if len(t) >= 3]
        if not terms:
            return []
        conn = self._conn()
        term_conditions = " OR ".join(["rel_path ILIKE %s"] * len(terms))
        params: list = [agent_id] + [f"%{t}%" for t in terms]
        # See search()'s comment: proposals and imports are excluded from a
        # corpus-agnostic query unless explicitly requested.
        if corpus:
            corpus_sql = " AND source_kind = %s"
            params.append(corpus)
        else:
            corpus_sql = " AND source_kind NOT IN ('proposal', 'import')"
        params.append(limit)

        rows = conn.execute(
            f"""
            SELECT id, rel_path, source_kind, chunk_index, heading_path, content,
                   start_line, end_line
            FROM memory_chunks
            WHERE agent_id = %s AND ({term_conditions}) {corpus_sql}
            ORDER BY chunk_index
            LIMIT %s
            """,
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "rel_path": r[1], "source_kind": r[2], "chunk_index": r[3],
                "heading_path": r[4], "content": r[5], "start_line": r[6], "end_line": r[7],
            }
            for r in rows
        ]

    def _vector_lane(self, agent_id: str, query: str, corpus: str | None, limit: int) -> list[dict]:
        """Cosine-similarity ranking against cached chunk embeddings.

        Returns an empty lane (never raises) if no embedding provider is
        configured, or if embedding the query itself fails — the vector
        lane is optional by design; its absence just means the other lanes
        carry the fusion.
        """
        if self._embedding_provider is None or not self.has_vector_lane:
            return []
        try:
            [query_vector] = self._embedding_provider.embed([query])
        except Exception as exc:
            _log.debug("Embedding the query failed, skipping vector lane: %s: %s",
                       type(exc).__name__, exc)
            return []

        conn = self._conn()
        # See search()'s comment: proposals and imports are excluded from a
        # corpus-agnostic query unless explicitly requested.
        if corpus:
            corpus_sql = " AND mc.source_kind = %s"
        else:
            corpus_sql = " AND mc.source_kind NOT IN ('proposal', 'import')"
        # Parameter order must match the SQL's %s occurrences top-to-bottom:
        # similarity's query_vector, the JOIN's model_identity, agent_id,
        # (corpus), ORDER BY's query_vector again, then LIMIT.
        params: list = [query_vector, self._embedding_provider.model_identity, agent_id]
        if corpus:
            params.append(corpus)
        params += [query_vector, limit]

        # %s::vector explicit casts: without them, psycopg sends a bare
        # Python list parameter as a plain float8[] array (even with
        # pgvector's adapter registered), and pgvector's <=> operator has
        # no overload for vector <=> float8[] — confirmed by a direct
        # smoke test against the dev database while building this lane.
        rows = conn.execute(
            f"""
            SELECT mc.id, mc.rel_path, mc.source_kind, mc.chunk_index, mc.heading_path,
                   mc.content, mc.start_line, mc.end_line,
                   1 - (mce.embedding <=> %s::vector) AS similarity
            FROM memory_chunks mc
            JOIN memory_chunk_embeddings mce
                ON mc.chunk_hash = mce.content_hash AND mce.model_identity = %s
            WHERE mc.agent_id = %s {corpus_sql}
            ORDER BY mce.embedding <=> %s::vector
            LIMIT %s
            """,
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "rel_path": r[1], "source_kind": r[2], "chunk_index": r[3],
                "heading_path": r[4], "content": r[5], "start_line": r[6], "end_line": r[7],
                "score": float(r[8]),
            }
            for r in rows
        ]

    def _pinned_lane(self, agent_id: str, corpus: str | None) -> list[dict]:
        """Every chunk of every currently pinned file, most recently pinned first.

        See ``search()``'s comment: proposals and imports are excluded from
        a corpus-agnostic query unless explicitly requested — applied here
        too so an unreviewed note can't reach automatic recall just because
        it happens to be pinned.
        """
        pinned_paths = self.pinned_files(agent_id)
        if not pinned_paths:
            return []
        conn = self._conn()
        corpus_sql = " AND source_kind = %s" if corpus else " AND source_kind NOT IN ('proposal', 'import')"
        params: list = [agent_id, pinned_paths]
        if corpus:
            params.append(corpus)

        rows = conn.execute(
            f"""
            SELECT id, rel_path, source_kind, chunk_index, heading_path, content,
                   start_line, end_line
            FROM memory_chunks
            WHERE agent_id = %s AND rel_path = ANY(%s) {corpus_sql}
            ORDER BY array_position(%s, rel_path), chunk_index
            """,
            [*params, pinned_paths],
        ).fetchall()
        return [
            {
                "id": r[0], "rel_path": r[1], "source_kind": r[2], "chunk_index": r[3],
                "heading_path": r[4], "content": r[5], "start_line": r[6], "end_line": r[7],
            }
            for r in rows
        ]

    def _recent_lane(self, agent_id: str, corpus: str | None, limit: int) -> list[dict]:
        """Most recently indexed files' chunks, regardless of content match.

        See ``search()``'s comment: proposals and imports are excluded from
        a corpus-agnostic query unless explicitly requested — applied here
        too so a just-captured unreviewed note can't reach automatic recall
        merely by being the most recently indexed file.
        """
        conn = self._conn()
        corpus_sql = (
            " AND mf.source_kind = %s" if corpus else " AND mf.source_kind NOT IN ('proposal', 'import')"
        )
        params: list = [agent_id]
        if corpus:
            params.append(corpus)
        params.append(limit)

        rows = conn.execute(
            f"""
            SELECT mc.id, mc.rel_path, mc.source_kind, mc.chunk_index, mc.heading_path,
                   mc.content, mc.start_line, mc.end_line
            FROM memory_chunks mc
            JOIN memory_files mf ON mc.agent_id = mf.agent_id AND mc.rel_path = mf.rel_path
            WHERE mc.agent_id = %s {corpus_sql}
            ORDER BY mf.indexed_at DESC, mc.chunk_index
            LIMIT %s
            """,
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "rel_path": r[1], "source_kind": r[2], "chunk_index": r[3],
                "heading_path": r[4], "content": r[5], "start_line": r[6], "end_line": r[7],
            }
            for r in rows
        ]

    def _mmr(self, ranked: list[dict], scores: dict[int, float], max_results: int) -> list[dict]:
        """Greedy maximal-marginal-relevance re-ranking to reduce near-duplicate snippets.

        Standard greedy MMR: repeatedly pick whichever remaining candidate
        maximizes ``score - max_similarity_to_already_selected``, so a
        highly-ranked chunk that's nearly identical to one already picked
        gets pushed down in favor of a lower-ranked but more novel one.
        Only ever called when an embedding provider is configured (see
        :meth:`hybrid_search`) — similarity here is cosine similarity
        between cached embeddings, falling back to no penalty (0.0
        similarity) for a candidate whose embedding isn't cached yet
        (e.g. it was written since the last successful embedding pass).

        Args:
            ranked: Candidates already sorted best-first by fused score.
            scores: chunk id -> fused score, from :meth:`hybrid_search`.
            max_results: How many to select.

        Returns:
            list[dict]: Up to ``max_results`` candidates, MMR-reordered.
        """
        model_identity = self._embedding_provider.model_identity
        vectors: dict[int, list[float] | None] = {}
        for row in ranked:
            content_hash = _hash_text(row["content"])
            vectors[row["id"]] = self.get_cached_embedding(content_hash, model_identity)

        selected: list[dict] = []
        selected_vectors: list[list[float]] = []
        remaining = list(ranked)

        while remaining and len(selected) < max_results:
            best_index = 0
            best_mmr_score = None
            for i, row in enumerate(remaining):
                vec = vectors.get(row["id"])
                if vec is not None and selected_vectors:
                    max_sim = max(_cosine_similarity(vec, sv) for sv in selected_vectors)
                else:
                    max_sim = 0.0
                mmr_score = 0.5 * scores[row["id"]] - 0.5 * max_sim
                if best_mmr_score is None or mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_index = i
            picked = remaining.pop(best_index)
            selected.append(picked)
            picked_vector = vectors.get(picked["id"])
            if picked_vector is not None:
                selected_vectors.append(picked_vector)

        return selected

    # ------------------------------------------------------------------
    # Pinning — Stage One Phase 4, slice B
    # ------------------------------------------------------------------
    #
    # Just the storage primitives here — the "pinned" fusion lane that
    # actually surfaces these in search results is Phase 4 slice C's job,
    # once fusion across lanes exists at all.

    def pin_file(self, agent_id: str, rel_path: str) -> None:
        """Pin a file so it's always surfaced by the pinned fusion lane, regardless of query match.

        Idempotent — pinning an already-pinned file just refreshes
        ``pinned_at`` rather than erroring or creating a duplicate row.
        """
        self._conn().execute(
            """
            INSERT INTO memory_pins (agent_id, rel_path, pinned_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (agent_id, rel_path) DO UPDATE SET pinned_at = EXCLUDED.pinned_at
            """,
            (agent_id, rel_path, time.time()),
        )

    def unpin_file(self, agent_id: str, rel_path: str) -> None:
        """Unpin a file. A no-op (not an error) if it wasn't pinned."""
        self._conn().execute(
            "DELETE FROM memory_pins WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        )

    def is_pinned(self, agent_id: str, rel_path: str) -> bool:
        """Whether a specific file is currently pinned."""
        row = self._conn().execute(
            "SELECT 1 FROM memory_pins WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        ).fetchone()
        return row is not None

    def pinned_files(self, agent_id: str) -> list[str]:
        """Every ``rel_path`` currently pinned for one agent, most recently pinned first."""
        rows = self._conn().execute(
            "SELECT rel_path FROM memory_pins WHERE agent_id = %s ORDER BY pinned_at DESC",
            (agent_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def _maybe_embed_chunks(self, chunks: list[Chunk]) -> None:
        """Best-effort: embed and cache ``chunks`` not already cached under the current model.

        Called from :meth:`reindex_file` and :meth:`force_rebuild_agent`
        right after their writes succeed — indexing itself never depends
        on this step. A no-op with no embedding provider configured. Skips
        any chunk whose content is already cached under the current
        model's identity (the common case on repeated indexing of mostly
        unchanged content), and batches everything else into one
        ``embed()`` call rather than one request per chunk.

        Never raises: a failed embedding call (provider unreachable, bad
        model name, etc.) is logged and swallowed, exactly like
        ``memory/service.py``'s ``_sync_index`` never lets an index-sync
        failure block the write that triggered it. The vector lane simply
        stays incomplete for these chunks until the next successful
        reindex retries them.
        """
        if self._embedding_provider is None or not self.has_vector_lane:
            return

        model_identity = self._embedding_provider.model_identity
        to_embed_hashes: list[str] = []
        to_embed_texts: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            content_hash = _hash_text(chunk.content)
            if content_hash in seen:
                continue  # duplicate content already queued this batch (e.g. two identical chunks)
            seen.add(content_hash)
            if self.get_cached_embedding(content_hash, model_identity) is not None:
                continue
            to_embed_hashes.append(content_hash)
            to_embed_texts.append(chunk.content)

        if not to_embed_texts:
            return

        try:
            vectors = self._embedding_provider.embed(to_embed_texts)
            for content_hash, vector in zip(to_embed_hashes, vectors):
                self.cache_embedding(content_hash, model_identity, vector)
        except Exception as exc:
            _log.debug(
                "Embedding %d chunk(s) failed (model %s): %s: %s",
                len(to_embed_texts), model_identity, type(exc).__name__, exc,
            )

    # ------------------------------------------------------------------
    # Recall telemetry — Stage One Phase 5, slice A
    # ------------------------------------------------------------------
    #
    # hybrid_search() calls record_recall() for every result it actually
    # returns (Task 1: "record recall telemetry only for results actually
    # surfaced"). mark_injected() is called separately, later in the same
    # turn, once the caller knows which of those surfaced results were
    # actually selected for prompt injection (build_prompt_section() may
    # only fit a few of them within its token budget) — see
    # memory/service.py's mark_injected() and agents/session.py's send().

    def record_recall(self, agent_id: str, rel_path: str, query_hash: str) -> None:
        """Record that ``rel_path`` was surfaced by a search, keyed by its (already-hashed) query.

        Never raises out of :meth:`hybrid_search` — a telemetry-write
        failure must not turn a working search into a failed one. Callers
        needing that guarantee wrap this themselves (see
        :meth:`hybrid_search`'s own try/except around its recording loop).
        """
        self._conn().execute(
            """
            INSERT INTO memory_recall_events (agent_id, rel_path, query_hash, surfaced_at)
            VALUES (%s, %s, %s, %s)
            """,
            (agent_id, rel_path, query_hash, time.time()),
        )

    def mark_injected(self, agent_id: str, rel_paths: list[str], query_hash: str) -> None:
        """Mark the most recent recall event for each ``rel_path`` under this query as injected.

        Correlates by ``(agent_id, rel_path, query_hash)`` rather than a
        row id passed back through the whole call chain — safe because a
        turn's own ``record_recall`` calls and its later ``mark_injected``
        call for the *same query* happen microseconds apart, serialized by
        ``AgentSession._lock``, so "most recent matching row" is
        unambiguous in practice.

        Args:
            agent_id: The agent whose recall events to update.
            rel_paths: Which surfaced files were actually injected.
            query_hash: Must match the ``query_hash`` :meth:`hybrid_search`
                recorded for this same search (see :func:`hash_query`).
        """
        conn = self._conn()
        for rel_path in rel_paths:
            conn.execute(
                """
                UPDATE memory_recall_events SET was_injected = TRUE
                WHERE id = (
                    SELECT id FROM memory_recall_events
                    WHERE agent_id = %s AND rel_path = %s AND query_hash = %s
                    ORDER BY surfaced_at DESC
                    LIMIT 1
                )
                """,
                (agent_id, rel_path, query_hash),
            )

    def recall_stats(self, agent_id: str, rel_path: str) -> dict:
        """Aggregate recall telemetry for one file — a primitive Phase 5 slice C's ranking uses.

        Returns:
            dict: ``recall_count`` (total times surfaced), ``unique_queries``
                (distinct ``query_hash`` values — Task 3's "query
                diversity"), ``injected_count`` (times actually injected),
                and ``last_recalled_at`` (``None`` if never recalled).
        """
        rows = self._conn().execute(
            "SELECT query_hash, was_injected, surfaced_at FROM memory_recall_events "
            "WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        ).fetchall()
        if not rows:
            return {
                "recall_count": 0,
                "unique_queries": 0,
                "injected_count": 0,
                "last_recalled_at": None,
            }
        return {
            "recall_count": len(rows),
            "unique_queries": len({r[0] for r in rows}),
            "injected_count": sum(1 for r in rows if r[1]),
            "last_recalled_at": max(r[2] for r in rows),
        }

    # ------------------------------------------------------------------
    # Consolidation previews — Stage One Phase 5, slice C
    # ------------------------------------------------------------------

    def record_consolidation_preview(
        self,
        agent_id: str,
        proposal_id: int,
        target_kind: str,
        target_key: str,
        based_on_content_hash: str,
        drafted_content: str,
        rationale: str,
    ) -> int:
        """Store one drafted consolidation preview — never applied, only recorded.

        Called by :class:`~minion_assist.memory.consolidation.MemoryConsolidator`'s
        ``preview()``. Every call inserts a new row rather than upserting —
        a proposal can be re-previewed after its target note changes, and
        keeping every draft lets a later slice's ``rollback``/``explain``
        commands see the full history rather than only the latest attempt.

        Args:
            agent_id: The agent this preview belongs to.
            proposal_id: Which ``memory_proposals`` row this drafts from.
            target_kind: ``"new_topic"`` or ``"revise_topic"``.
            target_key: The topic note key this would create/revise.
            based_on_content_hash: SHA256 of the target's content at the
                moment this preview was drafted (``""`` for a new topic) —
                lets a later apply step detect a human edit made since.
            drafted_content: The full proposed note content.
            rationale: One or two sentences explaining the draft.

        Returns:
            int: The new preview row's id.
        """
        row = self._conn().execute(
            """
            INSERT INTO memory_consolidation_previews
                (agent_id, proposal_id, target_kind, target_key,
                 based_on_content_hash, drafted_content, rationale, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                agent_id, proposal_id, target_kind, target_key,
                based_on_content_hash, drafted_content, rationale, time.time(),
            ),
        ).fetchone()
        return row[0]

    def list_consolidation_previews(
        self, agent_id: str, proposal_id: int | None = None
    ) -> list[dict]:
        """List consolidation previews for one agent, most recent first.

        Args:
            agent_id: Which agent's previews to list.
            proposal_id: Restrict to one proposal's previews, or ``None``
                for every preview belonging to the agent.

        Returns:
            list[dict]: ``{"id", "agent_id", "proposal_id", "target_kind",
                "target_key", "based_on_content_hash", "drafted_content",
                "rationale", "created_at"}`` dicts, newest first.
        """
        proposal_sql = " AND proposal_id = %s" if proposal_id is not None else ""
        params: list = [agent_id]
        if proposal_id is not None:
            params.append(proposal_id)
        rows = self._conn().execute(
            f"""
            SELECT id, agent_id, proposal_id, target_kind, target_key,
                   based_on_content_hash, drafted_content, rationale, created_at
            FROM memory_consolidation_previews
            WHERE agent_id = %s {proposal_sql}
            ORDER BY created_at DESC
            """,
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "agent_id": r[1], "proposal_id": r[2], "target_kind": r[3],
                "target_key": r[4], "based_on_content_hash": r[5], "drafted_content": r[6],
                "rationale": r[7], "created_at": r[8],
            }
            for r in rows
        ]

    def get_consolidation_preview(self, preview_id: int) -> dict | None:
        """Look up one preview by id.

        Stage One Phase 5, slice D: ``MemoryConsolidator.approve()``/CLI's
        ``explain``/``approve`` commands need a single preview, not a list.

        Returns:
            dict | None: Same shape as :meth:`list_consolidation_previews`'
                rows, or ``None`` if no preview with this id exists.
        """
        row = self._conn().execute(
            """
            SELECT id, agent_id, proposal_id, target_kind, target_key,
                   based_on_content_hash, drafted_content, rationale, created_at
            FROM memory_consolidation_previews
            WHERE id = %s
            """,
            (preview_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "agent_id": row[1], "proposal_id": row[2], "target_kind": row[3],
            "target_key": row[4], "based_on_content_hash": row[5], "drafted_content": row[6],
            "rationale": row[7], "created_at": row[8],
        }

    # ------------------------------------------------------------------
    # Topic revision history — Stage One Phase 5, slice D
    # ------------------------------------------------------------------

    def record_topic_revision(
        self, agent_id: str, target_key: str, proposal_id: int, prior_content: str
    ) -> int:
        """Snapshot a topic note's content right before an apply overwrites it.

        Args:
            agent_id: The agent the topic note belongs to.
            target_key: The topic note's key.
            proposal_id: Which proposal's approval triggered this apply —
                lets :meth:`~minion_assist.memory.consolidation.MemoryConsolidator.rollback`
                restore the proposal to ``"pending"`` too.
            prior_content: The note's full content immediately before the
                apply (``""`` if the topic didn't exist yet — a brand new
                topic, not a revision).

        Returns:
            int: The new revision row's id.
        """
        row = self._conn().execute(
            """
            INSERT INTO memory_topic_revisions
                (agent_id, target_key, proposal_id, prior_content, created_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (agent_id, target_key, proposal_id, prior_content, time.time()),
        ).fetchone()
        return row[0]

    def latest_topic_revision(self, agent_id: str, target_key: str) -> dict | None:
        """The most recent not-yet-rolled-back-from revision for one topic note.

        Returns:
            dict | None: ``{"id", "agent_id", "target_key", "proposal_id",
                "prior_content", "created_at"}``, or ``None`` if this topic
                has no revision history (never applied, or already fully
                rolled back).
        """
        row = self._conn().execute(
            """
            SELECT id, agent_id, target_key, proposal_id, prior_content, created_at
            FROM memory_topic_revisions
            WHERE agent_id = %s AND target_key = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (agent_id, target_key),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "agent_id": row[1], "target_key": row[2], "proposal_id": row[3],
            "prior_content": row[4], "created_at": row[5],
        }

    def delete_topic_revision(self, revision_id: int) -> None:
        """Consume (delete) one revision row — called once ``rollback()`` restores from it.

        Makes the revision table behave as a proper undo stack: rolling
        back twice in a row steps back through two prior applies, rather
        than restoring the same snapshot repeatedly.
        """
        self._conn().execute(
            "DELETE FROM memory_topic_revisions WHERE id = %s", (revision_id,)
        )

    # ------------------------------------------------------------------
    # Import review previews — Stage One Phase 7, slice E
    # ------------------------------------------------------------------

    def record_import_preview(
        self,
        agent_id: str,
        import_key: str,
        target_kind: str,
        target_key: str,
        based_on_content_hash: str,
        drafted_content: str,
        rationale: str,
    ) -> int:
        """Store one drafted import-review preview — never applied, only recorded.

        Called by :class:`~minion_assist.memory.import_review.ImportReviewer`'s
        ``preview()``. Same "insert, never upsert" shape as
        :meth:`record_consolidation_preview` — a quarantined import can be
        re-previewed after its content or a merge target changes, and
        keeping every draft preserves history for ``explain``.

        Args:
            agent_id: The agent this preview belongs to.
            import_key: Which quarantined ``memory/imports/{key}.md`` file
                this drafts from.
            target_kind: ``"new_topic"`` or ``"revise_topic"``.
            target_key: The topic note key this would create/revise.
            based_on_content_hash: SHA256 of the target's content at the
                moment this preview was drafted (``""`` for a new topic).
            drafted_content: The full proposed note content.
            rationale: One or two sentences explaining the draft.

        Returns:
            int: The new preview row's id.
        """
        row = self._conn().execute(
            """
            INSERT INTO memory_import_previews
                (agent_id, import_key, target_kind, target_key,
                 based_on_content_hash, drafted_content, rationale, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                agent_id, import_key, target_kind, target_key,
                based_on_content_hash, drafted_content, rationale, time.time(),
            ),
        ).fetchone()
        return row[0]

    def list_import_previews(self, agent_id: str, import_key: str | None = None) -> list[dict]:
        """List import-review previews for one agent, most recent first.

        Args:
            agent_id: Which agent's previews to list.
            import_key: Restrict to one import's previews, or ``None`` for
                every preview belonging to the agent.

        Returns:
            list[dict]: ``{"id", "agent_id", "import_key", "target_kind",
                "target_key", "based_on_content_hash", "drafted_content",
                "rationale", "created_at"}`` dicts, newest first.
        """
        key_sql = " AND import_key = %s" if import_key is not None else ""
        params: list = [agent_id]
        if import_key is not None:
            params.append(import_key)
        rows = self._conn().execute(
            f"""
            SELECT id, agent_id, import_key, target_kind, target_key,
                   based_on_content_hash, drafted_content, rationale, created_at
            FROM memory_import_previews
            WHERE agent_id = %s {key_sql}
            ORDER BY created_at DESC
            """,
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "agent_id": r[1], "import_key": r[2], "target_kind": r[3],
                "target_key": r[4], "based_on_content_hash": r[5], "drafted_content": r[6],
                "rationale": r[7], "created_at": r[8],
            }
            for r in rows
        ]

    def get_import_preview(self, preview_id: int) -> dict | None:
        """Look up one import-review preview by id.

        Returns:
            dict | None: Same shape as :meth:`list_import_previews`' rows,
                or ``None`` if no preview with this id exists.
        """
        row = self._conn().execute(
            """
            SELECT id, agent_id, import_key, target_kind, target_key,
                   based_on_content_hash, drafted_content, rationale, created_at
            FROM memory_import_previews
            WHERE id = %s
            """,
            (preview_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "agent_id": row[1], "import_key": row[2], "target_kind": row[3],
            "target_key": row[4], "based_on_content_hash": row[5], "drafted_content": row[6],
            "rationale": row[7], "created_at": row[8],
        }

    # ------------------------------------------------------------------
    # Embedding cache — Stage One Phase 4, slices A & C
    # ------------------------------------------------------------------
    #
    # Keyed by (content_hash, model_identity) — see the schema comment in
    # _ensure_schema() for why this is content-addressed rather than tied
    # to a specific memory_chunks row. Guarded by `has_vector_lane` rather
    # than relying on callers to check first, so a caller that forgets
    # simply gets a no-op/None instead of a SQL error from a table that
    # was never created.

    @property
    def has_vector_lane(self) -> bool:
        """Whether the embedding cache table exists (pgvector available and a model configured)."""
        return self._has_vector and bool(self._embedding_dimensions)

    def cache_embedding(
        self, content_hash: str, model_identity: str, embedding: list[float]
    ) -> None:
        """Store (or replace) the embedding for one exact chunk content, under one model.

        Args:
            content_hash: SHA256 of the chunk's content (matches
                ``memory_chunks.chunk_hash`` for the chunk(s) this text
                came from — see :func:`_hash_text`).
            model_identity: Identifies which model/endpoint produced this
                vector (:attr:`EmbeddingProvider.model_identity`) — lets a
                model change be detected rather than silently mixing
                vectors from two different embedding spaces.
            embedding: The vector itself. A no-op if no vector lane is
                configured (see :attr:`has_vector_lane`).
        """
        if not self.has_vector_lane:
            return
        self._conn().execute(
            """
            INSERT INTO memory_chunk_embeddings (content_hash, model_identity, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (content_hash, model_identity) DO UPDATE SET embedding = EXCLUDED.embedding
            """,
            (content_hash, model_identity, embedding),
        )

    def get_cached_embedding(self, content_hash: str, model_identity: str) -> list[float] | None:
        """Look up the cached embedding for this exact chunk content, under this model.

        Returns ``None`` (a cache miss, not an error) both when there is no
        vector lane configured and when this exact ``(content_hash,
        model_identity)`` pair has simply never been embedded.

        Args:
            content_hash: Must match the cached row's value exactly.
            model_identity: Must match the cached row's value exactly.

        Returns:
            list[float] | None: The cached vector, or ``None`` on a miss.
        """
        if not self.has_vector_lane:
            return None
        row = self._conn().execute(
            "SELECT embedding FROM memory_chunk_embeddings "
            "WHERE content_hash = %s AND model_identity = %s",
            (content_hash, model_identity),
        ).fetchone()
        # register_vector() (see _conn()) makes row[0] a pgvector.vector.Vector,
        # not a plain list — .to_list() is that class's conversion method.
        return row[0].to_list() if row else None

    # ------------------------------------------------------------------
    # Inspection (used by tests and later slices' status reporting)
    # ------------------------------------------------------------------

    def chunk_count(self, agent_id: str) -> int:
        """Total indexed chunk count for one agent — a primitive for tests/status."""
        row = self._conn().execute(
            "SELECT count(*) FROM memory_chunks WHERE agent_id = %s", (agent_id,)
        ).fetchone()
        return row[0] if row else 0

    def indexed_files(self, agent_id: str) -> list[str]:
        """Every ``rel_path`` currently in the reconciliation ledger for one agent."""
        rows = self._conn().execute(
            "SELECT rel_path FROM memory_files WHERE agent_id = %s ORDER BY rel_path", (agent_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def index_summary(self, agent_id: str) -> dict:
        """A snapshot of one agent's index health — the primitive behind ``memory status --deep``.

        Everything here is derived from tables a one-shot CLI process can
        actually read; it deliberately does *not* claim to know whether a
        live ``CaptureWorker``/``MemoryIndexWatcher`` thread is currently
        running in some other process — that's not observable from here.

        Returns:
            dict: ``total_chunks``, ``file_count``, ``by_corpus`` (chunk
                count per ``source_kind``), and ``last_indexed_at`` (the
                most recent ``memory_files.indexed_at`` — ``None`` if this
                agent has no indexed files at all).
        """
        conn = self._conn()
        total_chunks = self.chunk_count(agent_id)
        file_count = len(self.indexed_files(agent_id))

        by_corpus_rows = conn.execute(
            "SELECT source_kind, count(*) FROM memory_chunks "
            "WHERE agent_id = %s GROUP BY source_kind",
            (agent_id,),
        ).fetchall()
        by_corpus = {kind: count for kind, count in by_corpus_rows}

        last_indexed_row = conn.execute(
            "SELECT max(indexed_at) FROM memory_files WHERE agent_id = %s", (agent_id,)
        ).fetchone()
        last_indexed_at = last_indexed_row[0] if last_indexed_row else None

        return {
            "total_chunks": total_chunks,
            "file_count": file_count,
            "by_corpus": by_corpus,
            "last_indexed_at": last_indexed_at,
        }
