"""Measure the current :class:`LongTermMemory` search against a fixed query corpus.

Why this exists
----------------
Stage One Phase 3 replaces `LongTermMemory`'s linear substring scan with a
PostgreSQL lexical index, and Phase 4 adds optional embeddings. Neither of
those phases can claim to be an *improvement* without a real number to
compare against. This module measures the system as it exists today —
recall (did the right note come back at all?) and latency — against a
deterministic, checked-in query corpus
(``tests/fixtures/memory_corpus/``), so later phases have a concrete
baseline rather than an assumption that "linear scan is slow."

This is also the same shape of measurement the plan's "offline benchmark
should compare four modes: no memory, current substring search, new lexical
retrieval, and hybrid retrieval" (see
``minion-assist-docs/improve/memory-implementation-plan.md``) will reuse —
this module is *mode 2* of that eventual benchmark, not a one-off script.

Talks to
--------
- ``memory/long_term.py`` — the store being measured.
- ``tests/fixtures/memory_corpus/`` — the deterministic notes and
  ``queries.json`` used as measurement input (see
  ``tests/memory/test_baseline.py`` for the recorded Phase 0 numbers).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .long_term import LongTermMemory


@dataclass(frozen=True)
class QueryResult:
    """Outcome of running one fixture query against a memory store."""

    query: str
    expected_keys: tuple[str, ...]
    found_keys: tuple[str, ...]
    hit: bool
    latency_seconds: float


@dataclass(frozen=True)
class BaselineReport:
    """Aggregate recall/latency numbers for one corpus run."""

    note_count: int
    query_results: tuple[QueryResult, ...]

    @property
    def hits(self) -> int:
        """How many queries found at least one of their expected keys."""
        return sum(1 for r in self.query_results if r.hit)

    @property
    def recall(self) -> float:
        """Fraction of queries that found at least one expected key (0.0-1.0)."""
        if not self.query_results:
            return 0.0
        return self.hits / len(self.query_results)

    @property
    def mean_latency_seconds(self) -> float:
        """Average per-query search() latency, in seconds."""
        if not self.query_results:
            return 0.0
        return sum(r.latency_seconds for r in self.query_results) / len(self.query_results)


def load_queries(queries_file: Path) -> list[dict]:
    """Load a ``queries.json`` fixture file.

    Args:
        queries_file: Path to a JSON file — a list of objects with at least
            ``query`` (str) and ``expected_keys`` (list[str]).

    Returns:
        list[dict]: The parsed query definitions, in file order.
    """
    return json.loads(queries_file.read_text(encoding="utf-8"))


def run_baseline(notes_dir: Path, queries_file: Path) -> BaselineReport:
    """Measure :meth:`LongTermMemory.search` against a fixture corpus.

    Args:
        notes_dir: Directory of ``.md`` notes — passed straight to
            :class:`LongTermMemory` as its ``base_dir``.
        queries_file: Path to a ``queries.json`` fixture (see
            :func:`load_queries`).

    Returns:
        BaselineReport: Per-query hit/miss and latency, plus aggregate recall
            and mean latency.
    """
    mem = LongTermMemory(notes_dir)
    queries = load_queries(queries_file)

    results: list[QueryResult] = []
    for entry in queries:
        expected = tuple(entry["expected_keys"])

        start = time.perf_counter()
        hits = mem.search(entry["query"])
        elapsed = time.perf_counter() - start

        found_keys = tuple(key for key, _content in hits)
        hit = any(key in found_keys for key in expected)
        results.append(QueryResult(entry["query"], expected, found_keys, hit, elapsed))

    return BaselineReport(note_count=len(mem.list_keys()), query_results=tuple(results))


def format_report(report: BaselineReport) -> str:
    """Render a :class:`BaselineReport` as a human-readable multi-line string."""
    lines = [
        f"Baseline (current LongTermMemory linear scan): "
        f"{report.note_count} notes, {len(report.query_results)} queries",
        f"Recall: {report.hits}/{len(report.query_results)} ({report.recall:.0%})",
        f"Mean latency: {report.mean_latency_seconds * 1000:.3f} ms",
    ]
    for r in report.query_results:
        marker = "OK" if r.hit else "MISS"
        lines.append(
            f"  [{marker}] {r.query!r} -> {r.found_keys} (expected one of {r.expected_keys})"
        )
    return "\n".join(lines)
