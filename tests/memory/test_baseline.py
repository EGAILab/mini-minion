"""Tests for memory/baseline.py, run against the checked-in fixture corpus.

The fixture corpus at tests/fixtures/memory_corpus/ is deterministic and
checked in, so these tests measure the *real* current LongTermMemory search
behavior — not a synthetic tmp_path stand-in. This is Stage One Phase 0's
"record baseline capture/retrieval/latency metrics" task: the recorded
numbers from this run are written up in
tests/fixtures/memory_corpus/BASELINE.md for Phase 3/4 to compare against.
"""

from __future__ import annotations

import json
from pathlib import Path

from minion_assist.memory.baseline import (
    format_report,
    load_queries,
    run_baseline,
)

_CORPUS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "memory_corpus"
_NOTES_DIR = _CORPUS_DIR / "notes"
_QUERIES_FILE = _CORPUS_DIR / "queries.json"

_REQUIRED_CATEGORIES = {
    "preferences",
    "changed_preferences",
    "names",
    "code_identifiers",
    "temporary_constraints",
    "untrusted_imports",
    "session_scoped_facts",
}


# ---------------------------------------------------------------------------
# Fixture corpus integrity (Phase 0 task: deterministic fixture corpus)
# ---------------------------------------------------------------------------

def test_fixture_corpus_covers_every_required_category():
    """queries.json exercises every category the Phase 0 plan requires."""
    queries = load_queries(_QUERIES_FILE)
    categories = {q["category"] for q in queries}
    assert _REQUIRED_CATEGORIES.issubset(categories)


def test_fixture_corpus_queries_reference_existing_notes():
    """Every expected_keys entry in queries.json must be a real note file."""
    queries = load_queries(_QUERIES_FILE)
    note_keys = {p.stem for p in _NOTES_DIR.glob("*.md")}
    for q in queries:
        for key in q["expected_keys"]:
            assert key in note_keys, f"queries.json references missing note {key!r}"


def test_fixture_notes_are_non_empty():
    notes = list(_NOTES_DIR.glob("*.md"))
    assert notes, "fixture corpus must contain at least one note"
    for note in notes:
        assert note.read_text(encoding="utf-8").strip(), f"{note} is empty"


def test_queries_json_is_valid_json():
    json.loads(_QUERIES_FILE.read_text(encoding="utf-8"))  # raises on malformed JSON


# ---------------------------------------------------------------------------
# run_baseline / format_report
# ---------------------------------------------------------------------------

def test_run_baseline_measures_every_query():
    report = run_baseline(_NOTES_DIR, _QUERIES_FILE)
    queries = load_queries(_QUERIES_FILE)
    assert len(report.query_results) == len(queries)


def test_run_baseline_reports_positive_latency():
    report = run_baseline(_NOTES_DIR, _QUERIES_FILE)
    for result in report.query_results:
        assert result.latency_seconds >= 0.0


def test_run_baseline_recall_matches_recorded_measurement():
    """Pins the *current* system's real, measured recall (7/8 = 87.5%) on this
    corpus — see tests/fixtures/memory_corpus/README.md's "Recorded baseline"
    section. This is a baseline record, not a target: if it changes, update
    the recorded numbers there (and in this test) rather than silently
    tolerating drift.
    """
    report = run_baseline(_NOTES_DIR, _QUERIES_FILE)
    assert report.recall == 0.875, format_report(report)


def test_run_baseline_known_gap_apostrophe_breaks_exact_match():
    """Documents a specific, real recall miss: search() splits the query on
    whitespace without stripping punctuation, so the query term "dog's"
    never matches the stored word "dog". This is exactly the kind of gap
    Stage One Phase 3's proper lexical index (tokenized, not substring-based)
    is expected to close — see tests/fixtures/memory_corpus/README.md.
    """
    report = run_baseline(_NOTES_DIR, _QUERIES_FILE)
    [dog_result] = [r for r in report.query_results if "dog" in r.query.lower()]
    assert dog_result.hit is False
    assert dog_result.found_keys == ()


def test_format_report_includes_recall_and_latency():
    report = run_baseline(_NOTES_DIR, _QUERIES_FILE)
    text = format_report(report)
    assert "Recall:" in text
    assert "Mean latency:" in text
    assert str(report.note_count) in text


def test_run_baseline_note_count_matches_corpus_size():
    report = run_baseline(_NOTES_DIR, _QUERIES_FILE)
    assert report.note_count == len(list(_NOTES_DIR.glob("*.md")))
