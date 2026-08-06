"""Tests for memory/extractor.py against MemoryService (Stage One Phase 1, slice 3).

Previously extractor.py was only exercised indirectly through
tests/test_agents_session.py, which mocks extract_and_save_async entirely
and never checks what actually gets written. These tests cover the real
_worker/_append path directly, in particular that extracted facts land in
the quarantined memory/imports/ location (never memory/topics/) now that
the extractor writes through MemoryService instead of LongTermMemory.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from minion_assist.memory.extractor import _MAX_ROLLING_ENTRIES, _worker, extract_facts
from minion_assist.memory.files import MemoryFileRepository
from minion_assist.memory.service import MemoryService
from minion_assist.providers.base import LLMResponse


def _provider_returning(text: str) -> Mock:
    provider = Mock()
    provider.chat = Mock(return_value=LLMResponse(text=text, finish_reason="stop"))
    return provider


def _service(tmp_path) -> MemoryService:
    return MemoryService(MemoryFileRepository(tmp_path))


# A minimal 2-message user+assistant exchange — _worker() only needs len() >= 2
# and role/content fields; the actual text is irrelevant since the provider's
# response is mocked.
_EXCHANGE = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hi"},
]


def test_worker_saves_extracted_facts_to_quarantined_import(tmp_path):
    """Extracted facts land in memory/imports/_auto_extracted.md, not topics/."""
    service = _service(tmp_path)
    provider = _provider_returning("User prefers dark mode.")

    _worker(service, provider, _EXCHANGE)

    assert (tmp_path / "memory" / "imports" / "_auto_extracted.md").exists()
    assert not (tmp_path / "memory" / "topics" / "_auto_extracted.md").exists()
    assert service.load_import("_auto_extracted") == "User prefers dark mode."
    assert service.load("_auto_extracted") is None  # not a topic note


def test_worker_does_nothing_when_provider_says_nothing(tmp_path):
    service = _service(tmp_path)
    provider = _provider_returning("NOTHING")

    _worker(service, provider, _EXCHANGE)

    assert service.load_import("_auto_extracted") is None


def test_worker_caps_facts_at_three_per_turn(tmp_path):
    service = _service(tmp_path)
    provider = _provider_returning("fact one\nfact two\nfact three\nfact four")

    _worker(service, provider, _EXCHANGE)

    saved = service.load_import("_auto_extracted")
    assert saved.count("\n") == 2  # 3 lines total -> 2 newlines
    assert "fact four" not in saved


def test_worker_appends_to_existing_rolling_note(tmp_path):
    service = _service(tmp_path)
    service.remember_import("_auto_extracted", "existing fact")
    provider = _provider_returning("new fact")

    _worker(service, provider, _EXCHANGE)

    saved = service.load_import("_auto_extracted")
    assert "existing fact" in saved
    assert "new fact" in saved


def test_worker_trims_rolling_note_to_max_entries(tmp_path):
    service = _service(tmp_path)
    existing = "\n".join(f"old fact {i}" for i in range(_MAX_ROLLING_ENTRIES))
    service.remember_import("_auto_extracted", existing)
    provider = _provider_returning("newest fact")

    _worker(service, provider, _EXCHANGE)

    lines = service.load_import("_auto_extracted").splitlines()
    assert len(lines) == _MAX_ROLLING_ENTRIES
    assert "newest fact" in lines[-1]
    assert "old fact 0" not in lines  # oldest entry was trimmed


def test_worker_swallows_provider_exceptions(tmp_path):
    """A provider failure must never raise out of _worker (fail-silently contract)."""
    service = _service(tmp_path)
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("boom"))

    _worker(service, provider, _EXCHANGE)

    assert service.load_import("_auto_extracted") is None


# ---------------------------------------------------------------------------
# WorkerHealth wiring (MEM-GAP-013)
# ---------------------------------------------------------------------------

def test_worker_without_health_configured_does_not_raise(tmp_path):
    service = _service(tmp_path)
    provider = _provider_returning("A fact.")

    _worker(service, provider, _EXCHANGE)  # health=None is the default; must not raise


def test_worker_records_a_poll_and_success(tmp_path):
    from minion_assist.worker_health import WorkerHealth

    service = _service(tmp_path)
    provider = _provider_returning("A fact.")
    health = WorkerHealth("memory_extractor:main")

    _worker(service, provider, _EXCHANGE, health)

    snap = health.snapshot()
    assert snap["last_poll_at"] is not None
    assert snap["last_success_at"] is not None
    assert snap["consecutive_failures"] == 0


def test_worker_records_failure_on_provider_exception(tmp_path):
    from minion_assist.worker_health import WorkerHealth

    service = _service(tmp_path)
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("boom"))
    health = WorkerHealth("memory_extractor:main")

    _worker(service, provider, _EXCHANGE, health)

    snap = health.snapshot()
    assert snap["consecutive_failures"] == 1
    assert "boom" in snap["last_error"]
    assert snap["last_success_at"] is None


def test_extract_and_save_async_passes_health_through_to_the_worker(tmp_path):
    """Thread-based, so poll this briefly rather than asserting instantly."""
    import time as _time

    from minion_assist.memory.extractor import extract_and_save_async
    from minion_assist.worker_health import WorkerHealth

    service = _service(tmp_path)
    provider = _provider_returning("A fact.")
    health = WorkerHealth("memory_extractor:main")

    extract_and_save_async(service, provider, _EXCHANGE, health=health)

    for _ in range(50):
        if health.snapshot()["last_success_at"] is not None:
            break
        _time.sleep(0.02)
    assert health.snapshot()["last_success_at"] is not None


# ---------------------------------------------------------------------------
# extract_facts (Stage One Phase 2, slice C — the shared primitive)
# ---------------------------------------------------------------------------
# Unlike _worker, extract_facts() does NOT catch provider exceptions — the
# durable capture worker's own retry/backoff loop needs them to propagate.

def test_extract_facts_returns_parsed_lines():
    provider = _provider_returning("fact one\nfact two")
    assert extract_facts(provider, _EXCHANGE) == ["fact one", "fact two"]


def test_extract_facts_returns_empty_list_for_nothing():
    provider = _provider_returning("NOTHING")
    assert extract_facts(provider, _EXCHANGE) == []


def test_extract_facts_caps_at_three():
    provider = _provider_returning("one\ntwo\nthree\nfour")
    assert extract_facts(provider, _EXCHANGE) == ["one", "two", "three"]


def test_extract_facts_propagates_provider_exceptions():
    """The durable capture worker relies on this to trigger its retry/backoff."""
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        extract_facts(provider, _EXCHANGE)
