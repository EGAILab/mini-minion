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

from minion_assist.memory.extractor import _MAX_ROLLING_ENTRIES, _worker
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
