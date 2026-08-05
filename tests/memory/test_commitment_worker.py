"""Tests for memory/commitment_worker.py: CommitmentWorker (Stage One Phase 6, slice B).

Mirrors tests/memory/test_capture_worker.py's structure and testing
approach — most tests exercise _process_one() directly against a mock
SessionDB, no threading involved.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

from minion_assist.memory.commitment_worker import CommitmentWorker
from minion_assist.providers.base import LLMResponse
from minion_assist.worker_health import WorkerHealth


def _job(**overrides) -> dict:
    base = {
        "id": 1,
        "agent_id": "main",
        "session_id": "sess-1",
        "channel": "cli",
        "source_from_message_id": 10,
        "source_to_message_id": 12,
        "attempts": 0,
    }
    base.update(overrides)
    return base


def _provider_returning(candidates: list[dict]) -> Mock:
    import json

    provider = Mock()
    provider.chat = Mock(
        return_value=LLMResponse(
            text=json.dumps({"candidates": candidates}), finish_reason="stop"
        )
    )
    return provider


# ---------------------------------------------------------------------------
# _process_one
# ---------------------------------------------------------------------------

def test_process_one_returns_false_when_queue_empty():
    db = Mock()
    db.claim_next_commitment_job = Mock(return_value=None)
    worker = CommitmentWorker(db, provider_for_agent=lambda agent_id: _provider_returning([]))

    assert worker._process_one() is False


def test_process_one_completes_the_job_with_extracted_candidates():
    db = Mock()
    db.claim_next_commitment_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[
        {"id": 10, "role": "user", "content": "I have an interview tomorrow.",
         "tool_name": None, "timestamp": 1.0},
        {"id": 11, "role": "assistant", "content": "Good luck!",
         "tool_name": None, "timestamp": 2.0},
    ])
    db.list_pending_commitments_for_scope = Mock(return_value=[])
    provider = _provider_returning([])
    worker = CommitmentWorker(db, provider_for_agent=lambda agent_id: provider)

    processed = worker._process_one()

    assert processed is True
    db.complete_commitment_job.assert_called_once()
    assert db.complete_commitment_job.call_args.args[0] == 1
    db.fail_commitment_job.assert_not_called()


def test_process_one_uses_the_last_user_and_assistant_message_in_range():
    db = Mock()
    db.claim_next_commitment_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[
        {"id": 10, "role": "user", "content": "first user msg", "tool_name": None, "timestamp": 1.0},
        {"id": 11, "role": "assistant", "content": "first reply", "tool_name": None, "timestamp": 1.5},
        {"id": 12, "role": "user", "content": "second user msg", "tool_name": None, "timestamp": 2.0},
        {"id": 13, "role": "assistant", "content": "second reply", "tool_name": None, "timestamp": 2.5},
    ])
    db.list_pending_commitments_for_scope = Mock(return_value=[])
    provider = _provider_returning([])
    worker = CommitmentWorker(db, provider_for_agent=lambda agent_id: provider)

    worker._process_one()

    sent_content = provider.chat.call_args.kwargs["messages"][0]["content"]
    assert "second user msg" in sent_content
    assert "second reply" in sent_content
    assert "first user msg" not in sent_content


def test_process_one_passes_existing_pending_commitments_for_the_scope():
    db = Mock()
    db.claim_next_commitment_job = Mock(return_value=_job(agent_id="main", channel="!room:example.org"))
    db.get_messages_in_range = Mock(return_value=[])
    db.list_pending_commitments_for_scope = Mock(return_value=[
        {"kind": "open_loop", "reason": "Already tracking something.", "dedupe_key": "x",
         "due_earliest": 1.0, "due_latest": 2.0},
    ])
    provider = _provider_returning([])
    worker = CommitmentWorker(db, provider_for_agent=lambda agent_id: provider)

    worker._process_one()

    db.list_pending_commitments_for_scope.assert_called_once_with("main", "!room:example.org")
    sent_content = provider.chat.call_args.kwargs["messages"][0]["content"]
    assert "Already tracking something." in sent_content


def test_process_one_uses_provider_for_the_jobs_agent():
    db = Mock()
    db.claim_next_commitment_job = Mock(return_value=_job(agent_id="researcher"))
    db.get_messages_in_range = Mock(return_value=[])
    db.list_pending_commitments_for_scope = Mock(return_value=[])
    seen_agent_ids = []

    def _provider_for_agent(agent_id):
        seen_agent_ids.append(agent_id)
        return _provider_returning([])

    worker = CommitmentWorker(db, provider_for_agent=_provider_for_agent)
    worker._process_one()

    assert seen_agent_ids == ["researcher"]


def test_process_one_fails_the_job_on_provider_exception():
    db = Mock()
    db.claim_next_commitment_job = Mock(return_value=_job(attempts=0))
    db.get_messages_in_range = Mock(return_value=[])
    db.list_pending_commitments_for_scope = Mock(return_value=[])
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("provider unavailable"))
    worker = CommitmentWorker(db, provider_for_agent=lambda agent_id: provider)

    processed = worker._process_one()

    assert processed is True
    db.complete_commitment_job.assert_not_called()
    db.fail_commitment_job.assert_called_once()
    call_args = db.fail_commitment_job.call_args
    assert call_args.args[0] == 1  # job id
    assert "provider unavailable" in call_args.args[1]
    assert call_args.args[2] > 0  # backoff_seconds


def test_process_one_backoff_grows_with_attempts():
    db = Mock()
    db.claim_next_commitment_job = Mock(return_value=_job(attempts=3))
    db.get_messages_in_range = Mock(return_value=[])
    db.list_pending_commitments_for_scope = Mock(return_value=[])
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("boom"))
    worker = CommitmentWorker(db, provider_for_agent=lambda agent_id: provider)

    worker._process_one()

    backoff = db.fail_commitment_job.call_args.args[2]
    assert backoff == 2.0 * (2 ** 3)  # _BACKOFF_BASE_SECONDS * 2**attempts


def test_process_one_passes_min_due_seconds_through_to_extraction():
    db = Mock()
    db.claim_next_commitment_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[])
    db.list_pending_commitments_for_scope = Mock(return_value=[])
    provider = _provider_returning([])
    worker = CommitmentWorker(db, provider_for_agent=lambda agent_id: provider, min_due_seconds=999.0)

    with patch("minion_assist.memory.commitment_worker.extract_commitments") as mock_extract:
        mock_extract.return_value = []
        worker._process_one()

    assert mock_extract.call_args.args[-1] == 999.0


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------

def test_start_and_stop_join_cleanly():
    db = Mock()
    db.claim_next_commitment_job = Mock(return_value=None)  # empty queue
    worker = CommitmentWorker(db, provider_for_agent=lambda agent_id: _provider_returning([]))

    worker.start()
    worker.stop(timeout=2.0)

    assert not worker._thread.is_alive()


def test_stop_without_start_does_not_raise():
    db = Mock()
    worker = CommitmentWorker(db, provider_for_agent=lambda agent_id: _provider_returning([]))
    worker.stop(timeout=1.0)  # no thread was ever started


# ---------------------------------------------------------------------------
# WorkerHealth wiring (MEM-GAP-016)
# ---------------------------------------------------------------------------

def test_process_one_records_a_poll_even_when_queue_is_empty():
    db = Mock()
    db.claim_next_commitment_job = Mock(return_value=None)
    health = WorkerHealth("commitment_worker")
    worker = CommitmentWorker(
        db, provider_for_agent=lambda agent_id: _provider_returning([]), health=health
    )

    worker._process_one()

    assert health.snapshot()["last_poll_at"] is not None


def test_process_one_records_success_on_a_completed_job():
    db = Mock()
    db.claim_next_commitment_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[])
    db.list_pending_commitments_for_scope = Mock(return_value=[])
    health = WorkerHealth("commitment_worker")
    worker = CommitmentWorker(
        db, provider_for_agent=lambda agent_id: _provider_returning([]), health=health
    )

    worker._process_one()

    snap = health.snapshot()
    assert snap["last_success_at"] is not None
    assert snap["consecutive_failures"] == 0


def test_process_one_records_failure_on_provider_exception():
    db = Mock()
    db.claim_next_commitment_job = Mock(return_value=_job(attempts=0))
    db.get_messages_in_range = Mock(return_value=[])
    db.list_pending_commitments_for_scope = Mock(return_value=[])
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("provider unavailable"))
    health = WorkerHealth("commitment_worker")
    worker = CommitmentWorker(db, provider_for_agent=lambda agent_id: provider, health=health)

    worker._process_one()

    snap = health.snapshot()
    assert snap["consecutive_failures"] == 1
    assert "provider unavailable" in snap["last_error"]
