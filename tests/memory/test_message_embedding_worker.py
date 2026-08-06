"""Tests for memory/message_embedding_worker.py: MessageEmbeddingWorker (MEM-GAP-006).

Same structure as tests/memory/test_capture_worker.py: exercises
_process_one() directly against a mock SessionDB and a mock EmbeddingProvider
— fast and deterministic, no threading or real database involved.
"""

from __future__ import annotations

from unittest.mock import Mock

from minion_assist.memory.message_embedding_worker import MessageEmbeddingWorker
from minion_assist.worker_health import WorkerHealth


def _job(**overrides) -> dict:
    base = {
        "id": 1,
        "agent_id": "main",
        "session_id": "sess-1",
        "message_id": 42,
        "attempts": 0,
    }
    base.update(overrides)
    return base


def _provider_returning(vector: list[float]) -> Mock:
    provider = Mock()
    provider.model_identity = "test-endpoint::test-model"
    provider.embed = Mock(return_value=[vector])
    return provider


# ---------------------------------------------------------------------------
# _process_one
# ---------------------------------------------------------------------------

def test_process_one_returns_false_when_queue_empty():
    db = Mock()
    db.claim_next_message_embedding_job = Mock(return_value=None)
    worker = MessageEmbeddingWorker(db, _provider_returning([0.1, 0.2, 0.3]))

    assert worker._process_one() is False


def test_process_one_embeds_and_completes_the_job():
    db = Mock()
    db.claim_next_message_embedding_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[
        {"id": 42, "role": "user", "content": "hello world",
         "tool_name": None, "timestamp": 1.0},
    ])
    provider = _provider_returning([0.1, 0.2, 0.3])
    worker = MessageEmbeddingWorker(db, provider)

    processed = worker._process_one()

    assert processed is True
    provider.embed.assert_called_once_with(["hello world"])
    db.complete_message_embedding_job.assert_called_once_with(
        1, 42, "test-endpoint::test-model", [0.1, 0.2, 0.3]
    )
    db.fail_message_embedding_job.assert_not_called()


def test_process_one_fetches_the_jobs_own_message_by_id():
    db = Mock()
    db.claim_next_message_embedding_job = Mock(return_value=_job(message_id=99))
    db.get_messages_in_range = Mock(return_value=[
        {"id": 99, "role": "assistant", "content": "some content",
         "tool_name": None, "timestamp": 1.0},
    ])
    worker = MessageEmbeddingWorker(db, _provider_returning([0.1, 0.2, 0.3]))

    worker._process_one()

    db.get_messages_in_range.assert_called_once_with("sess-1", 99, 99)


def test_process_one_fails_terminally_when_message_has_no_content():
    db = Mock()
    db.claim_next_message_embedding_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[
        {"id": 42, "role": "user", "content": None, "tool_name": None, "timestamp": 1.0},
    ])
    provider = _provider_returning([0.1, 0.2, 0.3])
    worker = MessageEmbeddingWorker(db, provider)

    processed = worker._process_one()

    assert processed is True
    provider.embed.assert_not_called()
    db.complete_message_embedding_job.assert_not_called()
    db.fail_message_embedding_job.assert_called_once()
    call_args = db.fail_message_embedding_job.call_args
    assert call_args.args[0] == 1  # job id
    assert call_args.kwargs["max_attempts"] == 1  # terminal, no retry


def test_process_one_fails_terminally_when_message_was_deleted():
    db = Mock()
    db.claim_next_message_embedding_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[])  # message no longer exists
    provider = _provider_returning([0.1, 0.2, 0.3])
    worker = MessageEmbeddingWorker(db, provider)

    processed = worker._process_one()

    assert processed is True
    provider.embed.assert_not_called()
    db.fail_message_embedding_job.assert_called_once()


def test_process_one_fails_job_with_backoff_on_provider_exception():
    db = Mock()
    db.claim_next_message_embedding_job = Mock(return_value=_job(attempts=0))
    db.get_messages_in_range = Mock(return_value=[
        {"id": 42, "role": "user", "content": "hi", "tool_name": None, "timestamp": 1.0},
    ])
    provider = Mock()
    provider.model_identity = "test-endpoint::test-model"
    provider.embed = Mock(side_effect=RuntimeError("provider unavailable"))
    worker = MessageEmbeddingWorker(db, provider)

    processed = worker._process_one()

    assert processed is True
    db.complete_message_embedding_job.assert_not_called()
    db.fail_message_embedding_job.assert_called_once()
    call_args = db.fail_message_embedding_job.call_args
    assert call_args.args[0] == 1  # job id
    assert "provider unavailable" in call_args.args[1]
    assert call_args.args[2] > 0  # backoff_seconds


def test_process_one_backoff_grows_with_attempts():
    db = Mock()
    db.claim_next_message_embedding_job = Mock(return_value=_job(attempts=3))
    db.get_messages_in_range = Mock(return_value=[
        {"id": 42, "role": "user", "content": "hi", "tool_name": None, "timestamp": 1.0},
    ])
    provider = Mock()
    provider.model_identity = "test-endpoint::test-model"
    provider.embed = Mock(side_effect=RuntimeError("boom"))
    worker = MessageEmbeddingWorker(db, provider)

    worker._process_one()

    backoff = db.fail_message_embedding_job.call_args.args[2]
    assert backoff == 2.0 * (2 ** 3)  # _BACKOFF_BASE_SECONDS * 2**attempts


# ---------------------------------------------------------------------------
# start/stop lifecycle
# ---------------------------------------------------------------------------

def test_start_and_stop_join_cleanly():
    db = Mock()
    db.claim_next_message_embedding_job = Mock(return_value=None)  # empty queue
    worker = MessageEmbeddingWorker(db, _provider_returning([0.1, 0.2, 0.3]))

    worker.start()
    worker.stop(timeout=2.0)

    assert not worker._thread.is_alive()


def test_stop_without_start_does_not_raise():
    db = Mock()
    worker = MessageEmbeddingWorker(db, _provider_returning([0.1, 0.2, 0.3]))
    worker.stop(timeout=1.0)  # no thread was ever started


# ---------------------------------------------------------------------------
# WorkerHealth wiring (MEM-GAP-016)
# ---------------------------------------------------------------------------

def test_process_one_without_health_configured_does_not_raise():
    db = Mock()
    db.claim_next_message_embedding_job = Mock(return_value=None)
    worker = MessageEmbeddingWorker(db, _provider_returning([0.1, 0.2, 0.3]))

    assert worker._process_one() is False  # health=None is the default; must not raise


def test_process_one_records_a_poll_even_when_queue_is_empty():
    db = Mock()
    db.claim_next_message_embedding_job = Mock(return_value=None)
    health = WorkerHealth("message_embedding_worker")
    worker = MessageEmbeddingWorker(db, _provider_returning([0.1, 0.2, 0.3]), health=health)

    worker._process_one()

    assert health.snapshot()["last_poll_at"] is not None


def test_process_one_records_success_on_a_completed_job():
    db = Mock()
    db.claim_next_message_embedding_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[
        {"id": 42, "role": "user", "content": "hi", "tool_name": None, "timestamp": 1.0},
    ])
    health = WorkerHealth("message_embedding_worker")
    worker = MessageEmbeddingWorker(db, _provider_returning([0.1, 0.2, 0.3]), health=health)

    worker._process_one()

    snap = health.snapshot()
    assert snap["last_success_at"] is not None
    assert snap["consecutive_failures"] == 0


def test_process_one_records_failure_on_provider_exception():
    db = Mock()
    db.claim_next_message_embedding_job = Mock(return_value=_job(attempts=0))
    db.get_messages_in_range = Mock(return_value=[
        {"id": 42, "role": "user", "content": "hi", "tool_name": None, "timestamp": 1.0},
    ])
    provider = Mock()
    provider.model_identity = "test-endpoint::test-model"
    provider.embed = Mock(side_effect=RuntimeError("provider unavailable"))
    health = WorkerHealth("message_embedding_worker")
    worker = MessageEmbeddingWorker(db, provider, health=health)

    worker._process_one()

    snap = health.snapshot()
    assert snap["consecutive_failures"] == 1
    assert "provider unavailable" in snap["last_error"]
    assert snap["last_success_at"] is None
