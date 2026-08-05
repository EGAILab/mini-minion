"""Tests for memory/capture_worker.py: CaptureWorker (Stage One Phase 2, slice C).

Most tests exercise _process_one() directly against a mock SessionDB — fast
and deterministic, no threading involved. A couple of lifecycle tests cover
the real start()/stop() thread, relying on threading.Event.wait() returning
immediately once stop() sets the event (not after the full poll interval),
so they complete quickly regardless of the module's 5-second poll constant.
"""

from __future__ import annotations

from unittest.mock import Mock

from minion_assist.memory.capture_worker import CaptureWorker
from minion_assist.providers.base import LLMResponse
from minion_assist.worker_health import WorkerHealth


def _job(**overrides) -> dict:
    base = {
        "id": 1,
        "agent_id": "main",
        "session_id": "sess-1",
        "source_from_message_id": 10,
        "source_to_message_id": 12,
        "attempts": 0,
    }
    base.update(overrides)
    return base


def _provider_returning(text: str) -> Mock:
    provider = Mock()
    provider.chat = Mock(return_value=LLMResponse(text=text, finish_reason="stop"))
    return provider


# ---------------------------------------------------------------------------
# _process_one
# ---------------------------------------------------------------------------

def test_process_one_returns_false_when_queue_empty():
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=None)
    worker = CaptureWorker(db, provider_for_agent=lambda agent_id: _provider_returning("NOTHING"))

    assert worker._process_one() is False


def test_process_one_completes_job_with_extracted_facts():
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[
        {"id": 10, "role": "user", "content": "I like tea", "tool_name": None, "timestamp": 1.0},
        {"id": 11, "role": "assistant", "content": "Got it", "tool_name": None, "timestamp": 2.0},
    ])
    provider = _provider_returning("User prefers tea over coffee.")
    worker = CaptureWorker(db, provider_for_agent=lambda agent_id: provider)

    processed = worker._process_one()

    assert processed is True
    db.complete_capture_job.assert_called_once_with(1, ["User prefers tea over coffee."])
    db.fail_capture_job.assert_not_called()


def test_process_one_completes_with_empty_list_when_nothing_extracted():
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[
        {"id": 10, "role": "user", "content": "hi", "tool_name": None, "timestamp": 1.0},
    ])
    worker = CaptureWorker(db, provider_for_agent=lambda agent_id: _provider_returning("NOTHING"))

    worker._process_one()

    db.complete_capture_job.assert_called_once_with(1, [])


def test_process_one_filters_out_tool_and_empty_messages():
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[
        {"id": 10, "role": "user", "content": "question", "tool_name": None, "timestamp": 1.0},
        {"id": 11, "role": "tool", "content": "raw tool output", "tool_name": "read",
         "timestamp": 1.5},
        {"id": 12, "role": "assistant", "content": "", "tool_name": None, "timestamp": 2.0},
    ])
    provider = _provider_returning("NOTHING")
    worker = CaptureWorker(db, provider_for_agent=lambda agent_id: provider)

    worker._process_one()

    sent_messages = provider.chat.call_args.kwargs["messages"]
    assert sent_messages == [{"role": "user", "content": "question"}]


def test_process_one_uses_provider_for_the_jobs_agent():
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=_job(agent_id="researcher"))
    db.get_messages_in_range = Mock(return_value=[])
    seen_agent_ids = []

    def _provider_for_agent(agent_id):
        seen_agent_ids.append(agent_id)
        return _provider_returning("NOTHING")

    worker = CaptureWorker(db, provider_for_agent=_provider_for_agent)
    worker._process_one()

    assert seen_agent_ids == ["researcher"]


def test_process_one_fails_job_on_provider_exception():
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=_job(attempts=0))
    db.get_messages_in_range = Mock(return_value=[
        {"id": 10, "role": "user", "content": "hi", "tool_name": None, "timestamp": 1.0},
    ])
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("provider unavailable"))
    worker = CaptureWorker(db, provider_for_agent=lambda agent_id: provider)

    processed = worker._process_one()

    assert processed is True
    db.complete_capture_job.assert_not_called()
    db.fail_capture_job.assert_called_once()
    call_args = db.fail_capture_job.call_args
    assert call_args.args[0] == 1  # job id
    assert "provider unavailable" in call_args.args[1]
    assert call_args.args[2] > 0  # backoff_seconds


def test_process_one_backoff_grows_with_attempts():
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=_job(attempts=3))
    db.get_messages_in_range = Mock(return_value=[])
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("boom"))
    worker = CaptureWorker(db, provider_for_agent=lambda agent_id: provider)

    worker._process_one()

    backoff = db.fail_capture_job.call_args.args[2]
    assert backoff == 2.0 * (2 ** 3)  # _BACKOFF_BASE_SECONDS * 2**attempts


# ---------------------------------------------------------------------------
# index_proposal wiring (Stage One Phase 5, slice B)
# ---------------------------------------------------------------------------

def test_process_one_indexes_each_new_proposal():
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[
        {"id": 10, "role": "user", "content": "I like tea", "tool_name": None, "timestamp": 1.0},
    ])
    db.complete_capture_job = Mock(return_value=[
        {"id": 501, "agent_id": "main", "claim_text": "User prefers tea over coffee."},
    ])
    indexed = []
    worker = CaptureWorker(
        db,
        provider_for_agent=lambda agent_id: _provider_returning("User prefers tea over coffee."),
        index_proposal=lambda agent_id, proposal_id, claim_text: indexed.append(
            (agent_id, proposal_id, claim_text)
        ),
    )

    worker._process_one()

    assert indexed == [("main", 501, "User prefers tea over coffee.")]


def test_process_one_indexes_no_proposals_when_none_were_extracted():
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[])
    db.complete_capture_job = Mock(return_value=[])
    indexed = []
    worker = CaptureWorker(
        db,
        provider_for_agent=lambda agent_id: _provider_returning("NOTHING"),
        index_proposal=lambda agent_id, proposal_id, claim_text: indexed.append(proposal_id),
    )

    worker._process_one()

    assert indexed == []


def test_process_one_does_not_index_when_index_proposal_is_not_configured():
    # Default (index_proposal=None) — no lexical index configured. Must not
    # raise even though complete_capture_job returns real proposal rows.
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[])
    db.complete_capture_job = Mock(return_value=[
        {"id": 501, "agent_id": "main", "claim_text": "User prefers tea over coffee."},
    ])
    worker = CaptureWorker(db, provider_for_agent=lambda agent_id: _provider_returning("NOTHING"))

    processed = worker._process_one()

    assert processed is True  # must not raise


def test_process_one_treats_the_job_as_successful_even_if_indexing_fails():
    # Best-effort: an indexing failure must not turn an already-successful
    # capture job into a failed one — the proposal is already safely
    # recorded by complete_capture_job before indexing is attempted.
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[])
    db.complete_capture_job = Mock(return_value=[
        {"id": 501, "agent_id": "main", "claim_text": "User prefers tea over coffee."},
    ])

    def _broken_index(agent_id, proposal_id, claim_text):
        raise RuntimeError("index unavailable")

    worker = CaptureWorker(
        db,
        provider_for_agent=lambda agent_id: _provider_returning("NOTHING"),
        index_proposal=_broken_index,
    )

    processed = worker._process_one()

    assert processed is True
    db.fail_capture_job.assert_not_called()


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------

def test_start_and_stop_join_cleanly():
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=None)  # empty queue
    worker = CaptureWorker(db, provider_for_agent=lambda agent_id: _provider_returning("NOTHING"))

    worker.start()
    worker.stop(timeout=2.0)

    assert not worker._thread.is_alive()


def test_stop_without_start_does_not_raise():
    db = Mock()
    worker = CaptureWorker(db, provider_for_agent=lambda agent_id: _provider_returning("NOTHING"))
    worker.stop(timeout=1.0)  # no thread was ever started


# ---------------------------------------------------------------------------
# WorkerHealth wiring (MEM-GAP-016)
# ---------------------------------------------------------------------------

def test_process_one_without_health_configured_does_not_raise():
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=None)
    worker = CaptureWorker(db, provider_for_agent=lambda agent_id: _provider_returning("NOTHING"))

    assert worker._process_one() is False  # health=None is the default; must not raise


def test_process_one_records_a_poll_even_when_queue_is_empty():
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=None)
    health = WorkerHealth("capture_worker")
    worker = CaptureWorker(
        db, provider_for_agent=lambda agent_id: _provider_returning("NOTHING"), health=health
    )

    worker._process_one()

    assert health.snapshot()["last_poll_at"] is not None


def test_process_one_records_success_on_a_completed_job():
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=_job())
    db.get_messages_in_range = Mock(return_value=[
        {"id": 10, "role": "user", "content": "hi", "tool_name": None, "timestamp": 1.0},
    ])
    db.complete_capture_job = Mock(return_value=[])
    health = WorkerHealth("capture_worker")
    worker = CaptureWorker(
        db, provider_for_agent=lambda agent_id: _provider_returning("NOTHING"), health=health
    )

    worker._process_one()

    snap = health.snapshot()
    assert snap["last_success_at"] is not None
    assert snap["consecutive_failures"] == 0


def test_process_one_records_failure_on_provider_exception():
    db = Mock()
    db.claim_next_capture_job = Mock(return_value=_job(attempts=0))
    db.get_messages_in_range = Mock(return_value=[
        {"id": 10, "role": "user", "content": "hi", "tool_name": None, "timestamp": 1.0},
    ])
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("provider unavailable"))
    health = WorkerHealth("capture_worker")
    worker = CaptureWorker(db, provider_for_agent=lambda agent_id: provider, health=health)

    worker._process_one()

    snap = health.snapshot()
    assert snap["consecutive_failures"] == 1
    assert "provider unavailable" in snap["last_error"]
    assert snap["last_success_at"] is None
