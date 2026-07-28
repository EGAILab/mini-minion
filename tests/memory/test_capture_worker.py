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
