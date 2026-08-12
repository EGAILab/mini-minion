"""Tests for memory/image_caption_worker.py: ImageCaptionWorker (R3-GAP-002).

Same structure as tests/memory/test_message_embedding_worker.py: exercises
_process_one() directly against a mock SessionDB and a mock provider — fast
and deterministic, no threading or real database/LLM involved.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from minion_assist.memory.image_caption_worker import ImageCaptionWorker
from minion_assist.providers.base import LLMResponse
from minion_assist.worker_health import WorkerHealth


def _job(**overrides) -> dict:
    base = {
        "id": 1,
        "agent_id": "main",
        "session_id": "sess-1",
        "message_id": 42,
        "path": "/tmp/shot.png",
        "media_type": "image/png",
        "source_name": "shot.png",
        "attempts": 0,
    }
    base.update(overrides)
    return base


def _vision_agents_cfg() -> dict:
    return {"main": SimpleNamespace(input_modalities=("text", "image"))}


def _text_only_agents_cfg() -> dict:
    return {"main": SimpleNamespace(input_modalities=("text",))}


def _provider_returning(text: str) -> Mock:
    provider = Mock()
    provider.chat = Mock(return_value=LLMResponse(text=text, finish_reason="stop"))
    return provider


# ---------------------------------------------------------------------------
# _process_one — happy path
# ---------------------------------------------------------------------------

def test_process_one_returns_false_when_queue_empty():
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=None)
    worker = ImageCaptionWorker(db, _vision_agents_cfg(), lambda aid: _provider_returning("x"))

    assert worker._process_one() is False


def test_process_one_captions_and_completes_the_job():
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=_job())
    provider = _provider_returning("A whiteboard diagram of a deployment topology.")
    worker = ImageCaptionWorker(db, _vision_agents_cfg(), lambda aid: provider)

    processed = worker._process_one()

    assert processed is True
    db.complete_image_caption_job.assert_called_once_with(
        1, 42, "shot.png", "A whiteboard diagram of a deployment topology."
    )
    db.fail_image_caption_job.assert_not_called()
    db.mark_image_caption_unsupported.assert_not_called()


def test_process_one_sends_the_images_own_path_and_media_type(monkeypatch):
    db = Mock()
    db.claim_next_image_caption_job = Mock(
        return_value=_job(path="/tmp/other.jpg", media_type="image/jpeg", source_name="other.jpg")
    )
    provider = _provider_returning("A photo.")
    worker = ImageCaptionWorker(db, _vision_agents_cfg(), lambda aid: provider)

    worker._process_one()

    call = provider.chat.call_args
    content = call.kwargs["messages"][0]["content"]
    image_block = next(b for b in content if b["type"] == "image")
    assert image_block["path"] == "/tmp/other.jpg"
    assert image_block["media_type"] == "image/jpeg"
    assert image_block["source_name"] == "other.jpg"


def test_process_one_uses_the_provider_for_the_jobs_own_agent():
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=_job(agent_id="researcher"))
    provider_for_main = _provider_returning("wrong agent")
    provider_for_researcher = _provider_returning("right agent")
    agents_cfg = {
        "main": SimpleNamespace(input_modalities=("text", "image")),
        "researcher": SimpleNamespace(input_modalities=("text", "image")),
    }

    def _provider_for_agent(agent_id):
        return {"main": provider_for_main, "researcher": provider_for_researcher}[agent_id]

    worker = ImageCaptionWorker(db, agents_cfg, _provider_for_agent)
    worker._process_one()

    provider_for_researcher.chat.assert_called_once()
    provider_for_main.chat.assert_not_called()


# ---------------------------------------------------------------------------
# _process_one — unsupported provider (no image input modality)
# ---------------------------------------------------------------------------

def test_process_one_marks_unsupported_when_agent_has_no_image_modality():
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=_job())
    provider = _provider_returning("should never be called")
    worker = ImageCaptionWorker(db, _text_only_agents_cfg(), lambda aid: provider)

    processed = worker._process_one()

    assert processed is True
    provider.chat.assert_not_called()
    db.mark_image_caption_unsupported.assert_called_once()
    assert db.mark_image_caption_unsupported.call_args.args[0] == 1  # job id
    db.complete_image_caption_job.assert_not_called()
    db.fail_image_caption_job.assert_not_called()


def test_process_one_marks_unsupported_when_agent_is_unknown():
    """A job for an agent_id that no longer exists in agents_cfg (e.g.
    removed from config since the job was enqueued) is also unsupported,
    not a crash."""
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=_job(agent_id="deleted-agent"))
    worker = ImageCaptionWorker(db, _vision_agents_cfg(), lambda aid: _provider_returning("x"))

    processed = worker._process_one()

    assert processed is True


def test_process_one_marks_unsupported_when_cfg_has_no_input_modalities_attr():
    """Regression: agents_cfg's values aren't guaranteed to be a fully
    populated AgentModelConfig everywhere in this codebase — a stray
    object missing input_modalities entirely must not crash this worker's
    background thread (previously a direct cfg.input_modalities access
    raised AttributeError and silently killed the poll loop; caught by
    test_minion.py's own minimal fake agents_cfg fixture)."""
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=_job())
    agents_cfg = {"main": SimpleNamespace()}  # no input_modalities at all
    worker = ImageCaptionWorker(db, agents_cfg, lambda aid: _provider_returning("x"))

    processed = worker._process_one()

    assert processed is True
    db.mark_image_caption_unsupported.assert_called_once()
    db.mark_image_caption_unsupported.assert_called_once()


# ---------------------------------------------------------------------------
# _process_one — failure paths
# ---------------------------------------------------------------------------

def test_process_one_fails_terminally_on_an_empty_caption():
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=_job())
    provider = _provider_returning("")
    worker = ImageCaptionWorker(db, _vision_agents_cfg(), lambda aid: provider)

    processed = worker._process_one()

    assert processed is True
    db.complete_image_caption_job.assert_not_called()
    db.fail_image_caption_job.assert_called_once()
    assert db.fail_image_caption_job.call_args.kwargs["max_attempts"] == 1


def test_process_one_fails_terminally_when_the_staged_file_is_gone():
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=_job())
    provider = Mock()
    provider.chat = Mock(side_effect=FileNotFoundError("no such file"))
    worker = ImageCaptionWorker(db, _vision_agents_cfg(), lambda aid: provider)

    processed = worker._process_one()

    assert processed is True
    db.fail_image_caption_job.assert_called_once()
    assert db.fail_image_caption_job.call_args.kwargs["max_attempts"] == 1


def test_process_one_fails_job_with_backoff_on_provider_exception():
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=_job(attempts=0))
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("provider unavailable"))
    worker = ImageCaptionWorker(db, _vision_agents_cfg(), lambda aid: provider)

    processed = worker._process_one()

    assert processed is True
    db.complete_image_caption_job.assert_not_called()
    db.fail_image_caption_job.assert_called_once()
    call_args = db.fail_image_caption_job.call_args
    assert call_args.args[0] == 1  # job id
    assert "provider unavailable" in call_args.args[1]
    assert call_args.args[2] > 0  # backoff_seconds


def test_process_one_backoff_grows_with_attempts():
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=_job(attempts=3))
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("boom"))
    worker = ImageCaptionWorker(db, _vision_agents_cfg(), lambda aid: provider)

    worker._process_one()

    backoff = db.fail_image_caption_job.call_args.args[2]
    assert backoff == 2.0 * (2 ** 3)  # _BACKOFF_BASE_SECONDS * 2**attempts


# ---------------------------------------------------------------------------
# start/stop lifecycle
# ---------------------------------------------------------------------------

def test_start_and_stop_join_cleanly():
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=None)  # empty queue
    worker = ImageCaptionWorker(db, _vision_agents_cfg(), lambda aid: _provider_returning("x"))

    worker.start()
    worker.stop(timeout=2.0)

    assert not worker._thread.is_alive()


def test_stop_without_start_does_not_raise():
    db = Mock()
    worker = ImageCaptionWorker(db, _vision_agents_cfg(), lambda aid: _provider_returning("x"))
    worker.stop(timeout=1.0)  # no thread was ever started


# ---------------------------------------------------------------------------
# WorkerHealth wiring
# ---------------------------------------------------------------------------

def test_process_one_without_health_configured_does_not_raise():
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=None)
    worker = ImageCaptionWorker(db, _vision_agents_cfg(), lambda aid: _provider_returning("x"))

    assert worker._process_one() is False  # health=None is the default; must not raise


def test_process_one_records_a_poll_even_when_queue_is_empty():
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=None)
    health = WorkerHealth("image_caption_worker")
    worker = ImageCaptionWorker(
        db, _vision_agents_cfg(), lambda aid: _provider_returning("x"), health=health
    )

    worker._process_one()

    assert health.snapshot()["last_poll_at"] is not None


def test_process_one_records_success_on_a_completed_job():
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=_job())
    health = WorkerHealth("image_caption_worker")
    worker = ImageCaptionWorker(
        db, _vision_agents_cfg(), lambda aid: _provider_returning("A photo."), health=health
    )

    worker._process_one()

    snap = health.snapshot()
    assert snap["last_success_at"] is not None
    assert snap["consecutive_failures"] == 0


def test_process_one_records_failure_on_provider_exception():
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=_job(attempts=0))
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("provider unavailable"))
    health = WorkerHealth("image_caption_worker")
    worker = ImageCaptionWorker(db, _vision_agents_cfg(), lambda aid: provider, health=health)

    worker._process_one()

    snap = health.snapshot()
    assert snap["consecutive_failures"] == 1
    assert "provider unavailable" in snap["last_error"]
    assert snap["last_success_at"] is None


def test_process_one_does_not_record_failure_when_unsupported():
    """Unsupported is a config fact, not a runtime failure — it shouldn't
    move WorkerHealth's consecutive_failures counter, which is meant to
    flag "something is actually broken," not "this agent has no vision model."
    """
    db = Mock()
    db.claim_next_image_caption_job = Mock(return_value=_job())
    health = WorkerHealth("image_caption_worker")
    worker = ImageCaptionWorker(
        db, _text_only_agents_cfg(), lambda aid: _provider_returning("x"), health=health
    )

    worker._process_one()

    assert health.snapshot()["consecutive_failures"] == 0
