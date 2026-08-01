"""Tests for tools/commitment_response.py: RespondToCommitmentTool / DismissCommitmentTool
(Stage One Phase 6, slice C).
"""

from __future__ import annotations

from unittest.mock import Mock

from minion_assist.tools.commitment_response import (
    DismissCommitmentTool,
    RespondToCommitmentTool,
)


def _commitment(**overrides) -> dict:
    base = {"id": 1, "status": "pending", "channel": "!room:example.org"}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# RespondToCommitmentTool
# ---------------------------------------------------------------------------

def test_respond_to_commitment_schema():
    tool = RespondToCommitmentTool(Mock(), Mock())
    schema = tool.schema
    assert schema.name == "respond_to_commitment"
    assert "commitment_id" in schema.parameters["properties"]
    assert "message" in schema.parameters["properties"]


def test_respond_to_commitment_delivers_the_message_to_the_commitments_channel():
    db = Mock()
    db.get_commitment = Mock(return_value=_commitment())
    deliver_fn = Mock()
    tool = RespondToCommitmentTool(db, deliver_fn)

    tool.execute(commitment_id=1, message="How did it go?")

    deliver_fn.assert_called_once_with("How did it go?", "!room:example.org")


def test_respond_to_commitment_marks_the_commitment_sent():
    db = Mock()
    db.get_commitment = Mock(return_value=_commitment())
    tool = RespondToCommitmentTool(db, Mock())

    tool.execute(commitment_id=1, message="How did it go?")

    db.mark_commitment_sent.assert_called_once_with(1)


def test_respond_to_commitment_reports_unknown_id():
    db = Mock()
    db.get_commitment = Mock(return_value=None)
    deliver_fn = Mock()
    tool = RespondToCommitmentTool(db, deliver_fn)

    result = tool.execute(commitment_id=999, message="hi")

    assert "No commitment" in result
    deliver_fn.assert_not_called()


def test_respond_to_commitment_refuses_to_send_twice():
    db = Mock()
    db.get_commitment = Mock(return_value=_commitment(status="sent"))
    deliver_fn = Mock()
    tool = RespondToCommitmentTool(db, deliver_fn)

    result = tool.execute(commitment_id=1, message="hi")

    assert "already" in result.lower()
    deliver_fn.assert_not_called()
    db.mark_commitment_sent.assert_not_called()


def test_respond_to_commitment_refuses_an_empty_message():
    db = Mock()
    deliver_fn = Mock()
    tool = RespondToCommitmentTool(db, deliver_fn)

    result = tool.execute(commitment_id=1, message="   ")

    assert "Empty message" in result
    deliver_fn.assert_not_called()
    db.get_commitment.assert_not_called()


# ---------------------------------------------------------------------------
# DismissCommitmentTool
# ---------------------------------------------------------------------------

def test_dismiss_commitment_schema():
    tool = DismissCommitmentTool(Mock())
    schema = tool.schema
    assert schema.name == "dismiss_commitment"
    assert "commitment_id" in schema.parameters["properties"]


def test_dismiss_commitment_marks_dismissed():
    db = Mock()
    db.get_commitment = Mock(return_value=_commitment())
    tool = DismissCommitmentTool(db)

    result = tool.execute(commitment_id=1)

    db.mark_commitment_dismissed.assert_called_once_with(1)
    assert "Dismissed" in result


def test_dismiss_commitment_reports_unknown_id():
    db = Mock()
    db.get_commitment = Mock(return_value=None)
    tool = DismissCommitmentTool(db)

    result = tool.execute(commitment_id=999)

    assert "No commitment" in result
    db.mark_commitment_dismissed.assert_not_called()


def test_dismiss_commitment_refuses_to_dismiss_an_already_handled_commitment():
    db = Mock()
    db.get_commitment = Mock(return_value=_commitment(status="dismissed"))
    tool = DismissCommitmentTool(db)

    result = tool.execute(commitment_id=1)

    assert "already" in result.lower()
    db.mark_commitment_dismissed.assert_not_called()
