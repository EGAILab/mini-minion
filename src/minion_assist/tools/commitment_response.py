"""``RespondToCommitmentTool`` / ``DismissCommitmentTool`` — heartbeat-turn-only
tools for handling due commitments (Stage One Phase 6, slice C).

Injected as extra_tools for a heartbeat turn only (mirrors
``heartbeat_respond.py``'s ``HeartbeatRespondTool`` — never permanently
registered in the global ``ToolRegistry``), and only when
``HeartbeatScheduler`` actually has due commitments to show the model that
turn (see ``memory/commitments.py``'s ``format_due_commitments_block``).

Task 5 of the plan: "the agent may send one natural check-in or dismiss
it." These two tools are that choice, made explicit as two separate
actions rather than one tool with a mode flag — a model is less likely to
send an unwanted check-in by accident when "send" and "dismiss" are
genuinely different tool calls, not one call with an easy-to-miss
parameter.

"Commitment delivery cannot invoke tools" (acceptance criterion)
---------------------------------------------------------------------
``RespondToCommitmentTool.execute()`` delivers ``message`` as literal text
via the injected ``deliver_fn`` callable (``HeartbeatScheduler``'s
``_deliver_to_channel`` — a direct Matrix ``send_text`` call, or a
terminal print) — never as a new prompt fed back into an agent turn. There
is no code path here that could cause the delivered message itself to
trigger a further tool call; delivery is a dumb text-send, structurally
incapable of doing anything but sending text.

Talks to
--------
- ``session/db.py`` — :meth:`~minion_assist.session.db.SessionDB.get_commitment`,
  :meth:`~minion_assist.session.db.SessionDB.mark_commitment_sent`,
  :meth:`~minion_assist.session.db.SessionDB.mark_commitment_dismissed`.
- ``heartbeat.py`` — :class:`~minion_assist.heartbeat.HeartbeatScheduler`
  constructs and injects both tools, and supplies
  :meth:`~minion_assist.heartbeat.HeartbeatScheduler._deliver_to_channel`
  as ``RespondToCommitmentTool``'s ``deliver_fn``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Tool, ToolSchema

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..session.db import SessionDB


class RespondToCommitmentTool(Tool):
    """Send a natural check-in for one due commitment, delivered into its own channel.

    Args:
        db: The ``SessionDB`` commitments live in.
        deliver_fn: ``(message, channel) -> None`` — actually sends
            ``message`` into ``channel`` (a Matrix room id, or prints to
            the terminal for the ``"cli"`` sentinel). Injected rather than
            hard-coded so this tool never needs to know about Matrix or
            asyncio directly.
    """

    def __init__(self, db: SessionDB, deliver_fn: Callable[[str, str], None]) -> None:
        self._db = db
        self._deliver_fn = deliver_fn

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM."""
        return ToolSchema(
            name="respond_to_commitment",
            description=(
                "Send a short, natural check-in message for one due commitment "
                "(from the [Due commitments] block), delivered into the exact "
                "channel it came from. Use at most once per commitment_id — if "
                "it's no longer relevant, call dismiss_commitment instead."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "commitment_id": {
                        "type": "integer",
                        "description": "The commitment's id, from the [Due commitments] block.",
                    },
                    "message": {
                        "type": "string",
                        "description": "The natural check-in message to send.",
                    },
                },
                "required": ["commitment_id", "message"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Deliver the check-in and mark the commitment ``"sent"``.

        Args:
            commitment_id (int): Which commitment to respond to.
            message (str): The check-in text to deliver.

        Returns:
            str: Confirmation, or an explanation if the commitment
                doesn't exist or was already handled (never sends twice).
        """
        commitment_id = int(kwargs["commitment_id"])  # type: ignore[arg-type]
        message = str(kwargs.get("message", "")).strip()
        if not message:
            return "[respond_to_commitment] Empty message — nothing sent."

        commitment = self._db.get_commitment(commitment_id)
        if commitment is None:
            return f"[respond_to_commitment] No commitment with id {commitment_id}."
        if commitment["status"] != "pending":
            return (
                f"[respond_to_commitment] Commitment {commitment_id} is already "
                f"{commitment['status']!r} — not sending again."
            )

        self._deliver_fn(message, commitment["channel"])
        self._db.mark_commitment_sent(commitment_id)
        return f"[respond_to_commitment] Sent check-in for commitment {commitment_id}."


class DismissCommitmentTool(Tool):
    """Dismiss one due commitment without sending anything.

    Args:
        db: The ``SessionDB`` commitments live in.
    """

    def __init__(self, db: SessionDB) -> None:
        self._db = db

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM."""
        return ToolSchema(
            name="dismiss_commitment",
            description=(
                "Dismiss one due commitment (from the [Due commitments] block) "
                "without sending anything — use when it's no longer relevant or "
                "already resolved."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "commitment_id": {
                        "type": "integer",
                        "description": "The commitment's id, from the [Due commitments] block.",
                    },
                },
                "required": ["commitment_id"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Mark the commitment ``"dismissed"``.

        Args:
            commitment_id (int): Which commitment to dismiss.

        Returns:
            str: Confirmation, or an explanation if the commitment
                doesn't exist or was already handled.
        """
        commitment_id = int(kwargs["commitment_id"])  # type: ignore[arg-type]
        commitment = self._db.get_commitment(commitment_id)
        if commitment is None:
            return f"[dismiss_commitment] No commitment with id {commitment_id}."
        if commitment["status"] != "pending":
            return (
                f"[dismiss_commitment] Commitment {commitment_id} is already "
                f"{commitment['status']!r}."
            )

        self._db.mark_commitment_dismissed(commitment_id)
        return f"[dismiss_commitment] Dismissed commitment {commitment_id}."
