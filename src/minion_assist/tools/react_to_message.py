"""ReactToMessageTool — add an emoji reaction to a Matrix message.

Injected as an extra_tool for Matrix group-chat turns only.  The agent can call
``react_to_message`` instead of (or in addition to) sending a text reply when
a lightweight acknowledgement is more appropriate than prose.

The tool is a thin wrapper around :meth:`MatrixOutbound.send_reaction`, which
already handles the Matrix API call.  The room_id and event_id are baked in
at construction time so the agent only needs to supply the emoji.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .base import Tool, ToolSchema

if TYPE_CHECKING:
    from ..matrix.outbound import MatrixOutbound


class ReactToMessageTool(Tool):
    """Tool that lets the agent react to the current inbound message with an emoji.

    Args:
        outbound:  The Matrix outbound client (has ``send_reaction`` coroutine).
        room_id:   The Matrix room ID of the message being reacted to.
        event_id:  The event ID of the inbound message to react to.
        loop:      The asyncio event loop running the Matrix client.
    """

    def __init__(
        self,
        outbound: "MatrixOutbound",
        room_id: str,
        event_id: str,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._outbound = outbound
        self._room_id = room_id
        self._event_id = event_id
        self._loop = loop

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="react_to_message",
            description=(
                "React to the current user message with an emoji reaction. "
                "Use this for lightweight social acknowledgements (👍, ❤️, 😂, 🤔, ✅) "
                "when you want to acknowledge a message without sending a full text reply. "
                "One reaction per message — pick the emoji that fits best."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "emoji": {
                        "type": "string",
                        "description": "The emoji to react with, e.g. '👍', '❤️', '😂'.",
                    },
                },
                "required": ["emoji"],
            },
            is_read_only=True,
        )

    def execute(self, **kwargs: object) -> str:
        """Send the emoji reaction via the Matrix outbound client.

        Args:
            emoji (str): Emoji character(s) to react with.

        Returns:
            str: Confirmation or error message.
        """
        emoji = str(kwargs.get("emoji", "")).strip()
        if not emoji:
            return "[react_to_message] No emoji provided."

        # send_reaction is a coroutine; schedule it on the Matrix event loop from
        # this synchronous tool execute() method.
        future = asyncio.run_coroutine_threadsafe(
            self._outbound.send_reaction(self._room_id, self._event_id, emoji),
            self._loop,
        )
        try:
            future.result(timeout=10)
            return f"[react_to_message] Reacted with {emoji}."
        except Exception as exc:
            return f"[react_to_message] Failed to react: {exc}"
