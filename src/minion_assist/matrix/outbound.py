"""Matrix outbound message delivery.

Handles sending text messages, draft-placeholder creation (for streaming
preview), in-place edits, reaction posting, and typing indicators.

All methods are async and expect a connected ``matrix-nio`` ``AsyncClient``.

Text is converted from markdown to ``org.matrix.custom.html`` before sending
so Matrix clients render headings, bold, code blocks, and lists natively.
Messages longer than ``config.text_chunk_limit`` are split at paragraph
boundaries before the limit, falling back to newline then hard split.
"""

from __future__ import annotations

from .config import MatrixConfig
from .format import build_content, to_matrix_html


class MatrixOutbound:
    """Async outbound message adapter for a Matrix room.

    Args:
        client: An authenticated matrix-nio ``AsyncClient``.
        config: The :class:`~minion_assist.matrix.config.MatrixConfig` instance.
    """

    def __init__(self, client, config: MatrixConfig) -> None:
        self._client = client
        self._config = config

    async def send_typing(self, room_id: str, typing: bool) -> None:
        """Send a typing indicator to ``room_id``.

        Call with ``typing=True`` when the agent starts processing and
        ``typing=False`` when it finishes.  Matches openclaw behaviour.

        Args:
            room_id: Target Matrix room ID.
            typing:  ``True`` to show indicator, ``False`` to clear it.
        """
        try:
            await self._client.room_typing(room_id, typing_state=typing, timeout=30000)
        except Exception:
            pass  # typing indicator is best-effort

    async def send_text(
        self, room_id: str, text: str, thread_id: str | None = None
    ) -> str:
        """Send a markdown message to ``room_id``, rendered as HTML for Matrix clients.

        Markdown is converted to ``org.matrix.custom.html`` so headings,
        bold, code fences, and lists render natively in Element and Beeper.
        If the text exceeds ``config.text_chunk_limit`` it is split at paragraph
        boundaries into multiple messages.

        Args:
            room_id:   Target Matrix room ID.
            text:      Message body (markdown).
            thread_id: If set, sends as a reply in this thread.

        Returns:
            Event ID of the last sent message.
        """
        # build_content() converts markdown → HTML and splits long text into chunks.
        contents = build_content(text, self._config.text_chunk_limit)
        last_event_id = ""
        for content in contents:
            if thread_id:
                # m.relates_to with rel_type "m.thread" links the message into
                # a Matrix thread.  is_falling_back=True also adds a legacy reply
                # fallback so older clients see it as a normal reply.
                content["m.relates_to"] = {
                    "rel_type": "m.thread",
                    "event_id": thread_id,
                    "is_falling_back": True,
                    "m.in_reply_to": {"event_id": thread_id},
                }
            resp = await self._client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
            )
            last_event_id = getattr(resp, "event_id", "")
        return last_event_id

    async def send_draft(self, room_id: str, initial_text: str = "…") -> str:
        """Post a placeholder message that will be edited as the agent responds.

        Args:
            room_id:      Target Matrix room ID.
            initial_text: Initial placeholder text (default: ellipsis).

        Returns:
            Event ID of the placeholder message.
        """
        resp = await self._client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": initial_text},
        )
        return getattr(resp, "event_id", "")

    async def edit_draft(self, room_id: str, event_id: str, text: str) -> None:
        """Replace the content of a previously sent placeholder message.

        Uses MSC2676 (message edits).  The ``formatted_body`` is rendered from
        markdown so in-progress streaming previews also display with formatting.

        Args:
            room_id:  Room containing the placeholder.
            event_id: Event ID of the placeholder to replace.
            text:     New message body (markdown).
        """
        html = to_matrix_html(text)
        # m.new_content holds the true new message body as understood by modern
        # clients that support MSC2676 message edits.
        new_content: dict = {"msgtype": "m.text", "body": text}
        if html:
            new_content["format"] = "org.matrix.custom.html"
            new_content["formatted_body"] = html

        await self._client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.text",
                # The outer body field uses "* <text>" as a fallback for clients
                # that don't understand message edits — they see it as a plain
                # correction-style message (IRC convention for edits).
                "body": f"* {text}",
                "m.new_content": new_content,
                "m.relates_to": {
                    # "m.replace" tells the homeserver this event replaces another.
                    "rel_type": "m.replace",
                    "event_id": event_id,
                },
            },
        )

    async def finalise_draft(self, room_id: str, event_id: str, text: str) -> None:
        """Perform a final edit of a draft message to mark it as complete.

        Identical to ``edit_draft`` but semantically signals the agent has
        finished generating its response.

        Args:
            room_id:  Room containing the draft.
            event_id: Event ID of the draft message.
            text:     Final message body (markdown).
        """
        await self.edit_draft(room_id, event_id, text)

    async def send_reaction(
        self, room_id: str, target_event_id: str, emoji: str
    ) -> None:
        """Post an emoji reaction to an event.

        Args:
            room_id:         Room containing the target event.
            target_event_id: Event ID to react to.
            emoji:           Emoji string to use as the reaction key.
        """
        await self._client.room_send(
            room_id=room_id,
            message_type="m.reaction",
            content={
                "m.relates_to": {
                    "rel_type": "m.annotation",
                    "event_id": target_event_id,
                    "key": emoji,
                }
            },
        )

    async def resolve_or_create_dm(self, user_id: str) -> str | None:
        """Find or open a DM room with ``user_id``.

        Shared by any feature that needs to message a specific user directly
        (exec approvals, device verification) rather than a configured room.

        Args:
            user_id: The Matrix user ID to DM, e.g. ``"@admin:example.org"``.

        Returns:
            The room ID string, or None on failure.
        """
        try:
            # Check if the client already knows about a DM room with this user
            # from a previous session (loaded during the initial sync).
            direct_rooms = getattr(self._client, "direct_rooms", {})
            if user_id in direct_rooms:
                rooms = direct_rooms[user_id]
                if rooms:
                    # Return the first known DM room for this user.
                    return list(rooms)[0]
            # No existing DM room found — create a new one via the API.
            # is_direct=True tags it as a DM in the homeserver's account data.
            resp = await self._client.room_create(
                is_direct=True,
                invite=[user_id],
            )
            return getattr(resp, "room_id", None)
        except Exception:
            return None
