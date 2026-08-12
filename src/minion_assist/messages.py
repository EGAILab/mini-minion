"""Provider-neutral message content block helpers.

minion-assist's internal content format keeps images and text as plain dicts
so session JSONL files remain readable without base64 blobs.

Internal image block format (stored in history JSONL):
    {
        "type": "image",
        "media_type": "image/png",   # MIME type
        "path": "/abs/path/to/file", # stable file path in attachment store
        "size_bytes": 12345,
        "source_name": "screenshot.png",
    }

Provider conversion happens LATE — in the provider classes, not here.
Base64 is materialized on-demand by providers just before the API request.
This keeps JSONL history files small and human-readable.

Why a separate module?
-----------------------
Both providers and session need to understand content blocks, but neither
should import the other.  Placing helpers here avoids circular imports and
gives a single place to extend the format.
"""
from __future__ import annotations

import uuid

ALLOWED_IMAGE_TYPES: frozenset[str] = frozenset({
    "image/png", "image/jpeg", "image/webp", "image/gif",
})

# Key used to give a message dict a stable identity across process restarts,
# so PostgreSQL mirroring (session/db.py's message_mirrors table) can tell
# "already mirrored this exact message" apart from "new message" — see
# Stage One Phase 2, slice A (memory-implementation-plan.md).
#
# Leading underscore marks this as minion-assist-internal metadata, not part
# of the OpenAI/Anthropic wire format. It rides along in JSONL and in-memory
# history, but MUST be stripped before any message list reaches a provider's
# chat() call — see providers/openai_compatible.py's _prepare_messages_for_openai(),
# the only provider that rebuilds messages via dict-spread (Anthropic's and
# Codex's converters already extract named fields one at a time and would
# drop this key naturally).
EVENT_ID_KEY = "_event_id"


def ensure_event_id(msg: dict) -> str:
    """Return ``msg``'s stable event ID, assigning one if it doesn't have one yet.

    Mutates ``msg`` in place (adds :data:`EVENT_ID_KEY`) so the assignment is
    visible to every other reference to the same dict — in particular,
    ``AgentSession._history`` holds the same object, so the next
    ``ShortTermMemory.save()`` call persists the newly assigned ID to JSONL.

    Idempotent: calling this again on the same dict returns the same ID.

    Args:
        msg: A message dict (user/assistant/tool message from history).

    Returns:
        str: The message's event ID (a UUID4 string), new or existing.
    """
    existing = msg.get(EVENT_ID_KEY)
    if existing:
        return existing
    new_id = str(uuid.uuid4())
    msg[EVENT_ID_KEY] = new_id
    return new_id


def content_has_images(content: str | list) -> bool:
    """Return True if content contains at least one image block.

    Used by providers to decide whether to apply multimodal conversion.
    Plain string content always returns False (old behavior preserved).
    """
    if isinstance(content, str):
        return False
    return any(b.get("type") == "image" for b in content if isinstance(b, dict))


def content_text(content: str | list) -> str:
    """Extract plain text from content (string or block list).

    Images are represented as compact labels — never as base64.
    Used when we need text for memory search queries or logging.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "image":
                # Represent image as a compact label — no base64 leaks out.
                parts.append(f"[image: {block.get('source_name', 'unknown')}]")
    return " ".join(p for p in parts if p)


def content_to_summary_text(content: str | list) -> str:
    """Render content for compaction summaries — never includes base64.

    Unlike content_text(), includes image metadata (path, size) so the
    summarization LLM knows what media was involved, without loading bytes.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "image":
                name = block.get("source_name", "image")
                mt = block.get("media_type", "image")
                size = block.get("size_bytes", 0)
                path = block.get("path", "")
                parts.append(f"[image: {name}, {mt}, {size} bytes, path={path}]")
    return "\n".join(p for p in parts if p)


def strip_media_data(content: str | list) -> str | list:
    """Return content without any inline 'data' fields (base64).

    Used when persisting to JSONL — we store path/metadata, not raw bytes.
    This keeps history files small and prevents accidental base64 sprawl.
    """
    if isinstance(content, str):
        return content
    result = []
    for block in content:
        if isinstance(block, dict):
            # Drop the "data" key if present; keep everything else.
            b = {k: v for k, v in block.items() if k != "data"}
            result.append(b)
        else:
            result.append(block)
    return result


# Per-message character cap for format_message_excerpt() — generous enough to
# capture real content, small enough that one huge message can't make a
# flush note unreadable. Unlike Compactor's _max_head_content, this is a
# fixed constant, not proportional to a model's context window: the flush
# excerpt is a human-readable daily note, not an LLM prompt, so there is no
# token budget to scale against.
_EXCERPT_MAX_CHARS = 1_000


def format_message_excerpt(messages: list[dict]) -> str:
    """Render a list of messages as a plain-text transcript excerpt.

    Used for durable, human-readable checkpoints — e.g. Stage One Phase 2's
    pre-compaction flush, which appends this to a daily note *before*
    ``Compactor`` summarizes and discards the same messages. This is
    deliberately not an LLM prompt: no token-budget tuning, no summarization
    call, just a direct rendering of who said/did what, so it can never fail
    the way an LLM call can and adds no latency.

    Args:
        messages: The messages to render, in order.

    Returns:
        str: One line per message (or per tool call within a message),
            joined with newlines. Empty string if ``messages`` renders to
            nothing (e.g. an empty list).
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = content_to_summary_text(msg.get("content") or "")

        if role == "tool":
            lines.append(f"[tool result]: {content[:_EXCERPT_MAX_CHARS]}")
        elif role == "assistant":
            if content:
                lines.append(f"[assistant]: {content[:_EXCERPT_MAX_CHARS]}")
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                args_preview = fn.get("arguments", "")[:200]
                lines.append(f"[tool call]: {fn.get('name')}({args_preview})")
        elif content:
            lines.append(f"[{role}]: {content[:_EXCERPT_MAX_CHARS]}")

    return "\n".join(lines)


def make_user_content(text: str, attachments: list) -> str | list:
    """Build the user message content from text and a list of MediaAttachment.

    Returns a plain string when there are no attachments (preserves old
    behavior — providers that see a string content never hit the image path).
    Returns a list of content blocks when attachments are present.

    Block order: text first, then images.  Some models require at least one
    text block, so we always include one even for empty user messages.

    Audio attachments (TUI Phase 2) have no provider-side content-block
    representation — no provider in this codebase understands an audio
    input block yet (see media.py's module docstring) — so instead of being
    silently dropped, each one's filename/duration is folded into the text
    block itself, e.g. "[Attached audio: memo.wav, 12.3s]". The model can't
    hear it, but it isn't left unaware the file exists either.

    Args:
        text:        The user's typed message (may be empty).
        attachments: List of MediaAttachment objects from media.py.

    Returns:
        str  — when no attachments are present (backward compatible).
        list — when at least one image attachment exists.
    """
    if not attachments:
        return text

    has_image = any(att.kind == "image" for att in attachments)
    audio_descriptions = [
        "[Attached audio: " + att.source_name
        + (f", {att.duration_seconds:.1f}s" if att.duration_seconds is not None else "")
        + "]"
        for att in attachments if att.kind == "audio"
    ]

    # Use a default prompt when the user typed nothing (avoids a blank
    # message) — worded around whichever attachment kinds are actually
    # present rather than always assuming an image.
    if text.strip():
        prompt_text = text.strip()
    elif has_image:
        prompt_text = "Please analyze the attached image."
    else:
        prompt_text = "Please review the attached file(s)."
    if audio_descriptions:
        prompt_text = "\n".join([prompt_text, *audio_descriptions])

    blocks: list[dict] = [{"type": "text", "text": prompt_text}]

    # Add one image block per attachment. Audio attachments were already
    # folded into the text block above — nothing further to append here.
    for att in attachments:
        if att.kind == "image":
            blocks.append({
                "type": "image",
                "media_type": att.media_type,
                "path": str(att.path),
                "size_bytes": att.size_bytes,
                "source_name": att.source_name,
            })

    # If no image blocks were added (e.g. every attachment was audio-only),
    # fall back to a plain string so the provider sees normal text content.
    if len(blocks) == 1:
        return blocks[0]["text"]

    return blocks


def materialize_image_data(block: dict) -> str:
    """Read image bytes from disk and return base64-encoded string.

    Called by providers just before the API request — never during JSONL
    persistence.  Reading is deferred to this moment so the history file
    stays small even across process restarts.

    Args:
        block: An internal image content block with a "path" key.

    Returns:
        Base64-encoded string of the file's raw bytes.

    Raises:
        ValueError: If the block has no "path" key.
        FileNotFoundError: If the file has been deleted since staging.
    """
    import base64
    path = block.get("path", "")
    if not path:
        raise ValueError("Image block has no path")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")
