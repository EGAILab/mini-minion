"""Provider-neutral message content block helpers.

minion-assistant's internal content format keeps images and text as plain dicts
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

ALLOWED_IMAGE_TYPES: frozenset[str] = frozenset({
    "image/png", "image/jpeg", "image/webp", "image/gif",
})


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


def make_user_content(text: str, attachments: list) -> str | list:
    """Build the user message content from text and a list of MediaAttachment.

    Returns a plain string when there are no attachments (preserves old
    behavior — providers that see a string content never hit the image path).
    Returns a list of content blocks when attachments are present.

    Block order: text first, then images.  Some models require at least one
    text block, so we always include one even for empty user messages.

    Args:
        text:        The user's typed message (may be empty).
        attachments: List of MediaAttachment objects from media.py.

    Returns:
        str  — when no attachments are present (backward compatible).
        list — when at least one image attachment exists.
    """
    if not attachments:
        return text

    blocks: list[dict] = []

    # Text block first — required by most vision models even if empty.
    # Use a default prompt when the user typed nothing (avoids a blank message).
    prompt_text = text.strip() if text.strip() else "Please analyze the attached image."
    blocks.append({"type": "text", "text": prompt_text})

    # Add one image block per attachment.
    # Only image kind is supported in slice 1 — future slices may add audio.
    for att in attachments:
        if att.kind == "image":
            blocks.append({
                "type": "image",
                "media_type": att.media_type,
                "path": str(att.path),
                "size_bytes": att.size_bytes,
                "source_name": att.source_name,
            })

    # If no image blocks were added (e.g. all attachments were non-image),
    # fall back to a plain string so the provider sees normal content.
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
