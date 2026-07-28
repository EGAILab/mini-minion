"""Tests for minion_assist.messages — provider-neutral content block helpers.

Covers:
- make_user_content: no attachments → string; with attachments → block list
- content_has_images: detects image blocks
- content_text: extracts plain text, images become labels
- content_to_summary_text: includes image metadata, no base64
- strip_media_data: removes "data" keys for JSONL persistence
- materialize_image_data: reads base64 from disk
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from minion_assist.messages import (
    ALLOWED_IMAGE_TYPES,
    EVENT_ID_KEY,
    content_has_images,
    content_text,
    content_to_summary_text,
    ensure_event_id,
    make_user_content,
    materialize_image_data,
    strip_media_data,
)

# ---------------------------------------------------------------------------
# Minimal PNG bytes for tests that need a real image file.
# ---------------------------------------------------------------------------
MINIMAL_PNG = bytes([
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,  # PNG signature
    0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 pixels
    0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
    0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41,  # IDAT chunk
    0x54, 0x08, 0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00,
    0x00, 0x00, 0x02, 0x00, 0x01, 0xe2, 0x21, 0xbc,
    0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e,  # IEND chunk
    0x44, 0xae, 0x42, 0x60, 0x82,
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_attachment(path: str, source_name: str = "test.png") -> MagicMock:
    """Build a fake MediaAttachment for testing make_user_content."""
    att = MagicMock()
    att.kind = "image"
    att.media_type = "image/png"
    att.path = path
    att.size_bytes = 100
    att.source_name = source_name
    return att


def _make_image_block(source_name: str = "shot.png", path: str = "/tmp/shot.png") -> dict:
    return {
        "type": "image",
        "media_type": "image/png",
        "path": path,
        "size_bytes": 200,
        "source_name": source_name,
    }


# ---------------------------------------------------------------------------
# content_has_images
# ---------------------------------------------------------------------------

class TestContentHasImages:
    def test_string_content_returns_false(self):
        assert content_has_images("hello world") is False

    def test_empty_string_returns_false(self):
        assert content_has_images("") is False

    def test_text_only_list_returns_false(self):
        blocks = [{"type": "text", "text": "hello"}]
        assert content_has_images(blocks) is False

    def test_list_with_image_block_returns_true(self):
        blocks = [
            {"type": "text", "text": "look at this"},
            _make_image_block(),
        ]
        assert content_has_images(blocks) is True

    def test_image_only_list_returns_true(self):
        assert content_has_images([_make_image_block()]) is True

    def test_empty_list_returns_false(self):
        assert content_has_images([]) is False

    def test_non_dict_elements_are_ignored(self):
        # Guard against malformed history entries.
        assert content_has_images(["not a dict", 42]) is False


# ---------------------------------------------------------------------------
# content_text
# ---------------------------------------------------------------------------

class TestContentText:
    def test_string_returned_unchanged(self):
        assert content_text("hello") == "hello"

    def test_empty_string(self):
        assert content_text("") == ""

    def test_text_block_list_returns_text(self):
        blocks = [{"type": "text", "text": "hello world"}]
        assert content_text(blocks) == "hello world"

    def test_multiple_text_blocks_joined(self):
        blocks = [
            {"type": "text", "text": "one"},
            {"type": "text", "text": "two"},
        ]
        assert content_text(blocks) == "one two"

    def test_image_block_returns_label(self):
        blocks = [_make_image_block("diagram.png")]
        result = content_text(blocks)
        assert "[image: diagram.png]" in result

    def test_mixed_content_combines_text_and_label(self):
        blocks = [
            {"type": "text", "text": "analyze this"},
            _make_image_block("chart.png"),
        ]
        result = content_text(blocks)
        assert "analyze this" in result
        assert "[image: chart.png]" in result

    def test_no_base64_in_output(self):
        # Even if a block had a "data" key, content_text must not emit it.
        block = {**_make_image_block(), "data": "base64stuff"}
        result = content_text([block])
        assert "base64stuff" not in result


# ---------------------------------------------------------------------------
# content_to_summary_text
# ---------------------------------------------------------------------------

class TestContentToSummaryText:
    def test_string_returned_unchanged(self):
        assert content_to_summary_text("hello") == "hello"

    def test_text_block_list(self):
        blocks = [{"type": "text", "text": "summarize me"}]
        assert content_to_summary_text(blocks) == "summarize me"

    def test_image_block_shows_metadata(self):
        block = {
            "type": "image",
            "media_type": "image/jpeg",
            "path": "/store/2024-01-01/abc-shot.jpg",
            "size_bytes": 5000,
            "source_name": "shot.jpg",
        }
        result = content_to_summary_text([block])
        assert "shot.jpg" in result
        assert "image/jpeg" in result
        assert "5000" in result
        assert "/store/2024-01-01/abc-shot.jpg" in result

    def test_no_base64_in_summary(self):
        block = {**_make_image_block(), "data": "AAABBBCCC"}
        result = content_to_summary_text([block])
        assert "AAABBBCCC" not in result

    def test_mixed_content(self):
        blocks = [
            {"type": "text", "text": "here is the image"},
            _make_image_block("logo.png"),
        ]
        result = content_to_summary_text(blocks)
        assert "here is the image" in result
        assert "logo.png" in result


# ---------------------------------------------------------------------------
# strip_media_data
# ---------------------------------------------------------------------------

class TestStripMediaData:
    def test_string_returned_unchanged(self):
        assert strip_media_data("hello") == "hello"

    def test_text_block_unchanged(self):
        blocks = [{"type": "text", "text": "hi"}]
        result = strip_media_data(blocks)
        assert result == [{"type": "text", "text": "hi"}]

    def test_image_block_data_removed(self):
        blocks = [{
            "type": "image",
            "media_type": "image/png",
            "path": "/some/path.png",
            "size_bytes": 100,
            "source_name": "path.png",
            "data": "BASE64DATA",
        }]
        result = strip_media_data(blocks)
        assert isinstance(result, list)
        assert "data" not in result[0]
        # Other fields are preserved.
        assert result[0]["path"] == "/some/path.png"
        assert result[0]["media_type"] == "image/png"

    def test_block_without_data_unchanged(self):
        block = _make_image_block()
        result = strip_media_data([block])
        assert result[0] == block

    def test_original_list_not_mutated(self):
        block = {"type": "image", "data": "secret", "path": "/p.png"}
        original = [block]
        strip_media_data(original)
        # The original block still has the "data" key.
        assert "data" in original[0]


# ---------------------------------------------------------------------------
# make_user_content
# ---------------------------------------------------------------------------

class TestMakeUserContent:
    def test_no_attachments_returns_string(self):
        result = make_user_content("hello", [])
        assert isinstance(result, str)
        assert result == "hello"

    def test_no_attachments_empty_text_returns_string(self):
        result = make_user_content("", [])
        assert isinstance(result, str)
        assert result == ""

    def test_with_image_attachment_returns_list(self):
        att = _make_image_attachment("/tmp/test.png")
        result = make_user_content("look at this", [att])
        assert isinstance(result, list)

    def test_text_block_is_first(self):
        att = _make_image_attachment("/tmp/test.png")
        result = make_user_content("describe this", [att])
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "describe this"

    def test_image_block_follows_text(self):
        att = _make_image_attachment("/tmp/img.png", "img.png")
        result = make_user_content("hi", [att])
        assert result[1]["type"] == "image"
        assert result[1]["source_name"] == "img.png"
        assert result[1]["media_type"] == "image/png"

    def test_empty_text_with_image_gets_default_prompt(self):
        att = _make_image_attachment("/tmp/test.png")
        result = make_user_content("", [att])
        assert result[0]["text"] == "Please analyze the attached image."

    def test_multiple_images(self):
        att1 = _make_image_attachment("/tmp/a.png", "a.png")
        att2 = _make_image_attachment("/tmp/b.png", "b.png")
        result = make_user_content("compare", [att1, att2])
        image_blocks = [b for b in result if b.get("type") == "image"]
        assert len(image_blocks) == 2

    def test_non_image_attachments_skipped(self):
        # Non-image kind attachments are ignored (audio not supported in slice 1).
        att = MagicMock()
        att.kind = "audio"
        result = make_user_content("text only", [att])
        # No image blocks → falls back to string.
        assert isinstance(result, str)
        assert result == "text only"

    def test_image_block_has_path(self):
        att = _make_image_attachment("/abs/path/img.png")
        result = make_user_content("x", [att])
        assert result[1]["path"] == "/abs/path/img.png"


# ---------------------------------------------------------------------------
# materialize_image_data
# ---------------------------------------------------------------------------

class TestMaterializeImageData:
    def test_reads_and_encodes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test.png"
            img_path.write_bytes(MINIMAL_PNG)
            block = {"type": "image", "path": str(img_path)}
            result = materialize_image_data(block)
            # Should be valid base64 that decodes back to the original bytes.
            decoded = base64.b64decode(result)
            assert decoded == MINIMAL_PNG

    def test_raises_on_missing_path(self):
        with pytest.raises(ValueError, match="no path"):
            materialize_image_data({"type": "image", "path": ""})

    def test_raises_on_missing_file(self):
        block = {"type": "image", "path": "/nonexistent/path/image.png"}
        with pytest.raises(FileNotFoundError):
            materialize_image_data(block)


# ---------------------------------------------------------------------------
# ALLOWED_IMAGE_TYPES
# ---------------------------------------------------------------------------

class TestAllowedImageTypes:
    def test_contains_expected_types(self):
        assert "image/png" in ALLOWED_IMAGE_TYPES
        assert "image/jpeg" in ALLOWED_IMAGE_TYPES
        assert "image/webp" in ALLOWED_IMAGE_TYPES
        assert "image/gif" in ALLOWED_IMAGE_TYPES

    def test_pdf_not_allowed(self):
        assert "application/pdf" not in ALLOWED_IMAGE_TYPES


# ---------------------------------------------------------------------------
# ensure_event_id
# ---------------------------------------------------------------------------

class TestEnsureEventId:
    def test_assigns_id_to_message_without_one(self):
        msg = {"role": "user", "content": "hi"}
        event_id = ensure_event_id(msg)
        assert msg[EVENT_ID_KEY] == event_id
        assert event_id  # non-empty string

    def test_is_idempotent_on_repeated_calls(self):
        msg = {"role": "user", "content": "hi"}
        first = ensure_event_id(msg)
        second = ensure_event_id(msg)
        assert first == second

    def test_preserves_existing_id(self):
        msg = {"role": "user", "content": "hi", EVENT_ID_KEY: "already-set"}
        assert ensure_event_id(msg) == "already-set"

    def test_different_messages_get_different_ids(self):
        msg1 = {"role": "user", "content": "a"}
        msg2 = {"role": "user", "content": "b"}
        assert ensure_event_id(msg1) != ensure_event_id(msg2)

    def test_mutates_dict_in_place(self):
        msg = {"role": "user", "content": "hi"}
        assert EVENT_ID_KEY not in msg
        ensure_event_id(msg)
        assert EVENT_ID_KEY in msg
