"""Tests for multimodal content conversion in OpenAI-compatible provider.

WHY THESE TESTS EXIST
---------------------
The internal content format ({"type": "image", "path": "...", ...}) is
DIFFERENT from the OpenAI wire format ({"type": "image_url", "image_url": {...}}).
These tests verify that the conversion happens correctly so the OpenAI API
accepts the request.

Critically: base64 is materialized FROM DISK at conversion time. That means
these tests need a real file on disk — they can't use a fake path. The
`png_on_disk` fixture provides a minimal valid PNG for this purpose.

Covers _convert_content_for_openai and _prepare_messages_for_openai:
- Text-only string content passes through completely unchanged.
- Text-only block lists are flattened to a plain string for compatibility.
- Image blocks are converted to OpenAI "image_url" data URLs (base64).
- Multiple blocks preserve their order (text block before image block).
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minion_assist.providers.openai_compatible import (
    _convert_content_for_openai,
    _prepare_messages_for_openai,
)

# Minimal PNG bytes (same as in test_media.py).
MINIMAL_PNG = bytes([
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
    0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
    0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41,
    0x54, 0x08, 0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00,
    0x00, 0x00, 0x02, 0x00, 0x01, 0xe2, 0x21, 0xbc,
    0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e,
    0x44, 0xae, 0x42, 0x60, 0x82,
])


@pytest.fixture()
def png_on_disk(tmp_path: Path) -> Path:
    """Write a minimal PNG to a temp file and return its path."""
    p = tmp_path / "test.png"
    p.write_bytes(MINIMAL_PNG)
    return p


# ---------------------------------------------------------------------------
# _convert_content_for_openai
# ---------------------------------------------------------------------------

class TestConvertContentForOpenAI:
    def test_string_returned_unchanged(self):
        result = _convert_content_for_openai("hello world")
        assert result == "hello world"

    def test_empty_string_returned_unchanged(self):
        result = _convert_content_for_openai("")
        assert result == ""

    def test_text_only_block_list_flattened_to_string(self):
        blocks = [{"type": "text", "text": "just text"}]
        result = _convert_content_for_openai(blocks)
        assert isinstance(result, str)
        assert result == "just text"

    def test_multiple_text_blocks_joined(self):
        blocks = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        result = _convert_content_for_openai(blocks)
        assert isinstance(result, str)
        assert "hello" in result and "world" in result

    def test_image_block_converts_to_image_url(self, png_on_disk):
        blocks = [
            {"type": "text", "text": "look"},
            {"type": "image", "media_type": "image/png", "path": str(png_on_disk), "source_name": "t.png"},
        ]
        result = _convert_content_for_openai(blocks)
        assert isinstance(result, list)
        image_blocks = [b for b in result if b.get("type") == "image_url"]
        assert len(image_blocks) == 1

    def test_image_url_contains_data_uri(self, png_on_disk):
        blocks = [
            {"type": "text", "text": "x"},
            {"type": "image", "media_type": "image/png", "path": str(png_on_disk), "source_name": "t.png"},
        ]
        result = _convert_content_for_openai(blocks)
        image_block = next(b for b in result if b.get("type") == "image_url")
        url = image_block["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    def test_image_data_uri_decodes_correctly(self, png_on_disk):
        blocks = [
            {"type": "text", "text": "x"},
            {"type": "image", "media_type": "image/png", "path": str(png_on_disk), "source_name": "t.png"},
        ]
        result = _convert_content_for_openai(blocks)
        image_block = next(b for b in result if b.get("type") == "image_url")
        url = image_block["image_url"]["url"]
        _, b64 = url.split(",", 1)
        decoded = base64.b64decode(b64)
        assert decoded == MINIMAL_PNG

    def test_text_block_preserved_in_mixed_content(self, png_on_disk):
        blocks = [
            {"type": "text", "text": "describe"},
            {"type": "image", "media_type": "image/png", "path": str(png_on_disk), "source_name": "t.png"},
        ]
        result = _convert_content_for_openai(blocks)
        text_blocks = [b for b in result if b.get("type") == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "describe"

    def test_order_preserved_text_before_image(self, png_on_disk):
        blocks = [
            {"type": "text", "text": "first"},
            {"type": "image", "media_type": "image/png", "path": str(png_on_disk), "source_name": "t.png"},
        ]
        result = _convert_content_for_openai(blocks)
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"

    def test_uses_correct_mime_type(self, tmp_path):
        """JPEG mime type should appear in the data URL."""
        jpeg_magic = bytes([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]) + b"\x00" * 20
        jpg = tmp_path / "photo.jpg"
        jpg.write_bytes(jpeg_magic)
        blocks = [
            {"type": "text", "text": "x"},
            {"type": "image", "media_type": "image/jpeg", "path": str(jpg), "source_name": "photo.jpg"},
        ]
        result = _convert_content_for_openai(blocks)
        image_block = next(b for b in result if b.get("type") == "image_url")
        assert "image/jpeg" in image_block["image_url"]["url"]


# ---------------------------------------------------------------------------
# _prepare_messages_for_openai
# ---------------------------------------------------------------------------

class TestPrepareMessagesForOpenAI:
    def test_string_content_passes_through(self):
        messages = [{"role": "user", "content": "hello"}]
        result = _prepare_messages_for_openai(messages)
        assert result[0]["content"] == "hello"

    def test_string_role_preserved(self):
        messages = [{"role": "assistant", "content": "hi"}]
        result = _prepare_messages_for_openai(messages)
        assert result[0]["role"] == "assistant"

    def test_list_content_with_image_converted(self, png_on_disk):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "media_type": "image/png", "path": str(png_on_disk), "source_name": "t.png"},
            ],
        }]
        result = _prepare_messages_for_openai(messages)
        assert isinstance(result[0]["content"], list)
        assert any(b.get("type") == "image_url" for b in result[0]["content"])

    def test_multiple_messages_all_converted(self, png_on_disk):
        messages = [
            {"role": "user", "content": "text only"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "with image"},
                    {"type": "image", "media_type": "image/png", "path": str(png_on_disk), "source_name": "t.png"},
                ],
            },
        ]
        result = _prepare_messages_for_openai(messages)
        assert result[0]["content"] == "text only"
        assert isinstance(result[1]["content"], list)

    def test_original_messages_not_mutated(self):
        original_content = [{"type": "text", "text": "hi"}]
        messages = [{"role": "user", "content": original_content}]
        _prepare_messages_for_openai(messages)
        # The original list object should be unchanged.
        assert messages[0]["content"] is original_content

    def test_strips_internal_event_id_key(self):
        """_event_id (session/db.py mirroring metadata) must never reach the API."""
        messages = [{"role": "user", "content": "hi", "_event_id": "abc-123"}]
        result = _prepare_messages_for_openai(messages)
        assert "_event_id" not in result[0]
        assert result[0]["content"] == "hi"
