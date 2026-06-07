"""Tests for multimodal content conversion in the Anthropic provider.

WHY SEPARATE TESTS FROM OPENAI?
--------------------------------
Anthropic's image wire format is DIFFERENT from OpenAI's. Key differences:
  - Anthropic uses {"type": "image", "source": {"type": "base64", ...}}
  - OpenAI uses   {"type": "image_url", "image_url": {"url": "data:..."}}
  - Anthropic REQUIRES at least one text block per user message; OpenAI doesn't.

We test both providers separately to make sure each conversion is correct
and that changing one doesn't accidentally break the other.

Covers _convert_user_content_for_anthropic and _to_anthropic_messages:
- Plain string content passes through unchanged (text-only backward compat).
- Text-only block lists are flattened to a plain string.
- Image blocks become Anthropic's base64 source format (not image_url).
- Image-only content gets a default text block automatically inserted.
- Existing tool-result message merging is unaffected by the image changes.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest

from mini_minion.providers.anthropic import (
    _convert_user_content_for_anthropic,
    _to_anthropic_messages,
)

# Minimal 1x1 PNG (same bytes as other test files).
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


def _image_block(path: str, mime: str = "image/png") -> dict:
    """Build a minimal internal image block."""
    return {
        "type": "image",
        "media_type": mime,
        "path": path,
        "size_bytes": len(MINIMAL_PNG),
        "source_name": Path(path).name,
    }


# ---------------------------------------------------------------------------
# _convert_user_content_for_anthropic
# ---------------------------------------------------------------------------

class TestConvertUserContentForAnthropic:
    def test_string_returned_unchanged(self):
        result = _convert_user_content_for_anthropic("hello")
        assert result == "hello"

    def test_empty_string_returned_unchanged(self):
        result = _convert_user_content_for_anthropic("")
        assert result == ""

    def test_text_only_block_list_flattened_to_string(self):
        blocks = [{"type": "text", "text": "just text here"}]
        result = _convert_user_content_for_anthropic(blocks)
        assert isinstance(result, str)
        assert "just text here" in result

    def test_image_block_converts_to_anthropic_format(self, png_on_disk):
        blocks = [
            {"type": "text", "text": "look at this"},
            _image_block(str(png_on_disk)),
        ]
        result = _convert_user_content_for_anthropic(blocks)
        assert isinstance(result, list)
        image_blocks = [b for b in result if b.get("type") == "image"]
        assert len(image_blocks) == 1

    def test_image_block_has_base64_source(self, png_on_disk):
        blocks = [
            {"type": "text", "text": "x"},
            _image_block(str(png_on_disk)),
        ]
        result = _convert_user_content_for_anthropic(blocks)
        image_block = next(b for b in result if b.get("type") == "image")
        assert image_block["source"]["type"] == "base64"
        assert image_block["source"]["media_type"] == "image/png"

    def test_image_data_decodes_correctly(self, png_on_disk):
        blocks = [
            {"type": "text", "text": "x"},
            _image_block(str(png_on_disk)),
        ]
        result = _convert_user_content_for_anthropic(blocks)
        image_block = next(b for b in result if b.get("type") == "image")
        decoded = base64.b64decode(image_block["source"]["data"])
        assert decoded == MINIMAL_PNG

    def test_text_block_preserved_in_output(self, png_on_disk):
        blocks = [
            {"type": "text", "text": "describe"},
            _image_block(str(png_on_disk)),
        ]
        result = _convert_user_content_for_anthropic(blocks)
        text_blocks = [b for b in result if b.get("type") == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "describe"

    def test_image_only_content_gets_default_text(self, png_on_disk):
        """Anthropic requires at least one text block — insert a default if missing."""
        blocks = [_image_block(str(png_on_disk))]
        result = _convert_user_content_for_anthropic(blocks)
        assert isinstance(result, list)
        text_blocks = [b for b in result if b.get("type") == "text"]
        assert len(text_blocks) >= 1
        assert "analyze" in text_blocks[0]["text"].lower()

    def test_order_text_before_image(self, png_on_disk):
        blocks = [
            {"type": "text", "text": "first"},
            _image_block(str(png_on_disk)),
        ]
        result = _convert_user_content_for_anthropic(blocks)
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image"

    def test_jpeg_mime_preserved(self, tmp_path):
        jpeg_magic = bytes([0xff, 0xd8, 0xff]) + b"\x00" * 20
        jpg = tmp_path / "photo.jpg"
        jpg.write_bytes(jpeg_magic)
        blocks = [
            {"type": "text", "text": "x"},
            _image_block(str(jpg), "image/jpeg"),
        ]
        result = _convert_user_content_for_anthropic(blocks)
        image_block = next(b for b in result if b.get("type") == "image")
        assert image_block["source"]["media_type"] == "image/jpeg"


# ---------------------------------------------------------------------------
# _to_anthropic_messages — existing behavior not regressed
# ---------------------------------------------------------------------------

class TestToAnthropicMessagesMultimodal:
    def test_plain_user_message_unchanged(self):
        messages = [{"role": "user", "content": "hello"}]
        result = _to_anthropic_messages(messages)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    def test_user_message_with_image_converted(self, png_on_disk):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                _image_block(str(png_on_disk)),
            ],
        }]
        result = _to_anthropic_messages(messages)
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        image_blocks = [b for b in result[0]["content"] if b.get("type") == "image"]
        assert len(image_blocks) == 1
        assert image_blocks[0]["source"]["type"] == "base64"

    def test_assistant_message_text_unchanged(self):
        messages = [{"role": "assistant", "content": "sure thing"}]
        result = _to_anthropic_messages(messages)
        assert result[0]["content"] == "sure thing"

    def test_tool_result_still_wrapped_in_user(self):
        messages = [{
            "role": "tool",
            "tool_call_id": "call_abc",
            "content": "tool output",
        }]
        result = _to_anthropic_messages(messages)
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "tool_result"

    def test_consecutive_tool_results_merged(self):
        messages = [
            {"role": "tool", "tool_call_id": "c1", "content": "result1"},
            {"role": "tool", "tool_call_id": "c2", "content": "result2"},
        ]
        result = _to_anthropic_messages(messages)
        # Both tool results should be in a single user message.
        assert len(result) == 1
        assert len(result[0]["content"]) == 2

    def test_mixed_conversation_with_image(self, png_on_disk):
        """A full turn sequence with an image in the user message."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "analyze"},
                    _image_block(str(png_on_disk)),
                ],
            },
            {"role": "assistant", "content": "I see a PNG image."},
        ]
        result = _to_anthropic_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        # User message has multimodal blocks.
        assert isinstance(result[0]["content"], list)
