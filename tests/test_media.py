"""Tests for minion_assist.media — file staging and validation.

Covers:
- stage_attachment with a real PNG file (minimal 1x1 PNG)
- Reject too-large files
- Reject non-image files (wrong bytes)
- Reject .env paths and other dangerous paths
- describe_attachment output format
- Content-hash deduplication (same file staged twice lands at same path)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from minion_assist.media import (
    MediaAttachment,
    describe_attachment,
    stage_attachment,
)

# ---------------------------------------------------------------------------
# Minimal 1x1 PNG for tests (valid PNG signature + minimal IHDR/IDAT/IEND).
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

# Minimal JPEG bytes (just the SOI marker + a few bytes so sniff passes).
MINIMAL_JPEG = bytes([
    0xff, 0xd8, 0xff, 0xe0,  # SOI + APP0 marker
    0x00, 0x10,              # APP0 length
    0x4a, 0x46, 0x49, 0x46, 0x00,  # "JFIF\0"
    0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
    0xff, 0xd9,              # EOI
])


@pytest.fixture()
def tmp_store(tmp_path: Path) -> Path:
    """Return a fresh temp directory to use as the attachment store."""
    store = tmp_path / "attachments"
    store.mkdir()
    return store


@pytest.fixture()
def png_file(tmp_path: Path) -> Path:
    """Write a minimal valid PNG to a temp file and return its path."""
    p = tmp_path / "test.png"
    p.write_bytes(MINIMAL_PNG)
    return p


@pytest.fixture()
def jpeg_file(tmp_path: Path) -> Path:
    """Write a minimal JPEG to a temp file and return its path."""
    p = tmp_path / "photo.jpg"
    p.write_bytes(MINIMAL_JPEG)
    return p


# ---------------------------------------------------------------------------
# stage_attachment — happy path
# ---------------------------------------------------------------------------

class TestStageAttachmentHappyPath:
    def test_returns_media_attachment(self, png_file, tmp_store):
        att = stage_attachment(png_file, tmp_store)
        assert isinstance(att, MediaAttachment)

    def test_kind_is_image(self, png_file, tmp_store):
        att = stage_attachment(png_file, tmp_store)
        assert att.kind == "image"

    def test_media_type_is_png(self, png_file, tmp_store):
        att = stage_attachment(png_file, tmp_store)
        assert att.media_type == "image/png"

    def test_source_name_is_filename(self, png_file, tmp_store):
        att = stage_attachment(png_file, tmp_store)
        assert att.source_name == "test.png"

    def test_size_bytes_matches_file_size(self, png_file, tmp_store):
        att = stage_attachment(png_file, tmp_store)
        assert att.size_bytes == len(MINIMAL_PNG)

    def test_staged_file_exists_on_disk(self, png_file, tmp_store):
        att = stage_attachment(png_file, tmp_store)
        assert att.path.exists()

    def test_staged_file_content_matches_original(self, png_file, tmp_store):
        att = stage_attachment(png_file, tmp_store)
        assert att.path.read_bytes() == MINIMAL_PNG

    def test_id_is_8_hex_chars(self, png_file, tmp_store):
        att = stage_attachment(png_file, tmp_store)
        assert len(att.id) == 8
        assert all(c in "0123456789abcdef" for c in att.id)

    def test_staged_path_is_under_media_dir(self, png_file, tmp_store):
        att = stage_attachment(png_file, tmp_store)
        assert str(att.path).startswith(str(tmp_store))

    def test_staged_path_includes_date_subdir(self, png_file, tmp_store):
        from datetime import date
        att = stage_attachment(png_file, tmp_store)
        today = date.today().isoformat()
        assert today in str(att.path)

    def test_jpeg_accepted(self, jpeg_file, tmp_store):
        att = stage_attachment(jpeg_file, tmp_store)
        assert att.media_type == "image/jpeg"

    def test_deduplication_same_content_same_path(self, png_file, tmp_store):
        """Staging the same file twice should land at the same destination."""
        att1 = stage_attachment(png_file, tmp_store)
        att2 = stage_attachment(png_file, tmp_store)
        assert att1.path == att2.path


# ---------------------------------------------------------------------------
# stage_attachment — validation errors
# ---------------------------------------------------------------------------

class TestStageAttachmentErrors:
    def test_file_not_found(self, tmp_store):
        missing = Path("/nonexistent/absolutely/missing.png")
        with pytest.raises(FileNotFoundError):
            stage_attachment(missing, tmp_store)

    def test_rejects_non_image_file(self, tmp_path, tmp_store):
        """A text file with .txt extension should be rejected as unsupported type."""
        txt = tmp_path / "notes.txt"
        txt.write_text("just text here")
        with pytest.raises(ValueError, match="Unsupported file type"):
            stage_attachment(txt, tmp_store)

    def test_rejects_file_with_wrong_bytes_for_extension(self, tmp_path, tmp_store):
        """A ZIP file renamed to .png should be rejected (bytes don't match PNG magic)."""
        fake_png = tmp_path / "disguised.png"
        # ZIP magic bytes — definitely not a PNG.
        fake_png.write_bytes(b"PK\x03\x04" + b"\x00" * 50)
        with pytest.raises(ValueError, match="does not appear to be a valid image"):
            stage_attachment(fake_png, tmp_store)

    def test_rejects_too_large_file(self, tmp_path, tmp_store):
        """Files over 15 MB should be rejected before staging."""
        from minion_assist.media import _MAX_IMAGE_BYTES

        big = tmp_path / "big.png"
        # Write a valid PNG header followed by enough zeros to exceed the limit.
        big.write_bytes(MINIMAL_PNG + b"\x00" * (_MAX_IMAGE_BYTES + 1))
        with pytest.raises(ValueError, match="too large"):
            stage_attachment(big, tmp_store)

    def test_rejects_env_file(self, tmp_path, tmp_store):
        """Files named .env should be rejected as potential secrets."""
        env_file = tmp_path / ".env"
        env_file.write_bytes(MINIMAL_PNG)  # valid PNG bytes, but dangerous name
        with pytest.raises(ValueError, match="Rejected"):
            stage_attachment(env_file, tmp_store)

    def test_rejects_git_path(self, tmp_path, tmp_store):
        """Paths containing .git segment should be rejected."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        git_obj = git_dir / "config"
        git_obj.write_bytes(MINIMAL_PNG)
        with pytest.raises(ValueError, match="Rejected"):
            stage_attachment(git_obj, tmp_store)

    def test_rejects_dotenv_variant(self, tmp_path, tmp_store):
        """Files matching .env* pattern should also be rejected."""
        env_local = tmp_path / ".env.local"
        env_local.write_bytes(MINIMAL_PNG)
        with pytest.raises(ValueError, match="Rejected"):
            stage_attachment(env_local, tmp_store)


# ---------------------------------------------------------------------------
# describe_attachment
# ---------------------------------------------------------------------------

class TestDescribeAttachment:
    def test_includes_source_name(self, png_file, tmp_store):
        att = stage_attachment(png_file, tmp_store)
        desc = describe_attachment(att)
        assert "test.png" in desc

    def test_includes_media_type(self, png_file, tmp_store):
        att = stage_attachment(png_file, tmp_store)
        desc = describe_attachment(att)
        assert "image/png" in desc

    def test_includes_size_in_kb(self, png_file, tmp_store):
        att = stage_attachment(png_file, tmp_store)
        desc = describe_attachment(att)
        assert "KB" in desc

    def test_format_matches_pattern(self, png_file, tmp_store):
        att = stage_attachment(png_file, tmp_store)
        desc = describe_attachment(att)
        # Expected: "test.png (image/png, N KB)"
        assert "(" in desc and ")" in desc
