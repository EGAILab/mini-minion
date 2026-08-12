"""Tests for tui/waveform.py — build_waveform()."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from minion_assist.tui.waveform import build_waveform


@pytest.fixture()
def silent_wav(tmp_path):
    p = tmp_path / "silent.wav"
    sf.write(str(p), np.zeros(44100, dtype="float32"), 44100)
    return p


@pytest.fixture()
def loud_wav(tmp_path):
    """A full-scale square wave — every bucket should read close to 1.0."""
    p = tmp_path / "loud.wav"
    samples = np.ones(44100, dtype="float32")
    samples[::2] = -1.0
    sf.write(str(p), samples, 44100)
    return p


@pytest.fixture()
def stereo_wav(tmp_path):
    p = tmp_path / "stereo.wav"
    samples = np.ones((44100, 2), dtype="float32")
    sf.write(str(p), samples, 44100)
    return p


class TestBuildWaveform:
    def test_returns_requested_bucket_count(self, loud_wav):
        result = build_waveform(loud_wav, buckets=40)
        assert len(result) == 40

    def test_default_bucket_count(self, loud_wav):
        result = build_waveform(loud_wav)
        assert len(result) == 60

    def test_values_are_normalized_to_unit_range(self, loud_wav):
        result = build_waveform(loud_wav, buckets=20)
        assert all(0.0 <= v <= 1.0 + 1e-6 for v in result)
        assert max(result) == pytest.approx(1.0, abs=1e-3)

    def test_silent_audio_returns_all_zeros(self, silent_wav):
        result = build_waveform(silent_wav, buckets=20)
        assert all(v == 0.0 for v in result)

    def test_stereo_audio_is_averaged_to_mono_without_crashing(self, stereo_wav):
        result = build_waveform(stereo_wav, buckets=20)
        assert len(result) == 20

    def test_missing_file_returns_empty_list_not_an_exception(self, tmp_path):
        result = build_waveform(tmp_path / "does-not-exist.wav")
        assert result == []

    def test_corrupt_file_returns_empty_list_not_an_exception(self, tmp_path):
        bad = tmp_path / "corrupt.wav"
        bad.write_bytes(b"RIFF" + b"\x00" * 4 + b"WAVE" + b"not real audio data")
        result = build_waveform(bad)
        assert result == []

    def test_soundfile_unavailable_returns_empty_list(self, loud_wav, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "soundfile":
                raise ImportError("simulated: soundfile not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        result = build_waveform(loud_wav)
        assert result == []
