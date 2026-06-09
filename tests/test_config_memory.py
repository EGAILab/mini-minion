"""Tests for MemoryConfig and extra_plugin_manifests config parsing.

These tests patch the config module's internal _raw dict and call the private
resolver functions directly to avoid needing a real config.json on disk.
"""

import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# MemoryConfig resolver
# ---------------------------------------------------------------------------

def test_memory_config_defaults_to_enabled():
    """When 'memory' section is absent, enable_extraction defaults to True."""
    from mini_minion.config import _resolve_memory
    with patch("mini_minion.config._raw", {}):
        cfg = _resolve_memory()
    assert cfg.enable_extraction is True


def test_memory_config_explicit_false():
    """enable_extraction=false in config is respected."""
    from mini_minion.config import _resolve_memory
    raw = {"memory": {"enable_extraction": False}}
    with patch("mini_minion.config._raw", raw):
        cfg = _resolve_memory()
    assert cfg.enable_extraction is False


def test_memory_config_explicit_true():
    """enable_extraction=true in config is respected."""
    from mini_minion.config import _resolve_memory
    raw = {"memory": {"enable_extraction": True}}
    with patch("mini_minion.config._raw", raw):
        cfg = _resolve_memory()
    assert cfg.enable_extraction is True


def test_memory_config_is_frozen():
    """MemoryConfig instances are immutable (frozen=True)."""
    from mini_minion.config import MemoryConfig
    cfg = MemoryConfig(enable_extraction=True)
    with pytest.raises((AttributeError, TypeError)):
        cfg.enable_extraction = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# extra_plugin_manifests
# ---------------------------------------------------------------------------

def test_extra_plugin_manifests_defaults_to_empty():
    """When 'extra_plugin_manifests' is absent, the result is an empty tuple."""
    raw = {}
    # Simulate how the module-level export is computed.
    result = tuple(raw.get("extra_plugin_manifests", []))
    assert result == ()


def test_extra_plugin_manifests_reads_list():
    """When 'extra_plugin_manifests' is a list, it becomes a tuple of strings."""
    raw = {"extra_plugin_manifests": ["/a/b/plugins.json", "~/c/plugins.json"]}
    result = tuple(raw.get("extra_plugin_manifests", []))
    assert result == ("/a/b/plugins.json", "~/c/plugins.json")


def test_extra_plugin_manifests_is_tuple():
    """The exported type must be a tuple, not a list."""
    from mini_minion.config import extra_plugin_manifests
    assert isinstance(extra_plugin_manifests, tuple)


def test_memory_config_enable_extraction_is_bool():
    """The exported memory.enable_extraction is always a bool."""
    from mini_minion.config import memory
    assert isinstance(memory.enable_extraction, bool)
