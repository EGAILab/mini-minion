"""Tests for config.py validation — _validate() and ConfigError/ConfigIssue."""

import copy

import pytest

from mini_minion.config import ConfigError, ConfigIssue, _validate

# ---------------------------------------------------------------------------
# Minimal valid config used as a base; deep-copied per test before mutation.
# ---------------------------------------------------------------------------

_VALID: dict = {
    "models": {
        "providers": {
            "lmstudio": {
                "api": "lmstudio",
                "baseUrl": "http://localhost:1234/v1",
                "models": [
                    {"id": "qwen-9b", "contextWindow": 8192, "maxOutputTokens": 4096}
                ],
            }
        }
    },
    "agents": {
        "main": {"model": "lmstudio/qwen-9b"},
        "researcher": {"model": "lmstudio/qwen-9b", "route_prefix": "/research"},
    },
    "streaming": {"chat_mode": False, "task_mode": False},
    "compaction": {"preserve_tokens": 4000},
}


def _v(raw: dict) -> list[ConfigIssue]:
    return _validate(raw)


def _paths(issues: list[ConfigIssue]) -> list[str]:
    return [i.path for i in issues]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_config_has_no_issues():
    assert _v(copy.deepcopy(_VALID)) == []


# ---------------------------------------------------------------------------
# models.providers
# ---------------------------------------------------------------------------


def test_missing_providers_causes_unknown_provider_on_agents():
    # When providers is absent (empty dict), agent references report unknown provider.
    raw = copy.deepcopy(_VALID)
    del raw["models"]["providers"]
    issues = _v(raw)
    assert any("agents" in i.path and "Unknown provider" in i.message for i in issues)


def test_provider_missing_api_reported():
    raw = copy.deepcopy(_VALID)
    del raw["models"]["providers"]["lmstudio"]["api"]
    issues = _v(raw)
    assert "models.providers.lmstudio.api" in _paths(issues)


def test_provider_empty_api_reported():
    raw = copy.deepcopy(_VALID)
    raw["models"]["providers"]["lmstudio"]["api"] = ""
    issues = _v(raw)
    assert "models.providers.lmstudio.api" in _paths(issues)


def test_inline_api_key_reported():
    raw = copy.deepcopy(_VALID)
    raw["models"]["providers"]["lmstudio"]["apiKey"] = "sk-secret"
    issues = _v(raw)
    issue = next(i for i in issues if i.path == "models.providers.lmstudio.apiKey")
    assert "environment variable" in issue.message


def test_empty_inline_api_key_not_reported():
    """Empty or absent apiKey is fine — it falls through to the env var silently."""
    raw = copy.deepcopy(_VALID)
    raw["models"]["providers"]["lmstudio"]["apiKey"] = ""
    assert _v(raw) == []


def test_model_missing_context_window_reported():
    raw = copy.deepcopy(_VALID)
    del raw["models"]["providers"]["lmstudio"]["models"][0]["contextWindow"]
    issues = _v(raw)
    assert "models.providers.lmstudio.models[0].contextWindow" in _paths(issues)


def test_model_missing_max_output_tokens_reported():
    raw = copy.deepcopy(_VALID)
    del raw["models"]["providers"]["lmstudio"]["models"][0]["maxOutputTokens"]
    issues = _v(raw)
    assert "models.providers.lmstudio.models[0].maxOutputTokens" in _paths(issues)


def test_model_zero_context_window_reported():
    raw = copy.deepcopy(_VALID)
    raw["models"]["providers"]["lmstudio"]["models"][0]["contextWindow"] = 0
    issues = _v(raw)
    assert "models.providers.lmstudio.models[0].contextWindow" in _paths(issues)


def test_model_negative_max_output_tokens_reported():
    raw = copy.deepcopy(_VALID)
    raw["models"]["providers"]["lmstudio"]["models"][0]["maxOutputTokens"] = -1
    issues = _v(raw)
    assert "models.providers.lmstudio.models[0].maxOutputTokens" in _paths(issues)


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------


def test_empty_agents_reported():
    raw = copy.deepcopy(_VALID)
    raw["agents"] = {}
    issues = _v(raw)
    assert "agents" in _paths(issues)


def test_agent_missing_model_reported():
    raw = copy.deepcopy(_VALID)
    del raw["agents"]["main"]["model"]
    issues = _v(raw)
    assert "agents.main.model" in _paths(issues)


def test_agent_model_without_slash_reported():
    raw = copy.deepcopy(_VALID)
    raw["agents"]["main"]["model"] = "qwen-9b"
    issues = _v(raw)
    issue = next(i for i in issues if i.path == "agents.main.model")
    assert "provider/model-id" in issue.message


def test_agent_unknown_provider_reported():
    raw = copy.deepcopy(_VALID)
    raw["agents"]["main"]["model"] = "openai/qwen-9b"
    issues = _v(raw)
    issue = next(i for i in issues if i.path == "agents.main.model")
    assert "Unknown provider" in issue.message


def test_agent_unknown_provider_with_typo_suggests_correction():
    raw = copy.deepcopy(_VALID)
    # "lmstdio" is close enough to "lmstudio" for difflib to suggest it.
    raw["agents"]["main"]["model"] = "lmstdio/qwen-9b"
    issues = _v(raw)
    issue = next(i for i in issues if i.path == "agents.main.model")
    assert "lmstudio" in issue.message


def test_agent_unknown_model_id_reported():
    raw = copy.deepcopy(_VALID)
    raw["agents"]["main"]["model"] = "lmstudio/nonexistent-model"
    issues = _v(raw)
    issue = next(i for i in issues if i.path == "agents.main.model")
    assert "Unknown model" in issue.message


def test_agent_unknown_model_id_with_typo_suggests_correction():
    raw = copy.deepcopy(_VALID)
    # "qwen-9bb" is close enough to "qwen-9b" for difflib.
    raw["agents"]["main"]["model"] = "lmstudio/qwen-9bb"
    issues = _v(raw)
    issue = next(i for i in issues if i.path == "agents.main.model")
    assert "qwen-9b" in issue.message


# ---------------------------------------------------------------------------
# route_prefix
# ---------------------------------------------------------------------------


def test_route_prefix_missing_slash_reported():
    raw = copy.deepcopy(_VALID)
    raw["agents"]["researcher"]["route_prefix"] = "research"
    issues = _v(raw)
    issue = next(i for i in issues if i.path == "agents.researcher.route_prefix")
    assert "'/'" in issue.message or "start with" in issue.message


def test_duplicate_route_prefix_reported():
    raw = copy.deepcopy(_VALID)
    raw["agents"]["main"]["route_prefix"] = "/research"  # duplicate of researcher
    issues = _v(raw)
    paths = _paths(issues)
    assert any("route_prefix" in p for p in paths)


def test_all_agents_prefixed_reported():
    """Every agent having a route_prefix leaves no default fallback — must be flagged."""
    raw = copy.deepcopy(_VALID)
    raw["agents"]["main"]["route_prefix"] = "/main"  # now both agents have prefixes
    issues = _v(raw)
    assert any(i.path == "agents" and "default fallback" in i.message for i in issues)


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------


def test_streaming_chat_mode_string_reported():
    raw = copy.deepcopy(_VALID)
    raw["streaming"]["chat_mode"] = "true"
    issues = _v(raw)
    issue = next(i for i in issues if i.path == "streaming.chat_mode")
    assert "boolean" in issue.message.lower()


def test_streaming_task_mode_integer_reported():
    raw = copy.deepcopy(_VALID)
    raw["streaming"]["task_mode"] = 1
    issues = _v(raw)
    assert "streaming.task_mode" in _paths(issues)


def test_streaming_true_is_valid():
    raw = copy.deepcopy(_VALID)
    raw["streaming"]["chat_mode"] = True
    assert _v(raw) == []


# ---------------------------------------------------------------------------
# compaction
# ---------------------------------------------------------------------------


def test_compaction_preserve_tokens_zero_reported():
    raw = copy.deepcopy(_VALID)
    raw["compaction"]["preserve_tokens"] = 0
    issues = _v(raw)
    assert "compaction.preserve_tokens" in _paths(issues)


def test_compaction_preserve_tokens_negative_reported():
    raw = copy.deepcopy(_VALID)
    raw["compaction"]["preserve_tokens"] = -100
    issues = _v(raw)
    assert "compaction.preserve_tokens" in _paths(issues)


def test_compaction_preserve_tokens_string_reported():
    raw = copy.deepcopy(_VALID)
    raw["compaction"]["preserve_tokens"] = "4000"
    issues = _v(raw)
    assert "compaction.preserve_tokens" in _paths(issues)


# ---------------------------------------------------------------------------
# ConfigError and ConfigIssue
# ---------------------------------------------------------------------------


def test_config_error_collects_all_issues():
    """ConfigError must report every issue at once, not stop at the first."""
    raw = copy.deepcopy(_VALID)
    # Two independent problems: bad api + bad streaming value.
    raw["models"]["providers"]["lmstudio"]["api"] = ""
    raw["streaming"]["chat_mode"] = "yes"
    issues = _v(raw)
    paths = _paths(issues)
    assert "models.providers.lmstudio.api" in paths
    assert "streaming.chat_mode" in paths


def test_config_error_message_lists_all_paths():
    issues = [
        ConfigIssue("agents.main.model", "Unknown provider 'x'."),
        ConfigIssue("streaming.chat_mode", "Expected boolean."),
    ]
    err = ConfigError(issues)
    msg = str(err)
    assert "agents.main.model" in msg
    assert "streaming.chat_mode" in msg


def test_config_issue_frozen():
    issue = ConfigIssue("a.b", "bad")
    with pytest.raises(Exception):
        issue.path = "c.d"  # type: ignore[misc]


def test_model_id_not_validated_when_provider_unknown():
    """If the provider itself is unknown, skip model-id check (avoid redundant errors)."""
    raw = copy.deepcopy(_VALID)
    raw["agents"]["main"]["model"] = "badprovider/also-bad-model"
    issues = _v(raw)
    # Should report exactly one issue for the agent's model field.
    agent_issues = [i for i in issues if i.path == "agents.main.model"]
    assert len(agent_issues) == 1
    assert "Unknown provider" in agent_issues[0].message


# ---------------------------------------------------------------------------
# Malformed root and parent shapes
# ---------------------------------------------------------------------------


def test_root_list_returns_single_issue():
    """A JSON array at root must return a single ConfigIssue, not raise AttributeError."""
    issues = _v([])
    assert len(issues) == 1
    assert issues[0].path == "config.json"
    assert "JSON object" in issues[0].message


def test_models_section_string_returns_issue():
    """models: <string> must be reported as an issue, not crash with AttributeError."""
    raw = copy.deepcopy(_VALID)
    raw["models"] = "bad"
    issues = _v(raw)
    paths = _paths(issues)
    assert "models" in paths
