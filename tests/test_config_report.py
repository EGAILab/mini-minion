"""Tests for config_report.py: the `minion-assist config` CLI subcommand (MEM-GAP-019).

_render() is tested directly against small synthetic dataclasses (not the
real config.py ones) so the generic walking/redaction logic is covered
without depending on real config.json content — matching
tests/test_config_embeddings.py's "no real config.json on disk needed"
convention. format_config_report()/main() are covered separately by
monkeypatching specific minion_assist.config module attributes with
synthetic values containing a fake secret, then asserting that exact fake
value never appears in the rendered output — proving redaction against
the real code path without depending on (or risking leaking) any real
secret.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from minion_assist.config_report import (
    _collect_secrets,
    _redact_url,
    _render,
    collect_secret_values,
    format_config_report,
    main,
)


# ---------------------------------------------------------------------------
# _redact_url
# ---------------------------------------------------------------------------

def test_redact_url_masks_inline_credentials():
    assert _redact_url("postgresql://minion:secret@localhost:5433/db") == (
        "postgresql://***@localhost:5433/db"
    )


def test_redact_url_leaves_a_url_without_credentials_unchanged():
    assert _redact_url("postgresql://localhost:5433/db") == "postgresql://localhost:5433/db"


def test_redact_url_handles_a_plain_http_url_without_credentials():
    assert _redact_url("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/v1"


# ---------------------------------------------------------------------------
# _render — synthetic dataclasses, no dependency on real config.py content
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Inner:
    name: str
    api_key: str


@dataclass(frozen=True)
class _Outer:
    label: str
    inner: _Inner
    tags: tuple[str, ...] = ()
    extras: dict = field(default_factory=dict)
    child: object = None


def test_render_redacts_a_whole_sensitive_field():
    lines = _render(_Inner("prod", "sk-real-secret-value"), field_name=None, indent=0)
    text = "\n".join(lines)
    assert "sk-real-secret-value" not in text
    assert "api_key: ***" in text


def test_render_shows_not_set_for_an_empty_sensitive_field():
    lines = _render(_Inner("prod", ""), field_name=None, indent=0)
    assert "api_key: (not set)" in "\n".join(lines)


def test_render_recurses_into_a_nested_dataclass():
    outer = _Outer(label="x", inner=_Inner("prod", "secret"))
    lines = _render(outer, field_name="outer", indent=0)
    text = "\n".join(lines)
    assert "inner:" in text
    assert "name: 'prod'" in text
    assert "api_key: ***" in text  # redaction still applies at any depth


def test_render_shows_none_as_not_configured():
    lines = _render(None, field_name="route_prefix", indent=0)
    assert lines == ["route_prefix: (not configured)"]


def test_render_shows_empty_dict_as_none():
    outer = _Outer(label="x", inner=_Inner("p", "s"), extras={})
    lines = _render(outer.extras, field_name="extras", indent=0)
    assert lines == ["extras: (none)"]


def test_render_shows_empty_tuple_as_none():
    lines = _render((), field_name="tags", indent=0)
    assert lines == ["tags: (none)"]


def test_render_redacts_every_value_in_an_env_dict():
    env = {"API_KEY": "real-secret", "DEBUG": "true"}
    lines = _render(env, field_name="env", indent=0)
    text = "\n".join(lines)
    assert "real-secret" not in text
    assert "API_KEY: ***" in text
    assert "DEBUG: ***" in text  # every value masked, not just plausible-looking ones


def test_render_redacts_every_value_in_a_headers_dict():
    headers = {"Authorization": "Bearer real-token"}
    lines = _render(headers, field_name="headers", indent=0)
    assert "real-token" not in "\n".join(lines)


def test_render_does_not_redact_an_ordinary_dict():
    groups = {"!room:example.org": _Inner("room-agent", "secret")}
    lines = _render(groups, field_name="groups", indent=0)
    text = "\n".join(lines)
    assert "!room:example.org" in text
    assert "name: 'room-agent'" in text
    assert "api_key: ***" in text  # nested dataclass value still redacted on its own terms


def test_render_renders_a_list_of_dataclasses_by_index():
    servers = (_Inner("a", "s1"), _Inner("b", "s2"))
    lines = _render(servers, field_name="servers", indent=0)
    text = "\n".join(lines)
    assert "[0]:" in text
    assert "[1]:" in text
    assert "s1" not in text and "s2" not in text


def test_render_url_field_redacts_inline_credentials():
    lines = _render("postgresql://minion:secret@localhost/db", field_name="url", indent=0)
    assert "secret" not in "\n".join(lines)
    assert "***@localhost/db" in "\n".join(lines)


def test_render_plain_scalar_is_shown_as_is():
    lines = _render(42, field_name="tool_timeout", indent=0)
    assert lines == ["tool_timeout: 42"]


# ---------------------------------------------------------------------------
# format_config_report / main — against the real config module, with a
# monkeypatched section carrying a fake secret to prove real-path redaction
# ---------------------------------------------------------------------------

def test_format_config_report_includes_every_known_section():
    report = format_config_report()
    for section in (
        "agents:", "workspace:", "streaming:", "compaction:", "mcp:", "memory:",
        "bootstrap:", "channels:", "multi_agent:", "heartbeat:", "dreaming:",
        "memory_consolidation:", "memory_reconciliation:", "commitments:",
        "knowledge_digest:", "logging:", "codex:", "database:", "embeddings:", "voice:",
    ):
        assert section in report


def test_format_config_report_redacts_a_real_provider_api_key(monkeypatch):
    import minion_assist.config as _cfg
    from minion_assist.config import AgentModelConfig, ModelConfig, ProviderConfig

    fake_agents = {
        "main": AgentModelConfig(
            provider=ProviderConfig(
                name="fake", base_url="http://example.org", api_key="FAKE-SECRET-VALUE-12345",
                api="openai-completions",
            ),
            model=ModelConfig(id="fake-model", context_window=1000, max_output_tokens=100),
            route_prefix=None,
        )
    }
    monkeypatch.setattr(_cfg, "agents", fake_agents)

    report = format_config_report()

    assert "FAKE-SECRET-VALUE-12345" not in report
    assert "api_key: ***" in report


def test_format_config_report_redacts_a_real_database_url(monkeypatch):
    import minion_assist.config as _cfg
    from minion_assist.config import DatabaseConfig

    monkeypatch.setattr(
        _cfg, "database", DatabaseConfig(url="postgresql://realuser:realpass@dbhost:5432/prod")
    )

    report = format_config_report()

    assert "realpass" not in report
    assert "postgresql://***@dbhost:5432/prod" in report


def test_main_prints_the_report_and_returns_zero(capsys):
    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "agents:" in captured.out


def test_main_returns_2_and_does_not_raise_when_config_loading_fails(monkeypatch, capsys):
    import minion_assist.config_report as _report_mod

    def _boom():
        raise RuntimeError("config.json is broken")

    monkeypatch.setattr(_report_mod, "format_config_report", _boom)

    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Failed to load configuration" in captured.out


# ---------------------------------------------------------------------------
# _collect_secrets / collect_secret_values (MEM-GAP-017)
# ---------------------------------------------------------------------------

def test_collect_secrets_gathers_a_whole_sensitive_field():
    out: set[str] = set()
    _collect_secrets(_Inner("prod", "sk-real-secret-value"), field_name=None, out=out)
    assert out == {"sk-real-secret-value"}


def test_collect_secrets_skips_an_empty_sensitive_field():
    out: set[str] = set()
    _collect_secrets(_Inner("prod", ""), field_name=None, out=out)
    assert out == set()


def test_collect_secrets_recurses_into_a_nested_dataclass():
    outer = _Outer(label="x", inner=_Inner("prod", "nested-secret"))
    out: set[str] = set()
    _collect_secrets(outer, field_name=None, out=out)
    assert out == {"nested-secret"}


def test_collect_secrets_gathers_every_value_in_an_env_dict():
    out: set[str] = set()
    _collect_secrets({"API_KEY": "env-secret", "DEBUG": "true"}, field_name="env", out=out)
    assert out == {"env-secret", "true"}


def test_collect_secrets_extracts_a_url_credential_only():
    out: set[str] = set()
    _collect_secrets("postgresql://user:pw@localhost:5433/db", field_name="url", out=out)
    assert out == {"user:pw"}


def test_collect_secrets_ignores_a_url_without_credentials():
    out: set[str] = set()
    _collect_secrets("postgresql://localhost:5433/db", field_name="url", out=out)
    assert out == set()


def test_collect_secrets_gathers_from_a_list_of_dataclasses():
    servers = (_Inner("a", "secret-a"), _Inner("b", "secret-b"))
    out: set[str] = set()
    _collect_secrets(servers, field_name="servers", out=out)
    assert out == {"secret-a", "secret-b"}


def test_collect_secrets_ignores_plain_scalars_and_none():
    out: set[str] = set()
    _collect_secrets(42, field_name="tool_timeout", out=out)
    _collect_secrets(None, field_name="route_prefix", out=out)
    assert out == set()


def test_collect_secret_values_finds_a_real_provider_api_key(monkeypatch):
    import minion_assist.config as _cfg
    from minion_assist.config import AgentModelConfig, ModelConfig, ProviderConfig

    fake_agents = {
        "main": AgentModelConfig(
            provider=ProviderConfig(
                name="fake", base_url="http://example.org", api_key="FAKE-SECRET-VALUE-12345",
                api="openai-completions",
            ),
            model=ModelConfig(id="fake-model", context_window=1000, max_output_tokens=100),
            route_prefix=None,
        )
    }
    monkeypatch.setattr(_cfg, "agents", fake_agents)

    assert "FAKE-SECRET-VALUE-12345" in collect_secret_values()


def test_collect_secret_values_finds_a_real_database_url_credential(monkeypatch):
    import minion_assist.config as _cfg
    from minion_assist.config import DatabaseConfig

    monkeypatch.setattr(
        _cfg, "database", DatabaseConfig(url="postgresql://realuser:realpass@dbhost:5432/prod")
    )

    assert "realuser:realpass" in collect_secret_values()
