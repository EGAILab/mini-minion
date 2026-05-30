"""Tests for agent message routing.

The router is config-driven: routing rules come from the ``route_prefix`` field
in ``config.json`` agents, not from hard-coded Python. These tests cover both
the behavior expected with the real config (integration-style) and the generic
routing logic using monkeypatching to simulate arbitrary configs.
"""


import mini_minion.agents.router as router_mod
from mini_minion.agents.router import _build_routes, resolve

# ---------------------------------------------------------------------------
# Tests against the real config (validates the current config.json setup)
# ---------------------------------------------------------------------------

def test_default_routes_to_main():
    agent_id, message = resolve("hello world")
    assert agent_id == "main"
    assert message == "hello world"


def test_research_prefix_routes_to_researcher():
    agent_id, message = resolve("/research find something")
    assert agent_id == "researcher"
    assert message == "find something"


def test_research_prefix_strips_command_only():
    _, message = resolve("/research   extra spaces")
    assert message == "  extra spaces"


def test_research_without_space_routes_to_main():
    agent_id, _ = resolve("/researchfoo")
    assert agent_id == "main"


def test_research_alone_no_trailing_space_routes_to_main():
    """/research with no trailing space is NOT the research command."""
    agent_id, _ = resolve("/research")
    assert agent_id == "main"


def test_research_with_trailing_space_routes_to_researcher():
    """/research<space> with empty payload still routes to researcher."""
    agent_id, message = resolve("/research ")
    assert agent_id == "researcher"
    assert message == ""


def test_empty_message_routes_to_main():
    agent_id, message = resolve("")
    assert agent_id == "main"
    assert message == ""


# ---------------------------------------------------------------------------
# Tests that verify config-driven routing logic (monkeypatched routes)
# ---------------------------------------------------------------------------

def test_custom_prefix_routes_correctly(monkeypatch):
    """A custom route_prefix in config is honoured by resolve()."""
    monkeypatch.setattr(router_mod, "_ROUTES", [("/code ", "coder")])
    monkeypatch.setattr(router_mod, "_DEFAULT_AGENT", "main")

    agent_id, msg = resolve("/code write a sort function")
    assert agent_id == "coder"
    assert msg == "write a sort function"


def test_no_match_falls_back_to_default(monkeypatch):
    """Messages with no matching prefix fall back to the default agent."""
    monkeypatch.setattr(router_mod, "_ROUTES", [("/code ", "coder")])
    monkeypatch.setattr(router_mod, "_DEFAULT_AGENT", "main")

    agent_id, msg = resolve("hello there")
    assert agent_id == "main"
    assert msg == "hello there"


def test_multiple_prefixes_all_route_correctly(monkeypatch):
    """Multiple configured prefixes each route to their respective agent."""
    monkeypatch.setattr(router_mod, "_ROUTES", [
        ("/research ", "researcher"),
        ("/code ", "coder"),
    ])
    monkeypatch.setattr(router_mod, "_DEFAULT_AGENT", "main")

    assert resolve("/research find x") == ("researcher", "find x")
    assert resolve("/code write x") == ("coder", "write x")
    assert resolve("plain message") == ("main", "plain message")


def test_longest_prefix_matched_first(monkeypatch):
    """Longer prefixes take priority over shorter ones that share a start."""
    # "/research advanced" is longer than "/research" — it should match first.
    monkeypatch.setattr(router_mod, "_ROUTES", [
        ("/research advanced ", "expert"),
        ("/research ", "researcher"),
    ])
    monkeypatch.setattr(router_mod, "_DEFAULT_AGENT", "main")

    agent_id, msg = resolve("/research advanced deep dive")
    assert agent_id == "expert"
    assert msg == "deep dive"

    agent_id, msg = resolve("/research regular query")
    assert agent_id == "researcher"
    assert msg == "regular query"


# ---------------------------------------------------------------------------
# Tests for _build_routes() with mock agent configs
# ---------------------------------------------------------------------------

def test_build_routes_splits_prefixed_and_default(monkeypatch):
    """_build_routes correctly separates prefixed agents from the default."""
    from mini_minion.config import AgentModelConfig, ModelConfig, ProviderConfig

    dummy_provider = ProviderConfig(name="x", base_url="", api_key="", api="x")
    dummy_model = ModelConfig(id="x", context_window=8192, max_output_tokens=4096)

    fake_cfg = {
        "main":       AgentModelConfig(provider=dummy_provider, model=dummy_model, route_prefix=None),
        "researcher": AgentModelConfig(provider=dummy_provider, model=dummy_model, route_prefix="/research"),
        "coder":      AgentModelConfig(provider=dummy_provider, model=dummy_model, route_prefix="/code"),
    }
    monkeypatch.setattr(router_mod, "_agents_cfg", fake_cfg)

    routes, default = _build_routes()
    assert default == "main"
    assert ("/research ", "researcher") in routes
    assert ("/code ", "coder") in routes
    # No entry for the default agent in the routes list.
    assert all(agent_id != "main" for _, agent_id in routes)


def test_build_routes_appends_space_to_prefix(monkeypatch):
    """_build_routes appends a trailing space so /foo doesn't match /foobar."""
    from mini_minion.config import AgentModelConfig, ModelConfig, ProviderConfig

    dummy_provider = ProviderConfig(name="x", base_url="", api_key="", api="x")
    dummy_model = ModelConfig(id="x", context_window=8192, max_output_tokens=4096)

    fake_cfg = {
        "main":  AgentModelConfig(provider=dummy_provider, model=dummy_model, route_prefix=None),
        "coder": AgentModelConfig(provider=dummy_provider, model=dummy_model, route_prefix="/code"),
    }
    monkeypatch.setattr(router_mod, "_agents_cfg", fake_cfg)

    routes, _ = _build_routes()
    prefixes = [prefix for prefix, _ in routes]
    assert "/code " in prefixes     # trailing space present
    assert "/code" not in prefixes  # bare prefix is NOT in the table


def test_build_routes_sorts_longest_first(monkeypatch):
    """_build_routes sorts routes so the longest prefix is tried first."""
    from mini_minion.config import AgentModelConfig, ModelConfig, ProviderConfig

    dummy_provider = ProviderConfig(name="x", base_url="", api_key="", api="x")
    dummy_model = ModelConfig(id="x", context_window=8192, max_output_tokens=4096)

    fake_cfg = {
        "main":   AgentModelConfig(provider=dummy_provider, model=dummy_model, route_prefix=None),
        "short":  AgentModelConfig(provider=dummy_provider, model=dummy_model, route_prefix="/r"),
        "longer": AgentModelConfig(provider=dummy_provider, model=dummy_model, route_prefix="/research"),
    }
    monkeypatch.setattr(router_mod, "_agents_cfg", fake_cfg)

    routes, _ = _build_routes()
    prefix_lengths = [len(p) for p, _ in routes]
    assert prefix_lengths == sorted(prefix_lengths, reverse=True)


# ---------------------------------------------------------------------------
# Config integration: verify the real config.json has the right route_prefix
# ---------------------------------------------------------------------------

def test_researcher_route_prefix_in_config():
    """The researcher agent's route_prefix is read correctly from config.json."""
    from mini_minion.config import agents
    assert agents["researcher"].route_prefix == "/research"


def test_main_agent_has_no_route_prefix():
    """The main agent has no route_prefix — it is the default fallback."""
    from mini_minion.config import agents
    assert agents["main"].route_prefix is None
