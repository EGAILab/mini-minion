"""Entry point and interactive chat interface."""

from .agents import AGENTS, resolve, run_turn
from .config import agents as agents_cfg, workspace
from .memory import LongTermMemory, ShortTermMemory
from .providers import create_provider
from .session import SessionStore
from .tools import default_registry


def main() -> None:
    """Start the interactive mini-minion chat session."""
    short_term = ShortTermMemory(workspace / "sessions")
    long_term = LongTermMemory(workspace / "memory")
    session_store = SessionStore(workspace / "sessions.json")

    tools = default_registry(long_term=long_term)

    providers = {
        agent_id: create_provider(
            api=cfg.provider.api,
            base_url=cfg.provider.base_url,
            api_key=cfg.provider.api_key,
            model=cfg.model.id,
        )
        for agent_id, cfg in agents_cfg.items()
    }

    histories: dict[str, list[dict]] = {
        agent_id: short_term.load(agent_id) for agent_id in AGENTS
    }
    for agent_id in AGENTS:
        session_store.get_or_create(agent_id)

    print("Mini-Minion ready. Type 'exit' to quit.")
    print("  /research <message>  → Elizabeth (research agent)")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        agent_id, message = resolve(user_input)
        agent = AGENTS[agent_id]
        cfg = agents_cfg[agent_id]

        histories[agent_id].append({"role": "user", "content": message})
        run_turn(
            providers[agent_id],
            agent.name,
            agent.soul,
            cfg.model.max_tokens,
            tools,
            histories[agent_id],
        )
        short_term.save(agent_id, histories[agent_id])
        session_store.touch(agent_id, increment_turns=True)


if __name__ == "__main__":
    main()
