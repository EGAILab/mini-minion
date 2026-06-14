"""minion-assistant — a minimal multi-agent CLI assistant.

A small but complete example of building an LLM-powered agent from scratch using
nothing but plain Python, the OpenAI SDK, and flat files.

Two agents run in the same process:

- **Ada** (``"main"``) — a general assistant that handles any question.
- **Elizabeth** (``"researcher"``) — a research specialist activated via ``/research``.

Both agents run a **Think–Act–Observe** (TAO) loop: they think about the user's
request, optionally call tools (read files, run shell commands, search memory),
observe the results, and loop until they produce a final answer.

Package layout
--------------
- ``config``    — load and validate ``config.json`` + ``.env`` at startup.
- ``agents``    — agent identities, message routing, and the TAO execution loop.
- ``providers`` — adapters for OpenAI-compatible and Anthropic LLM APIs.
- ``tools``     — file-system, shell, and memory tools the agents can invoke.
- ``memory``    — short-term (JSONL transcripts) and long-term (Markdown notes).
- ``session``   — lightweight usage metadata: turn counts and timestamps.
- ``context``   — context-window management and conversation compaction.
- ``minion``    — the CLI entry point that wires every subsystem together.

**Start here:** ``src/minion_assistant/minion.py:main()`` is where execution begins.
See ``minion-assistant-documents/10-codebase-reading-guide.md`` for a recommended
reading order if you are new to the codebase.
"""
