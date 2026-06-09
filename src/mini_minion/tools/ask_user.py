"""AskUserTool — pause the agent and request a human response.

During a long-running task, the agent may reach a decision point that only the
human operator can resolve: which branch to commit to, whether to overwrite a
file, what name to give a new function, etc.  Without this tool the agent must
either guess or abort.

Design decisions
----------------
- **Callback pattern**: The tool does not call ``input()`` directly.  Instead it
  accepts a ``prompt_fn`` callable at construction time.  The CLI passes its own
  ``input()``-backed function; a headless runner (tests, batch mode) passes
  ``None``.  This mirrors the pattern used by :class:`BashTool`'s ``confirm``
  callback and keeps the tool fully testable without a TTY.
- **Headless-safe**: When ``prompt_fn`` is ``None`` the tool returns an error
  string that tells the agent to continue without human input.  This is better
  than hanging or raising an exception, which would freeze the TAO loop.
- **Single string response**: The user's answer is returned as a plain string,
  which the LLM sees as the tool result and incorporates into its next thought.

Talks to
--------
- ``base.py`` — extends :class:`Tool`, returns :class:`ToolSchema`.
- ``registry.py`` / ``__init__.py`` — registered via ``default_registry()``
  when an ``ask_user_fn`` callable is provided.
- ``minion.py`` — the CLI passes ``_console_ask_user`` (backed by ``input()``)
  as the ``prompt_fn``.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import Tool, ToolSchema


class AskUserTool(Tool):
    """Tool that pauses the agent and prompts the human for input.

    The agent calls ``ask_user(question=...)`` whenever it needs information
    only the human can provide.  The tool blocks until the human responds,
    then returns the answer as a string.

    When ``prompt_fn`` is ``None`` (headless / batch mode), the tool returns
    an error string instructing the agent to proceed without human input.
    """

    def __init__(self, prompt_fn: Callable[[str], str] | None = None) -> None:
        # prompt_fn: called with the agent's question, returns the human's answer.
        # None = headless mode; the tool returns an error instead of blocking.
        self._prompt_fn = prompt_fn

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM."""
        return ToolSchema(
            name="ask_user",
            description=(
                "Ask the user a question and wait for their response. "
                "Use when you need clarification, a decision, or information "
                "that only the human can provide before you can proceed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the user.",
                    },
                },
                "required": ["question"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Ask the user the given question and return their response.

        Args:
            question (str): The question to present to the human operator.

        Returns:
            str: The human's response, or an error message when no prompt
                function is available (headless mode).
        """
        question = str(kwargs["question"])
        if self._prompt_fn is None:
            return (
                "Error: human input is not available in this context. "
                "Proceed with your best judgment or skip this step."
            )
        return self._prompt_fn(question)
