"""Base types for all tools: the Tool abstract class, ToolSchema, and the _within() path guard.

Every tool in mini-minion follows the same two-step contract:

1. **Declare** — describe the tool via a :class:`ToolSchema` so the LLM knows
   the tool exists and how to call it (name, description, parameter types).
2. **Execute** — implement the actual logic in :meth:`Tool.execute`, which
   receives the model's arguments as keyword arguments and returns a plain string.

Key concepts for new readers
-----------------------------

**Abstract Base Class (ABC)**
  :class:`Tool` is an *abstract base class* — a blueprint that cannot be
  instantiated directly. It forces every concrete tool class to implement
  both ``schema`` and ``execute``. If a subclass forgets either method, Python
  raises ``TypeError`` at instantiation time rather than silently producing a
  broken object that crashes mid-conversation.

  Think of it as a contract: "any class that wants to be a Tool *must* provide
  these two things."

**Why ToolSchema is a separate dataclass**
  The LLM never reads Python source code. :class:`ToolSchema` is the data that
  :class:`ToolRegistry` serialises into the JSON ``"tools"`` field that every
  API request includes. A clear ``description`` and self-documenting parameter
  names are what the model reads when deciding *whether* and *how* to call a
  tool — they are the tool's user-facing API.

**The _within() sandbox guard**
  :func:`_within` checks whether a file path falls inside the allowed workspace
  root directory. :class:`ReadTool`, :class:`WriteTool`, and :class:`GlobTool`
  call it before every file operation. If the resolved path escapes the root,
  the tool returns an error string instead of touching the file.

  The guard uses ``Path.resolve()`` before comparing, so ``../../``, symlinks,
  and mixed separators are all normalised first — there is no way to sneak a
  path past it with cleverly constructed strings.

Talks to
--------
- ``registry.py`` — calls ``tool.schema`` to build LLM definitions, and
  ``tool.execute(**args)`` to run a requested tool call.
- Every concrete tool file (``read.py``, ``write.py``, etc.) imports this
  module and subclasses :class:`Tool`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolSchema:
    """Metadata describing a tool to the LLM.

    The LLM reads these fields when deciding whether and how to call a tool.
    Writing clear, specific descriptions is important — vague descriptions
    lead to the model using tools incorrectly.

    Attributes:
        name (str): The tool's unique identifier, e.g. ``"read"`` or ``"bash"``.
            This is what the LLM sends back when it wants to call the tool.
        description (str): A plain-English explanation of what the tool does.
            Should be specific enough that the model knows exactly when to
            use this tool vs. another.
        parameters (dict): A JSON Schema object describing the tool's arguments.
            The LLM uses this to know what fields to fill in. Example:
            ``{"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}``
    """
    name: str
    description: str
    parameters: dict


class Tool(ABC):
    """Abstract base class for all tools in the system.

    Subclass this to create a new tool. You must implement both ``schema``
    (what the tool is) and ``execute`` (what the tool does).

    Example:
        class GreetTool(Tool):
            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(
                    name="greet",
                    description="Say hello to someone.",
                    parameters={
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                )

            def execute(self, **kwargs: object) -> str:
                return f"Hello, {kwargs['name']}!"
    """

    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """Return the tool's schema — name, description, and parameter spec.

        Returns:
            ToolSchema: Metadata the LLM uses to decide how to call this tool.
        """
        ...

    @abstractmethod
    def execute(self, **kwargs: object) -> str:
        """Run the tool with the given arguments and return the result as a string.

        Args:
            **kwargs: Keyword arguments matching the ``parameters`` schema.
                The registry calls this as ``tool.execute(**arguments_dict)``
                where ``arguments_dict`` was parsed from the LLM's tool call.

        Returns:
            str: The tool's output. Always a string, even on error.
                Return an error message string rather than raising an exception,
                so the agent can read and react to the failure.
        """
        ...


def _within(path: Path, root: Path) -> bool:
    """Return True if path is inside root (both paths are resolved before comparison)."""
    try:
        path.resolve().relative_to(root)
        return True
    except (ValueError, OSError):
        return False
