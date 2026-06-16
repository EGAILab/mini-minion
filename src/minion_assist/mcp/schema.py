"""Schema and naming utilities for MCP tools.

Handles:
- Generating provider-safe tool names (mcp__server__tool)
- Normalizing JSON Schema for OpenAI-compatible providers
- Redacting secrets from status output
- Windows command normalization for stdio servers
"""
from __future__ import annotations
import re
import sys


# Characters allowed in provider tool names (OpenAI enforces [a-zA-Z0-9_-])
_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9_-]")

# Secret-looking key substrings to redact from log/status output.
# Checked case-insensitively against env var names and header keys.
_SECRET_KEYS = frozenset({
    "token", "secret", "password", "apikey", "api_key", "authorization",
    "bearer", "credential", "passwd",
})

# Windows commands that must be wrapped through cmd.exe /d /c to work correctly
# when launched as a subprocess. npx/npm on Windows are .cmd batch files, not
# .exe files, so Python's subprocess needs cmd.exe as the actual launcher.
_WINDOWS_WRAP_COMMANDS = frozenset({
    "npx", "npm", "pnpm", "yarn", "bunx",
})


def sanitize_tool_segment(value: str) -> str:
    """Replace characters not allowed in provider tool names with underscores.

    Provider-safe characters: A-Z, a-z, 0-9, underscore, hyphen.
    OpenAI rejects tool names with dots, slashes, spaces, etc.

    Examples:
        "my-server"  → "my-server"    (unchanged)
        "my.server"  → "my_server"
        "my server"  → "my_server"
    """
    sanitized = _SAFE_PATTERN.sub("_", value)
    # Tool names must start with alphanumeric (OpenAI rejects leading underscore/hyphen).
    if sanitized and not sanitized[0].isalnum():
        sanitized = "t" + sanitized
    return sanitized or "unknown"


def mcp_tool_name(server: str, tool: str) -> str:
    """Build the provider-safe minion-assist name for an MCP tool.

    Format: mcp__{sanitized_server}__{sanitized_tool}
    The double-underscore separator makes it visually obvious that this is
    an MCP tool and which server it came from.

    Examples:
        server="context7", tool="search" → "mcp__context7__search"
        server="my.server", tool="list-files" → "mcp__my_server__list_files"
    """
    return f"mcp__{sanitize_tool_segment(server)}__{sanitize_tool_segment(tool)}"


def normalize_schema_for_provider(schema: object) -> dict:
    """Normalize an MCP JSON Schema to be safe for OpenAI-compatible providers.

    Some MCP servers emit nullable schemas that OpenAI rejects:
      {"type": ["string", "null"]}         → {"type": "string", "nullable": true}
      {"anyOf": [{"type": "string"}, {"type": "null"}]}
                                           → {"type": "string", "nullable": true}

    If schema is not a dict, returns a bare object schema as fallback.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    result = dict(schema)

    # Normalize {"type": ["X", "null"]} → {"type": "X", "nullable": true}
    t = result.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        if len(non_null) == 1:
            result["type"] = non_null[0]
            result["nullable"] = True
        elif not non_null:
            result.pop("type", None)

    # Normalize anyOf/oneOf with a single non-null + null branch
    for key in ("anyOf", "oneOf"):
        variants = result.get(key)
        if isinstance(variants, list):
            non_null = [v for v in variants if v != {"type": "null"} and v.get("type") != "null"]
            null_present = len(non_null) < len(variants)
            if null_present and len(non_null) == 1 and isinstance(non_null[0], dict):
                result.pop(key)
                result.update(non_null[0])
                result["nullable"] = True

    # Recursively normalize nested property schemas
    if "properties" in result and isinstance(result["properties"], dict):
        result["properties"] = {
            k: normalize_schema_for_provider(v)
            for k, v in result["properties"].items()
        }

    return result


def redact_secret_text(text: str) -> str:
    """Replace likely secret values in a string with '[REDACTED]'.

    Used when logging or displaying MCP config details so tokens/passwords
    in headers or env vars are not printed to the terminal.

    Matches patterns like: "token=abc123", "Bearer abc123", "api_key: abc123"
    """
    if not text:
        return text
    for key in _SECRET_KEYS:
        # Match "key=value" or "key: value" (case-insensitive)
        text = re.sub(
            rf"(?i)({re.escape(key)}\s*[:=]\s*)\S+",
            r"\1[REDACTED]",
            text,
        )
    return text


def normalize_windows_command(command: str, args: list[str]) -> tuple[str, list[str]]:
    """Wrap Windows launcher commands through cmd.exe /d /c.

    On Windows, npx/npm/pnpm/yarn are .cmd batch files, not .exe files.
    Python's subprocess needs cmd.exe as the actual process to launch them.
    On non-Windows systems, returns (command, args) unchanged.

    Args:
        command: The MCP server's "command" field (e.g. "npx")
        args:    The MCP server's "args" list

    Returns:
        Tuple of (normalized_command, normalized_args)
    """
    if sys.platform != "win32":
        return command, args

    cmd_lower = command.lower()
    # Wrap known Windows-only launchers and .cmd/.bat scripts
    if cmd_lower in _WINDOWS_WRAP_COMMANDS or cmd_lower.endswith((".cmd", ".bat")):
        return "cmd.exe", ["/d", "/c", command] + list(args)

    return command, args
