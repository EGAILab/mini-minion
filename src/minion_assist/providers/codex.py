"""OpenAI Codex app-server provider.

Uses the Codex CLI binary in app-server mode, communicating via
JSON-RPC 2.0 over stdio.  Auth is injected by minion-assist via the
``account/login/start`` JSON-RPC call after ``initialize``.

Authenticate once with::

    codex-login

Tokens are stored in ``~/.minion-assist/codex-auth.json`` and
auto-refreshed before each session starts.  Because the subprocess is kept
running for the life of the bot, the token is also re-injected into it on a
background timer (see ``CodexProvider._start_auth_refresh_thread``) so a
long-running bot doesn't need to be restarted when the token expires
overnight — configurable via ``codex.auth_refresh_interval_seconds`` in
``config.json``.

The binary defaults to ``codex`` (looked up on PATH).  Override with the
``CODEX_BIN`` environment variable.

Protocol
--------
The binary is invoked as::

    codex app-server --listen stdio://

All JSON-RPC messages are newline-delimited on stdout/stdin.

Turn lifecycle (confirmed via live testing against codex-cli 0.142.3)
----------------------------------------------------------------------
1. ``initialize`` with ``capabilities: {experimentalApi: true}`` — required
   or ``account/login/start`` returns -32600.
2. ``thread/start`` — creates the thread and registers dynamic tool specs.
   The binary sits idle until ``turn/start`` is called.
3. ``turn/start`` — sends the user message and begins model inference.
   Used for EVERY turn including the first.
4. Server streams ``item/agentMessage/delta`` notifications as tokens arrive,
   then ``item/completed`` with the full assembled text.
5. ``turn/completed`` fires last — but ``params.turn.items`` is always ``[]``
   (the binary does not hydrate items in completion notifications).
   All response text must be collected from ``item/completed`` in step 4.

Dynamic tool bridge (openclaw approach)
----------------------------------------
All tools from the ToolRegistry are registered with Codex as dynamic tools
under a ``"minion-assist"`` namespace in the ``thread/start`` call.  When
Codex decides to invoke one, it sends an ``item/tool/call`` server request;
the reader thread executes the tool inline via ``registry.execute()`` and
replies with ``{contentItems: [...], success: bool}``.  This mirrors how
openclaw bridges its tools into the Codex app-server.

Approval requests (``item/commandExecution/requestApproval``,
``item/fileChange/requestApproval``, ``item/permissions/requestApproval``)
are handled separately — they reflect Codex's own built-in shell/file
capabilities and are NOT routed through the tool registry.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .base import LLMResponse, ToolCall
from ..llm_logger import log_request, log_response

if TYPE_CHECKING:
    from ..tools import ToolRegistry

# Namespace name sent to Codex for all minion-assist dynamic tools.
# Mirrors openclaw's CODEX_OPENCLAW_DYNAMIC_TOOL_NAMESPACE = "openclaw".
_DYNAMIC_TOOL_NAMESPACE = "minion-assist"

# Approval request methods — handled separately from dynamic tool calls.
_APPROVAL_METHODS = frozenset({
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
    "item/permissions/requestApproval",
})


def _codex_tool_name(registry_name: str) -> str:
    """Return a Codex-safe tool name for the given registry tool name.

    Codex reserves all names matching the ``mcp__*`` pattern because it uses
    that prefix for its own internal MCP server integration.  Stripping the
    leading ``mcp__`` yields a name Codex accepts while keeping the MCP server
    and tool name intact (e.g. ``playwright__browser_close``).

    Non-MCP tools (e.g. ``web_search``, ``bash``) pass through unchanged.
    """
    if registry_name.startswith("mcp__"):
        return registry_name[5:]  # strip leading "mcp__"
    return registry_name


def _build_dynamic_tool_specs(registry: "ToolRegistry") -> tuple[list[dict], dict[str, str]]:
    """Convert a ToolRegistry to a Codex dynamic tool namespace spec.

    Mirrors openclaw's ``createCodexDynamicToolSpecs()`` in dynamic-tools.ts:
    all tools are placed inside a single namespace with ``deferLoading: true``
    so Codex loads their schemas lazily rather than flooding its context window.

    The OpenAI format uses ``function.parameters`` for the JSON Schema;
    Codex uses ``inputSchema`` — same content, different key.

    MCP tool names (``mcp__server__tool``) are registered under sanitized names
    (``server__tool``) because Codex reserves the ``mcp__*`` namespace for its
    own MCP integration.  The returned ``name_map`` maps each Codex-side name
    back to the original registry name so ``_handle_tool_call`` can dispatch
    correctly when ``item/tool/call`` arrives.

    Returns:
        (specs, name_map) where ``specs`` is a list with one namespace entry
        (empty list if the registry has no tools), and ``name_map`` maps each
        Codex tool name to its registry name.
    """
    name_map: dict[str, str] = {}  # codex_name → registry_name
    namespace_tools: list[dict] = []
    for defn in registry.definitions:
        fn = defn.get("function") or {}
        registry_name: str = fn.get("name") or ""
        if not registry_name:
            continue
        codex_name = _codex_tool_name(registry_name)
        name_map[codex_name] = registry_name
        namespace_tools.append({
            "type": "function",
            "name": codex_name,
            "description": fn.get("description") or "",
            # Codex uses "inputSchema"; OpenAI uses "parameters" — same JSON Schema object.
            "inputSchema": fn.get("parameters") or {},
            # deferLoading=True: Codex loads the full schema only when it intends to call
            # the tool, keeping the initial context window lean.
            "deferLoading": True,
        })

    if not namespace_tools:
        return [], {}

    specs = [{
        "type": "namespace",
        "name": _DYNAMIC_TOOL_NAMESPACE,
        # Empty description: the namespace is an implementation detail; the
        # individual tool descriptions carry the semantically useful content.
        "description": "",
        "tools": namespace_tools,
    }]
    return specs, name_map


class _CodexRpcClient:
    """JSON-RPC 2.0 over a subprocess's stdio.

    Three message types from server:
    - response (has ``id``, no ``method``) → matches a pending client request.
    - notification (has ``method``, no ``id``) → broadcast to registered handlers.
    - server request (has both ``id`` and ``method``) → we reply immediately.

    Server requests fall into two categories:
    - ``item/tool/call`` — Codex calling a dynamic tool we registered;
      dispatched to ``registry.execute()`` inline on the reader thread.
    - approval requests (``item/commandExecution/requestApproval`` etc.) —
      Codex asking permission to use its own built-in bash/file capabilities;
      dispatched to the ``approve_command`` callback.
    """

    def __init__(
        self,
        command: list[str],
        registry: "ToolRegistry | None" = None,
        approve_command: Callable[[str, dict], str] | None = None,
    ) -> None:
        """Launch the Codex subprocess and start the background reader thread.

        Args:
            command: Full command-line list to launch the Codex binary, e.g.
                ``["codex", "app-server", "--listen", "stdio://"]``.
            registry: Tool registry whose tools can be called back by Codex
                via ``item/tool/call`` server requests.
            approve_command: Callback invoked when Codex requests permission to
                use one of its built-in capabilities (shell, file writes).
                Should return ``"approve"`` or ``"deny"``.
        """
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._id_lock = threading.Lock()
        self._next_id = 1
        self._pending: dict[int, dict] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._handlers: list[Callable[[dict], None]] = []
        # Tool registry for dynamic tool calls (item/tool/call).
        self._registry = registry
        # Maps Codex-side name → registry name.  Populated by CodexProvider.chat()
        # when thread/start registers dynamic tools.  Needed because mcp__* names
        # are stripped of their "mcp__" prefix before registration (Codex reserves
        # the mcp__ namespace for its own MCP server integration).
        self._tool_name_map: dict[str, str] = {}
        # Callback for approval requests (item/commandExecution/requestApproval etc.).
        # Returns "approve" or "deny"; None means auto-deny.
        self._approve_command = approve_command
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        """Background thread: read lines from the Codex process stdout and dispatch them."""
        assert self._proc.stdout
        for raw in self._proc.stdout:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg: dict = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_id = msg.get("id")
            method = msg.get("method")

            if method is not None and msg_id is not None:
                # Server → client request (dynamic tool call or approval).
                self._handle_server_request(msg_id, method, msg.get("params"))
            elif msg_id is not None:
                # Response to a client request.
                with self._pending_lock:
                    pending = self._pending.pop(msg_id, None)
                if pending:
                    pending["result"] = msg
                    pending["event"].set()
            elif method is not None:
                # Notification — broadcast to all registered handlers.
                for handler in list(self._handlers):
                    try:
                        handler(msg)
                    except Exception:
                        pass

    def _handle_server_request(self, req_id: int, method: str, _params: object) -> None:
        """Dispatch an incoming server request to the correct handler.

        Two distinct paths, mirroring openclaw's run-attempt.ts dispatch:
        - ``item/tool/call``  → dynamic tool execution via the registry.
        - approval methods   → built-in bash/file permission requests.
        """
        params = _params if isinstance(_params, dict) else {}

        if method == "item/tool/call":
            self._handle_tool_call(req_id, params)
        elif method in _APPROVAL_METHODS:
            self._handle_approval(req_id, method, params)
        # Unknown methods: no response sent; Codex will time out on its own.

    def _handle_tool_call(self, req_id: int, params: dict) -> None:
        """Execute a dynamic tool requested by Codex and reply with the result.

        Mirrors openclaw's ``CodexDynamicToolBridge.handleToolCall()``:
        look up the tool name in the registry, call ``execute()``, wrap the
        output in ``{contentItems: [...], success: bool}``.

        Executes inline on the reader thread — safe because Codex is blocked
        waiting for this response and will not send further messages until
        we reply.  (Equivalent to openclaw's ``await tool.execute()``.)
        """
        tool_name: str = params.get("tool") or ""
        arguments: dict = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}

        if not tool_name or self._registry is None:
            self._write_tool_response(req_id, f"Tool not available: {tool_name!r}", success=False)
            return

        # Translate Codex-side name back to the registry name.
        # "mcp__" was stripped when registering (Codex reserves the mcp__ namespace),
        # so "playwright__browser_close" maps back to "mcp__playwright__browser_close".
        registry_name = self._tool_name_map.get(tool_name, tool_name)

        # Check existence before dispatching — ToolRegistry.execute() never raises
        # (it returns error strings for both unknown tools and tool exceptions).
        # We need to signal success=False for unknown tools ourselves.
        if registry_name not in self._registry._tools:
            self._write_tool_response(req_id, f"Unknown tool: {tool_name!r}", success=False)
            return

        # registry.execute() runs pre/post hooks and catches tool exceptions,
        # returning them as error strings.  Always success=True from the protocol
        # perspective — the content carries the error message if something failed.
        output = self._registry.execute(registry_name, arguments)
        self._write_tool_response(req_id, output, success=True)

    def _write_tool_response(self, req_id: int, text: str, *, success: bool) -> None:
        """Send a ``{contentItems, success}`` response for an item/tool/call request."""
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "contentItems": [{"type": "inputText", "text": text}],
                "success": success,
            },
        }
        self._write_raw(json.dumps(resp))

    def _handle_approval(self, req_id: int, method: str, params: dict) -> None:
        """Handle a Codex built-in tool approval request.

        Reads ``params["available"]`` to find the valid decision values for this
        specific request (mirrors openclaw's ``hasAvailableDecision()`` check),
        then maps our internal "approve"/"deny" to a protocol-valid value.

        ``item/permissions/requestApproval`` uses a different response shape
        (``{permissions, scope}``) from the other approval methods
        (``{decision}``).
        """
        available: list[str] = params.get("available") or []

        if self._approve_command is not None:
            raw = self._approve_command(method, params)
        else:
            raw = "deny"

        approved = (raw == "approve")

        if method == "item/permissions/requestApproval":
            # Grant all requested permissions for the session; or deny with
            # an empty permissions dict scoped to this turn only.
            if approved:
                result: dict = {"permissions": params.get("permissions") or {}, "scope": "session"}
            else:
                result = {"permissions": {}, "scope": "turn"}
        else:
            # Command/file approval: pick the best valid decision value.
            if approved:
                # Prefer "acceptForSession" (persists for session, avoids repeated prompts).
                decision = (
                    "acceptForSession" if "acceptForSession" in available
                    else "accept" if "accept" in available
                    else (available[0] if available else "accept")
                )
            else:
                decision = (
                    "decline" if "decline" in available
                    else "cancel" if "cancel" in available
                    else "decline"
                )
            result = {"decision": decision}

        self._write_raw(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}))

    def _write_raw(self, line: str) -> None:
        """Write a newline-terminated JSON-RPC message to the Codex process stdin."""
        assert self._proc.stdin
        with self._write_lock:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()

    def request(self, method: str, params: dict | None = None, timeout: float = 60.0) -> dict:
        """Send a JSON-RPC request and wait synchronously for its response.

        Args:
            method: JSON-RPC method name, e.g. ``"initialize"`` or ``"turn/start"``.
            params: Optional parameters dict. Defaults to an empty dict.
            timeout: Seconds to wait for a response before raising ``TimeoutError``.

        Returns:
            The ``result`` field from the JSON-RPC response dict.

        Raises:
            TimeoutError: The response did not arrive within *timeout* seconds.
            RuntimeError: The server returned a JSON-RPC error object.
        """
        with self._id_lock:
            req_id = self._next_id
            self._next_id += 1
        event = threading.Event()
        slot: dict = {"result": None, "event": event}
        with self._pending_lock:
            self._pending[req_id] = slot
        self._write_raw(json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }))
        if not event.wait(timeout):
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise TimeoutError(f"Codex RPC {method!r} timed out after {timeout}s")
        result = slot["result"]
        if result is None or "error" in result:
            err = (result or {}).get("error", "no result")
            raise RuntimeError(f"Codex RPC {method!r} failed: {err}")
        return result.get("result") or {}

    def add_handler(self, handler: Callable[[dict], None]) -> None:
        """Register a notification handler to receive all server-sent notifications."""
        self._handlers.append(handler)

    def remove_handler(self, handler: Callable[[dict], None]) -> None:
        """Unregister a previously added notification handler (no-op if not found)."""
        try:
            self._handlers.remove(handler)
        except ValueError:
            pass

    def close(self) -> None:
        """Close stdin and terminate the Codex subprocess."""
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
        except Exception:
            pass


class CodexProvider:
    """LLM provider backed by the Codex app-server binary (OAuth subscription).

    Each provider instance maintains one Codex thread across multiple
    ``chat()`` calls so the binary owns conversation continuity.

    Not safe to share across independent conversations: ``_thread_id`` and
    ``_sent_count`` are single instance-level values, and the notification
    handler registered per ``chat()`` call does not filter by thread/turn ID.
    Two concurrent ``chat()`` calls on one shared instance (e.g. two Matrix
    rooms whose messages happen to be dispatched at the same time) can end
    up delivering one call's response to the other. Each independent
    conversation — one per Matrix room, for instance — needs its own
    ``CodexProvider`` instance (see ``minion.py``'s ``_matrix_session_factory``).

    All tools from the optional ``registry`` are registered with Codex as
    dynamic tools under the ``"minion-assist"`` namespace on the first call.
    Codex can then invoke them via ``item/tool/call`` server requests, which
    are executed inline and replied to before Codex continues.

    Args:
        codex_bin: Path to the ``codex`` binary.  Defaults to ``"codex"``
            (on PATH).  Overridden by ``CODEX_BIN``.
        model: Codex model ID (e.g. ``"gpt-5.5"``).  Pass an empty string
            to let the binary choose.
        turn_timeout: Seconds to wait for a turn to complete (default 120).
        registry: Tool registry whose tools are exposed to Codex as dynamic
            tools.  Pass the same registry used by runner.py so the tool
            execution logic is shared across all backends.
        approve_command: Called when Codex requests approval for its own
            built-in shell/file operations.  Receives ``(method, params)``
            and must return ``"approve"`` or ``"deny"``.  ``None`` = auto-deny.
        auth_refresh_interval: Seconds between background re-injections of the
            OAuth token into the (long-lived) Codex subprocess.  The binary is
            started once and kept alive for the life of the bot process, but
            it only ever receives the access token once (via
            ``account/login/start``) unless we push a new one to it.  Left
            unattended, the binary's own internal background jobs (e.g. its
            periodic "refresh available models" call) start failing with
            401 token_expired once that token expires — see
            ``_start_auth_refresh_thread``.  Default 300s (5 min), matching
            the refresh-before-expiry buffer already used in
            ``auth/codex_auth.py``.
    """

    def __init__(
        self,
        codex_bin: str = "codex",
        model: str = "",
        turn_timeout: float = 120.0,
        log_dir: Path | None = None,
        registry: "ToolRegistry | None" = None,
        approve_command: Callable[[str, dict], str] | None = None,
        auth_refresh_interval: float = 300.0,
    ) -> None:
        env_bin = os.environ.get("CODEX_BIN", "").strip()
        self._codex_bin = env_bin or codex_bin
        self._model = model
        self._turn_timeout = turn_timeout
        self._rpc: _CodexRpcClient | None = None
        self._thread_id: str | None = None
        self._sent_count: int = 0
        self._log_dir = log_dir
        # Tool registry exposed to Codex as dynamic tools (item/tool/call path).
        self._registry = registry
        # Callback for Codex built-in tool approval (item/commandExecution/requestApproval etc.).
        self._approve_command = approve_command
        # See _start_auth_refresh_thread — how often we re-push the OAuth
        # token into the running subprocess so it never goes stale overnight.
        self._auth_refresh_interval = auth_refresh_interval
        self._auth_refresh_thread: threading.Thread | None = None
        self._auth_refresh_stop: threading.Event | None = None

    def reset_session(self) -> None:
        """Forget the current Codex thread so the next chat() starts a fresh one.

        Called by AgentSession.reset() (/new command).  Without this, the Codex
        binary retains the full prior conversation in its thread even after the
        minion-assist message history is cleared, causing answers from the old
        context to bleed into the new conversation.
        """
        self._thread_id = None
        self._sent_count = 0

    def _get_rpc(self) -> _CodexRpcClient:
        """Return the active RPC client, launching the Codex subprocess if not yet started."""
        if self._rpc is None:
            # npm global installs on Windows create codex.cmd (a batch wrapper),
            # not codex.exe.  subprocess.Popen needs the full resolved path with
            # extension; shutil.which respects PATHEXT and returns it.
            resolved = shutil.which(self._codex_bin) or self._codex_bin
            cmd = [resolved, "app-server", "--listen", "stdio://"]
            try:
                self._rpc = _CodexRpcClient(
                    cmd,
                    registry=self._registry,
                    approve_command=self._approve_command,
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Codex binary not found: {self._codex_bin!r}\n"
                    "Install it with: npm install -g @openai/codex\n"
                    "Or set CODEX_BIN to its full path."
                ) from None
            # experimentalApi is required — without it the binary rejects
            # account/login/start with JSON-RPC error -32600.
            self._rpc.request("initialize", {
                "clientInfo": {"name": "minion-assist", "version": "0.1"},
                "capabilities": {"experimentalApi": True},
            }, timeout=30.0)
            self._inject_auth(self._rpc)
            self._start_auth_refresh_thread(self._rpc)
        return self._rpc

    def _start_auth_refresh_thread(self, rpc: _CodexRpcClient) -> None:
        """Start a background daemon thread that periodically re-injects auth.

        Why this is needed: the Codex subprocess is launched once and kept
        alive for the life of the bot process (see ``_get_rpc``), but the
        OAuth access token is only ever pushed into it once, at launch.  The
        binary itself runs its own internal background jobs against OpenAI's
        API (e.g. periodically refreshing its list of available models) using
        that same token.  Once the token expires — independent of whether
        minion-assist ever calls ``chat()`` — those internal jobs start
        failing with 401 token_expired and spam the log every few minutes
        until the whole bot is restarted.

        Restarting the bot "fixes" it only because a fresh subprocess gets a
        freshly (auto-)refreshed token via ``load_token()``.  This thread
        does the same re-injection on a timer, without needing a restart, by
        repeatedly calling ``_inject_auth`` (which itself calls
        ``codex_auth.load_token()`` — already auto-refreshing the token via
        its refresh_token when nearing expiry).
        """
        stop_event = threading.Event()
        self._auth_refresh_stop = stop_event
        thread = threading.Thread(
            target=self._auth_refresh_loop,
            args=(rpc, stop_event),
            name="codex-auth-refresh",
            daemon=True,
        )
        self._auth_refresh_thread = thread
        thread.start()

    def _auth_refresh_loop(self, rpc: _CodexRpcClient, stop_event: threading.Event) -> None:
        """Re-inject auth every ``_auth_refresh_interval`` seconds until stopped.

        ``Event.wait(timeout)`` returns ``True`` only when the event was set,
        so this doubles as both the sleep and the stop check: a normal tick
        waits out the full interval and refreshes; a stop request wakes the
        wait immediately and exits the loop without refreshing.
        """
        while not stop_event.wait(self._auth_refresh_interval):
            self._inject_auth(rpc)

    def _inject_auth(self, rpc: _CodexRpcClient) -> None:
        """Pass the stored OAuth token to the Codex binary via account/login/start."""
        from ..auth.codex_auth import load_token
        token = load_token()
        if not token:
            import sys
            print(
                "codex: no auth token found — run 'codex-login' to authenticate.\n"
                "       The binary will try its own stored credentials if any.",
                file=sys.stderr,
            )
            return
        try:
            rpc.request("account/login/start", {
                "type": "chatgptAuthTokens",
                "accessToken": token["access_token"],
                "chatgptAccountId": token.get("account_id", ""),
                "chatgptPlanType": None,
            }, timeout=30.0)
        except Exception as exc:
            import sys
            print(f"codex: account/login/start failed: {exc}", file=sys.stderr)

    @staticmethod
    def _extract_text(turn: dict) -> str:
        """Join non-empty ``text`` fields from all items in a completed turn."""
        parts: list[str] = []
        for item in turn.get("items") or []:
            if isinstance(item, dict):
                text = item.get("text") or ""
                if text:
                    parts.append(text)
        return "\n\n".join(parts)

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send the next turn to the Codex app-server.

        ``messages`` is the full conversation so far (Chat Completions format).
        Only messages added since the last call are treated as new input.
        The latest user message becomes the turn input; prior messages live
        in the Codex thread's own history.

        Tool definitions in ``tools`` are not forwarded per-turn — they are
        registered once via ``dynamicTools`` in ``thread/start`` using the
        registry provided at construction.  Codex calls them back via
        ``item/tool/call`` server requests during inference; the reader thread
        executes them inline and replies before Codex continues.

        Returns:
            LLMResponse with ``text`` populated and no tool_calls (tool
            execution happens inside Codex's inference loop, not in runner.py).
        """
        rpc = self._get_rpc()

        new_messages = messages[self._sent_count:]
        self._sent_count = len(messages)

        # Find the most recent user message in the new slice.
        user_text = ""
        for msg in reversed(new_messages):
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                user_text = content if isinstance(content, str) else str(content)
                break

        # Fallback: concatenate all new content if no user message found.
        if not user_text and new_messages:
            user_text = "\n".join(
                str(m.get("content") or "")
                for m in new_messages
                if m.get("content")
            )

        turn_input = [{"type": "text", "text": user_text}] if user_text else []
        model_kwargs = {"model": self._model} if self._model else {}

        # Register handlers BEFORE the RPC call to avoid missing early notifications.
        done = threading.Event()
        turn_holder: dict[str, dict] = {}
        agent_texts: list[str] = []

        def _on_notification(notification: dict) -> None:
            method = notification.get("method", "")
            params = notification.get("params")
            if not isinstance(params, dict):
                params = {}

            # item/completed carries the full assembled text for each output item.
            # turn/completed always has items: [] — the binary never hydrates items
            # in the completion notification, so we cannot use _extract_text there.
            if method == "item/completed":
                item = params.get("item") or {}
                if item.get("type") == "agentMessage":
                    t = (item.get("text") or "").strip()
                    if t:
                        agent_texts.append(t)

            if turn_holder:
                return
            turn = params.get("turn")
            status = turn.get("status", "") if isinstance(turn, dict) else ""
            if method == "turn/completed" or status in ("completed", "canceled", "error"):
                turn_holder["turn"] = turn or {}
                done.set()

        rpc.add_handler(_on_notification)
        try:
            if self._thread_id is None:
                # thread/start creates the thread and registers dynamic tool specs.
                # Dynamic tools are built here (not at construction) so MCP tools
                # that load asynchronously after provider creation are included.
                thread_params: dict = {"developerInstructions": system, **model_kwargs}
                if self._registry is not None:
                    dynamic_tools, name_map = _build_dynamic_tool_specs(self._registry)
                    if dynamic_tools:
                        thread_params["dynamicTools"] = dynamic_tools
                        rpc._tool_name_map = name_map
                resp = rpc.request("thread/start", thread_params, timeout=30.0)
                self._thread_id = (resp.get("thread") or {}).get("id") or ""

            if self._log_dir is not None:
                log_request(self._log_dir, "stdio://codex", {
                    "model": self._model or "codex-default",
                    "messages": [{"role": "system", "content": system}, *messages],
                })
            rpc.request("turn/start", {
                "threadId": self._thread_id,
                "input": turn_input,
                **model_kwargs,
            }, timeout=30.0)

            # Poll in 250 ms slices rather than a single Event.wait() call.
            # On Windows, a blocking Event.wait() swallows KeyboardInterrupt
            # because CPython only checks for signals between bytecode ops,
            # not inside a native wait.  Short slices let Ctrl+C break out.
            deadline = time.monotonic() + self._turn_timeout
            while not done.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Codex turn timed out after {self._turn_timeout}s"
                    )
                done.wait(timeout=min(0.25, remaining))
        finally:
            rpc.remove_handler(_on_notification)

        # Prefer text collected from item/completed; fall back to _extract_text
        # for future binary versions that may hydrate turn.items.
        text = " ".join(agent_texts) if agent_texts else self._extract_text(turn_holder.get("turn") or {})
        if self._log_dir is not None:
            log_response(self._log_dir, self._model or "codex-default", {"output": text})
        if on_token and text:
            on_token(text)
        return LLMResponse(text=text, tool_calls=[], finish_reason="stop")
