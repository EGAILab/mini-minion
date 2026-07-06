"""OpenAI Codex app-server provider.

Uses the Codex CLI binary in app-server mode, communicating via
JSON-RPC 2.0 over stdio.  Auth is injected by minion-assist via the
``account/login/start`` JSON-RPC call after ``initialize``.

Authenticate once with::

    codex-login

Tokens are stored in ``~/.minion-assist/codex-auth.json`` and
auto-refreshed before each session starts.

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
2. ``thread/start`` — creates the thread only (no input, ``turns: []``).
   The binary does NOT start processing until ``turn/start`` is called.
3. ``turn/start`` — sends the user message and begins model inference.
   Used for EVERY turn including the first.
4. Server streams ``item/agentMessage/delta`` notifications as tokens arrive,
   then ``item/completed`` with the full assembled text.
5. ``turn/completed`` fires last — but ``params.turn.items`` is always ``[]``
   (the binary does not hydrate items in completion notifications).
   All response text must be collected from ``item/completed`` in step 4.
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

from .base import LLMResponse, ToolCall
from ..llm_logger import log_request, log_response


class _CodexRpcClient:
    """JSON-RPC 2.0 over a subprocess's stdio.

    Three message types from server:
    - response (has ``id``, no ``method``) → matches a pending client request.
    - notification (has ``method``, no ``id``) → broadcast to registered handlers.
    - server request (has both ``id`` and ``method``) → we reply immediately.
    """

    def __init__(
        self,
        command: list[str],
        approve_command: Callable[[str, dict], str] | None = None,
    ) -> None:
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
        self._approve_command = approve_command
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
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
                # Server → client request (e.g. item/tool/call for dynamic tools)
                self._handle_server_request(msg_id, method, msg.get("params"))
            elif msg_id is not None:
                # Response to a client request
                with self._pending_lock:
                    pending = self._pending.pop(msg_id, None)
                if pending:
                    pending["result"] = msg
                    pending["event"].set()
            elif method is not None:
                # Notification
                for handler in list(self._handlers):
                    try:
                        handler(msg)
                    except Exception:
                        pass

    def _handle_server_request(self, req_id: int, method: str, _params: object) -> None:
        # Codex sends server requests when it wants to execute a built-in tool
        # (bash command, file write, web search, etc.).  The binary expects a
        # response with a "decision" field using its own enum values:
        #   "accept" | "acceptForSession" | "decline" | "cancel" | ...
        # Our internal callback returns "approve" or "deny" for simplicity;
        # we map those to the protocol values here.
        params = _params if isinstance(_params, dict) else {}
        if self._approve_command is not None:
            raw = self._approve_command(method, params)
        else:
            raw = "deny"
        decision = "accept" if raw == "approve" else "decline"
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"decision": decision},
        }
        self._write_raw(json.dumps(resp))

    def _write_raw(self, line: str) -> None:
        assert self._proc.stdin
        with self._write_lock:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()

    def request(self, method: str, params: dict | None = None, timeout: float = 60.0) -> dict:
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
        self._handlers.append(handler)

    def remove_handler(self, handler: Callable[[dict], None]) -> None:
        try:
            self._handlers.remove(handler)
        except ValueError:
            pass

    def close(self) -> None:
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

    Args:
        codex_bin: Path to the ``codex`` binary.  Defaults to ``"codex"``
            (on PATH).  Overridden by ``CODEX_BIN``.
        model: Codex model ID (e.g. ``"gpt-5.5"``).  Pass an empty string
            to let the binary choose.
        turn_timeout: Seconds to wait for a turn to complete (default 120).
    """

    def __init__(
        self,
        codex_bin: str = "codex",
        model: str = "",
        turn_timeout: float = 120.0,
        log_dir: Path | None = None,
        approve_command: Callable[[str, dict], str] | None = None,
    ) -> None:
        env_bin = os.environ.get("CODEX_BIN", "").strip()
        self._codex_bin = env_bin or codex_bin
        self._model = model
        self._turn_timeout = turn_timeout
        self._rpc: _CodexRpcClient | None = None
        self._thread_id: str | None = None
        self._sent_count: int = 0
        self._log_dir = log_dir
        # Called when Codex requests approval to execute a built-in tool command.
        # Receives (method, params) and must return "approve" or "deny".
        # None means auto-deny (safe default for tests and non-interactive use).
        self._approve_command = approve_command

    def _get_rpc(self) -> _CodexRpcClient:
        if self._rpc is None:
            # npm global installs on Windows create codex.cmd (a batch wrapper),
            # not codex.exe.  subprocess.Popen needs the full resolved path with
            # extension; shutil.which respects PATHEXT and returns it.
            resolved = shutil.which(self._codex_bin) or self._codex_bin
            cmd = [resolved, "app-server", "--listen", "stdio://"]
            try:
                self._rpc = _CodexRpcClient(cmd, approve_command=self._approve_command)
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
        return self._rpc

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

        Tools registered via ``tools`` are not forwarded to the binary —
        Codex handles all tool execution internally using its built-in
        capabilities (bash, filesystem, web search).

        Returns:
            LLMResponse with ``text`` populated and no tool_calls (Codex
            handles tools internally and returns a final text answer).
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
                # thread/start creates the thread but does NOT start inference.
                # The binary sits idle until turn/start is called.
                resp = rpc.request("thread/start", {
                    "developerInstructions": system,
                    **model_kwargs,
                }, timeout=30.0)
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
