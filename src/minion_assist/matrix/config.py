"""Matrix channel configuration dataclasses.

Parsed from the ``channels.matrix`` section of ``config.json``.  All fields
mirror the openclaw MatrixConfigSchema shape so documentation and field names
are transferable between projects.

The top-level type is :class:`MatrixConfig`.  Nested types handle room-level
policy, DM policy, thread bindings, exec approvals, and bot-loop protection.

Usage::

    raw = config_json["channels"]["matrix"]
    cfg = MatrixConfig.from_dict(raw)
    print(cfg.homeserver, cfg.user_id)

Raises:
    ValueError: If required fields (``homeserver``, ``userId``) are missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MatrixRoomConfig:
    """Per-room overrides under ``channels.matrix.groups.<room_id>``."""
    agent: str = "main"
    enabled: bool = True
    require_mention: bool = False
    allow_bots: bool | str = False  # False, True, or "mentions"
    users: list[str] = field(default_factory=list)
    system_prompt: str | None = None
    # Emoji reaction level for inbound messages in this room.
    # "off" — never react; "all" — react to every message;
    # "mentions" — react only when the bot is mentioned.
    reaction_level: str = "off"

    @classmethod
    def from_dict(cls, raw: dict) -> "MatrixRoomConfig":
        """Parse a raw config dict into a MatrixRoomConfig."""
        return cls(
            agent=raw.get("agent", "main"),
            enabled=raw.get("enabled", True),
            require_mention=raw.get("requireMention", False),
            allow_bots=raw.get("allowBots", False),
            users=list(raw.get("users") or []),
            system_prompt=raw.get("systemPrompt"),
            reaction_level=raw.get("reactionLevel", "off"),
        )


@dataclass
class MatrixDmConfig:
    """DM-specific policy under ``channels.matrix.dm``."""
    enabled: bool = True
    policy: str = "allowlist"        # "allowlist", "pairing", "disabled"
    allow_from: list[str] = field(default_factory=list)
    session_scope: str = "per-user"  # "per-user" or "per-room"

    @classmethod
    def from_dict(cls, raw: dict) -> "MatrixDmConfig":
        """Parse a raw config dict into a MatrixDmConfig."""
        return cls(
            enabled=raw.get("enabled", True),
            policy=raw.get("policy", "allowlist"),
            allow_from=list(raw.get("allowFrom") or []),
            session_scope=raw.get("sessionScope", "per-user"),
        )


@dataclass
class MatrixExecApprovalsConfig:
    """Exec-approval settings under ``channels.matrix.execApprovals``."""
    enabled: bool = False
    approvers: list[str] = field(default_factory=list)
    target: str = "dm"  # "dm", "channel", "both"

    @classmethod
    def from_dict(cls, raw: dict) -> "MatrixExecApprovalsConfig":
        """Parse a raw config dict into a MatrixExecApprovalsConfig."""
        return cls(
            enabled=raw.get("enabled", False),
            approvers=list(raw.get("approvers") or []),
            target=raw.get("target", "dm"),
        )


@dataclass
class MatrixBotLoopConfig:
    """Bot-loop protection settings under ``channels.matrix.botLoopProtection``."""
    enabled: bool = True
    max_events_per_window: int = 10
    window_seconds: int = 60
    cooldown_seconds: int = 30

    @classmethod
    def from_dict(cls, raw: dict) -> "MatrixBotLoopConfig":
        """Parse a raw config dict into a MatrixBotLoopConfig."""
        return cls(
            enabled=raw.get("enabled", True),
            max_events_per_window=int(raw.get("maxEventsPerWindow", 10)),
            window_seconds=int(raw.get("windowSeconds", 60)),
            cooldown_seconds=int(raw.get("cooldownSeconds", 30)),
        )


@dataclass
class MatrixConfig:
    """Complete configuration for the Matrix channel.

    Attributes:
        homeserver:        Matrix homeserver URL, e.g. ``"https://matrix.example.org"``.
        user_id:           Bot's Matrix user ID, e.g. ``"@bot:example.org"``.
        access_token:      Pre-issued access token (preferred auth method).
        password:          Password for login-based auth (used when no access_token).
        device_id:         Existing device ID to resume a session.
        device_name:       Display name for new device registrations.
        auto_join:         Room invite policy: ``"always"``, ``"allowlist"``, or ``"off"``.
        auto_join_allowlist: Room IDs allowed when ``auto_join="allowlist"``.
        ack_reaction:      Emoji reaction posted when the agent starts processing.
        text_chunk_limit:  Max characters per outbound Matrix message.
        group_policy:      ``"open"`` (anyone in group triggers) or ``"allowlist"``.
        group_allow_from:  Global sender allowlist for group rooms.
        dm:                DM-specific policy config.
        groups:            Per-room config keyed by room ID or alias.
        exec_approvals:    Remote approval-via-DM config for tool execution.
        bot_loop:          Bot-loop rate-limit protection config.
        default_agent_id:  Agent to use when no per-room mapping matches.
        history_limit:     Number of past messages to inject as context per turn.
        media_max_mb:      Max media file size in MB to download/process.
    """
    homeserver: str
    user_id: str
    access_token: str | None = None
    password: str | None = None
    device_id: str | None = None
    device_name: str = "minion-assist"
    auto_join: str = "off"
    auto_join_allowlist: list[str] = field(default_factory=list)
    ack_reaction: str | None = "👀"
    text_chunk_limit: int = 4000
    group_policy: str = "open"
    group_allow_from: list[str] = field(default_factory=list)
    dm: MatrixDmConfig = field(default_factory=MatrixDmConfig)
    groups: dict[str, MatrixRoomConfig] = field(default_factory=dict)
    exec_approvals: MatrixExecApprovalsConfig = field(default_factory=MatrixExecApprovalsConfig)
    bot_loop: MatrixBotLoopConfig = field(default_factory=MatrixBotLoopConfig)
    default_agent_id: str = "main"
    history_limit: int = 0
    media_max_mb: int = 20

    @classmethod
    def from_dict(cls, raw: dict) -> "MatrixConfig":
        """Parse a ``channels.matrix`` dict from config.json into a MatrixConfig.

        Args:
            raw: The raw dict from config.json under ``channels.matrix``.

        Returns:
            MatrixConfig: Populated config instance.

        Raises:
            ValueError: If ``homeserver`` or ``userId`` is missing.
        """
        homeserver = raw.get("homeserver", "").strip()
        user_id = raw.get("userId", "").strip()
        if not homeserver:
            raise ValueError("channels.matrix.homeserver is required.")
        if not user_id:
            raise ValueError("channels.matrix.userId is required.")
        if not raw.get("accessToken") and not raw.get("password"):
            raise ValueError(
                "channels.matrix: at least one of 'accessToken' or 'password' is required."
            )

        groups: dict[str, MatrixRoomConfig] = {}
        for room_id, room_raw in (raw.get("groups") or {}).items():
            groups[room_id] = MatrixRoomConfig.from_dict(room_raw or {})

        dm_raw = raw.get("dm") or {}
        exec_raw = raw.get("execApprovals") or {}
        loop_raw = raw.get("botLoopProtection") or {}

        return cls(
            homeserver=homeserver,
            user_id=user_id,
            access_token=raw.get("accessToken") or None,
            password=raw.get("password") or None,
            device_id=raw.get("deviceId") or None,
            device_name=raw.get("deviceName", "minion-assist"),
            auto_join=raw.get("autoJoin", "off"),
            auto_join_allowlist=list(raw.get("autoJoinAllowlist") or []),
            ack_reaction=raw.get("ackReaction", "👀") or None,
            text_chunk_limit=int(raw.get("textChunkLimit", 4000)),
            group_policy=raw.get("groupPolicy", "open"),
            group_allow_from=list(raw.get("groupAllowFrom") or []),
            dm=MatrixDmConfig.from_dict(dm_raw),
            groups=groups,
            exec_approvals=MatrixExecApprovalsConfig.from_dict(exec_raw),
            bot_loop=MatrixBotLoopConfig.from_dict(loop_raw),
            default_agent_id=raw.get("defaultAgent", "main"),
            history_limit=int(raw.get("historyLimit", 0)),
            media_max_mb=int(raw.get("mediaMaxMb", 20)),
        )
