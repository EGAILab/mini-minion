# ADR 0005: No end-to-end encryption in the Matrix channel

## Status

Accepted.

## Context

The Matrix channel (`src/minion_assist/matrix/`) originally included E2E
encryption support: matrix-nio's Olm/Megolm crypto store wired up during
auth, automatic device-ID discovery via `whoami()`, device-key upload, and
an interactive SAS ("emoji") device-verification responder
(`MatrixVerificationHandler`) so a human could confirm Ada's device via
Element's "Verify User" flow.

In practice, none of it could produce working encrypted messaging:

- **matrix-nio has no cross-signing support at all** — no signing methods on
  `AsyncClient`, confirmed by inspecting the library directly. Cross-signing
  is what lets a device be trusted automatically once the account's identity
  is trusted, and is the mechanism modern Matrix clients rely on.
- **matrix-nio only implements the legacy direct-start verification flow**
  (`m.key.verification.start`), not the modern request/ready/start protocol
  (MSC2366) that current Element ("Verify User") actually uses. Since nio
  never receives a `request`, it never gets to `start`, so the interactive
  verification handler built for it could never fire from Element's actual
  UI — confirmed by testing against a real self-hosted Synapse instance.
  Manual/fingerprint verification (which would have sidestepped the missing
  protocol support) also wasn't exposed in either Element Web's or Element
  X's UI.
- With no way to verify Ada's device, clients correctly withheld room keys
  from it by design (`m.room_key.withheld`, which nio also doesn't surface
  — confirmed by inspecting the library — so this was invisible until
  checked directly against Synapse's access log). This happened
  identically across multiple independent clients and devices, ruling out
  a client-side bug or stale cache as the cause.

Separately, the actual threat model this channel operates under doesn't
call for E2E in the first place: it talks to a self-hosted homeserver where
the operator and the sole user are the same person. E2E's primary purpose —
preventing an untrusted server operator from reading message content — does
not apply when the user *is* the operator. Transport is already
TLS-protected regardless of whether Matrix-layer E2E is enabled. The
remaining benefit (defense in depth if the server itself is later
compromised) is real but secondary, and not worth the cost below.

Alternatives considered:

- **Migrate to `mautrix-python`**, which does support cross-signing. Ruled
  out for now: this channel's ~15 modules are built directly against
  matrix-nio's API shape (not behind an abstraction), so this would be a
  full rewrite, not a swap.
- **Implement MSC2366 verification from scratch** on top of nio's lower-level
  Sas/Olm primitives. Technically possible (nio exposes the primitives even
  though it doesn't wire up the modern flow) but a genuine multi-hour
  protocol-implementation project.
- **Switch protocols entirely** (Telegram, Signal, XMPP, IRC, and everything
  else in Wikipedia's
  [instant-messaging protocol comparison](https://en.wikipedia.org/wiki/Comparison_of_instant_messaging_protocols)
  were checked against "open protocol, fully self-hostable, no third-party
  service dependency"). Telegram and Signal fail that bar outright (neither
  is self-hostable independent of their respective companies' servers).
  XMPP is the only other protocol that passes, but OMEMO has the same
  device-trust/fingerprint-verification model as Matrix, so it would
  relocate this exact problem rather than solve it, at the cost of a full
  channel rewrite. Matrix remains the right protocol for the stated
  constraints; the issue was never protocol choice.

## Decision

The Matrix channel does not implement E2E encryption. `resolve_matrix_auth`
performs plain authentication only (access token / password / SSO), with no
crypto store, no device-key upload, and no verification handling. Rooms used
with this bot must be unencrypted — since Matrix rooms cannot be
un-encrypted once `m.room.encryption` is set, a room that was accidentally
encrypted needs to be replaced with a fresh unencrypted one, not fixed in
place.

If a genuine need for E2E arises later (e.g. the homeserver is no longer
self-hosted, or other people join the room), that's a new decision to
revisit against this ADR — not a partial patch on top of the current
library.

## Consequences

- No `crypto.py`, `verification.py`, `encryption`/`deviceId`/`verification`
  config fields, or related tests exist in the codebase — removed rather
  than left disabled-by-default, since dead crypto-adjacent code invites
  bit-rot and false confidence.
- `MatrixConfig.device_id` is retained: it's a general Matrix concept (every
  login has a device ID, encrypted or not), independent of this decision.
- `MatrixOutbound.resolve_or_create_dm()` (extracted from the exec-approval
  DM-resolution logic during this work) and the shared `ReactionEvent`
  callback in `monitor.py` are retained — both are used by exec approvals,
  which is unrelated to encryption.
- Anyone deploying this channel against a third-party or multi-tenant
  homeserver should treat message content as readable by that server's
  operator, and choose rooms/deployments accordingly.
