"""Shared channel/domain models.

The channel implementations deliberately depend on these small dataclasses
instead of leaking Telegram or iLink payloads into the rest of the app.  All
objects are plain Python values so they are straightforward to persist as JSON
in the inbox/outbox tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


def _jsonable(value: Any) -> Any:
    """Return a conservative JSON-compatible representation of *value*."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return str(value)


@dataclass(slots=True)
class Attachment:
    """An inbound or outbound file.

    ``path`` is always local to the service.  ``source`` retains the provider
    identifiers (Telegram ``file_id`` or iLink media fields) needed to fetch
    the bytes later, but is never interpolated into a filesystem path.
    """

    path: str
    filename: str
    kind: str = "file"
    mime_type: str = "application/octet-stream"
    size: int = 0
    source: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "filename": self.filename,
            "kind": self.kind,
            "mime_type": self.mime_type,
            "size": self.size,
            "source": _jsonable(self.source),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Attachment":
        return cls(
            path=str(value.get("path", "")),
            filename=str(value.get("filename", "attachment")),
            kind=str(value.get("kind", "file")),
            mime_type=str(value.get("mime_type", "application/octet-stream")),
            size=int(value.get("size", 0) or 0),
            source=dict(value.get("source") or {}),
        )


@dataclass(slots=True)
class ChannelPeer:
    """A remote chat/user that can be bound to an Agent."""

    channel: str
    account_id: str
    external_id: str
    display_name: str = ""
    kind: str = "private"
    chat_id: str | None = None
    approved: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def peer_id(self) -> str:
        return self.chat_id or self.external_id

    @property
    def route_key(self) -> str:
        return f"{self.channel}:{self.account_id}:{self.peer_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "account_id": self.account_id,
            "external_id": self.external_id,
            "display_name": self.display_name,
            "kind": self.kind,
            "chat_id": self.chat_id,
            "approved": self.approved,
            "metadata": _jsonable(self.metadata),
        }


@dataclass(slots=True)
class IncomingMessage:
    """Provider-neutral inbound message delivered to an Agent mailbox."""

    channel: str
    account_id: str
    peer_id: str
    external_id: str
    text: str = ""
    sender_id: str | None = None
    sender_name: str = ""
    kind: str = "private"
    context_token: str | None = None
    attachments: list[Attachment] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    received_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def user_id(self) -> str:
        """Compatibility alias used by channel-specific integrations."""

        return self.sender_id or self.peer_id

    @property
    def route_key(self) -> str:
        return f"{self.channel}:{self.account_id}:{self.peer_id}"

    @property
    def message_id(self) -> str:
        return self.external_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "account_id": self.account_id,
            "peer_id": self.peer_id,
            "external_id": self.external_id,
            "text": self.text,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "kind": self.kind,
            "context_token": self.context_token,
            "attachments": [item.to_dict() for item in self.attachments],
            "raw": _jsonable(self.raw),
            "metadata": _jsonable(self.metadata),
            "received_at": self.received_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IncomingMessage":
        return cls(
            channel=str(value.get("channel", "")),
            account_id=str(value.get("account_id", "")),
            peer_id=str(value.get("peer_id", "")),
            external_id=str(value.get("external_id", value.get("message_id", ""))),
            text=str(value.get("text", "") or ""),
            sender_id=(str(value["sender_id"]) if value.get("sender_id") is not None else None),
            sender_name=str(value.get("sender_name", "") or ""),
            kind=str(value.get("kind", "private") or "private"),
            context_token=(str(value["context_token"]) if value.get("context_token") is not None else None),
            attachments=[Attachment.from_dict(item) for item in value.get("attachments", []) or []],
            raw=dict(value.get("raw") or {}),
            metadata=dict(value.get("metadata") or {}),
            received_at=str(value.get("received_at") or datetime.now(UTC).isoformat()),
        )


@dataclass(slots=True)
class OutgoingMessage:
    """Provider-neutral item placed in a channel outbox."""

    channel: str
    account_id: str
    peer_id: str
    text: str = ""
    attachments: list[Attachment] = field(default_factory=list)
    context_token: str | None = None
    reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "account_id": self.account_id,
            "peer_id": self.peer_id,
            "text": self.text,
            "attachments": [item.to_dict() for item in self.attachments],
            "context_token": self.context_token,
            "reply_to": self.reply_to,
            "metadata": _jsonable(self.metadata),
        }


@dataclass(slots=True)
class DeliveryResult:
    """Result of a send operation, retaining raw provider data for auditing."""

    ok: bool
    external_id: str | None = None
    raw: Any = None
    error: str | None = None
    attempts: int = 1
    status: int | None = None
    retry_after: float | None = None

    @classmethod
    def success(cls, raw: Any = None, external_id: str | None = None) -> "DeliveryResult":
        return cls(ok=True, external_id=external_id, raw=raw)

    @classmethod
    def failure(
        cls,
        error: str,
        raw: Any = None,
        attempts: int = 1,
        *,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> "DeliveryResult":
        return cls(ok=False, error=error, raw=raw, attempts=attempts, status=status, retry_after=retry_after)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "external_id": self.external_id,
            "raw": _jsonable(self.raw),
            "error": self.error,
            "attempts": self.attempts,
            "status": self.status,
            "retry_after": self.retry_after,
        }


__all__ = [
    "Attachment",
    "ChannelPeer",
    "IncomingMessage",
    "OutgoingMessage",
    "DeliveryResult",
]
