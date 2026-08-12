"""Provider-neutral channel adapter contract and shared errors."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from xmagents.models import Attachment, DeliveryResult, IncomingMessage


class ChannelError(RuntimeError):
    """Base exception raised for a provider/API failure."""


class ChannelHTTPError(ChannelError):
    """An HTTP or provider-level error, optionally retryable."""

    def __init__(self, message: str, *, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class ChannelAdapter(ABC):
    """Common lifecycle and send/poll surface used by the gateway.

    Adapters are intentionally stateful per configured account.  A caller may
    inject a client/session in the constructor of a concrete adapter for
    deterministic tests; no network connection is opened by this base class.
    """

    channel: str = "unknown"

    def __init__(self, account_id: str):
        self.account_id = str(account_id)
        self._started = False

    async def start(self) -> "ChannelAdapter":
        self._started = True
        return self

    async def stop(self) -> None:
        self._started = False

    async def close(self) -> None:
        await self.stop()

    @abstractmethod
    async def poll_once(self, cursor: str | None = None) -> list[IncomingMessage]:
        """Fetch and decode one provider poll response."""

    async def iter_messages(self, *, cursor: str | None = None, stop_event: asyncio.Event | None = None) -> AsyncIterator[IncomingMessage]:
        """Yield messages forever, stopping when ``stop_event`` is set."""

        current = cursor
        while stop_event is None or not stop_event.is_set():
            messages = await self.poll_once(current)
            for message in messages:
                current = str(message.metadata.get("next_cursor", current or ""))
                yield message
            if not messages:
                await asyncio.sleep(0.1)

    @abstractmethod
    async def send_text(self, peer_id: str, text: str, **kwargs: Any) -> DeliveryResult:
        """Send text to a peer."""

    @abstractmethod
    async def send_file(self, peer_id: str, path: str, **kwargs: Any) -> DeliveryResult:
        """Send a local file to a peer."""

    async def download_attachments(self, message: IncomingMessage, destination: str) -> list[Attachment]:
        """Download attachments and return updated models.

        Providers may override this when attachment metadata needs a second
        API call; the default makes an explicit capability error.
        """

        raise ChannelError(f"{self.channel} 不支持附件下载")

    async def run(self, on_message: Callable[[IncomingMessage], Awaitable[None]], *, stop_event: asyncio.Event | None = None, cursor: str | None = None) -> None:
        """Consume messages and dispatch them sequentially to ``on_message``."""

        async for message in self.iter_messages(cursor=cursor, stop_event=stop_event):
            await on_message(message)


__all__ = ["ChannelAdapter", "ChannelError", "ChannelHTTPError"]
