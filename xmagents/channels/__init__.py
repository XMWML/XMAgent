"""Channel adapters and provider-neutral message models."""

from .base import ChannelAdapter, ChannelError, ChannelHTTPError
from .models import Attachment, ChannelPeer, DeliveryResult, IncomingMessage, OutgoingMessage
from .telegram import TelegramAdapter, split_text
from .redaction import redact_mapping, redact_secret, redact_text

try:  # WeChat's optional QR rendering dependencies are imported lazily.
    from .wechat import QRLoginSession, WeChatAdapter, WeChatIlinkAdapter
except ImportError:  # pragma: no cover - allows partial installs to import Telegram
    QRLoginSession = None  # type: ignore[assignment]
    WeChatAdapter = None  # type: ignore[assignment]
    WeChatIlinkAdapter = None  # type: ignore[assignment]

__all__ = [
    "Attachment",
    "ChannelAdapter",
    "ChannelError",
    "ChannelHTTPError",
    "ChannelPeer",
    "DeliveryResult",
    "IncomingMessage",
    "OutgoingMessage",
    "QRLoginSession",
    "TelegramAdapter",
    "WeChatAdapter",
    "WeChatIlinkAdapter",
    "split_text",
    "redact_mapping",
    "redact_secret",
    "redact_text",
]
