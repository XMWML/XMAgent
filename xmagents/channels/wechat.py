"""WeChat iLink channel adapter.

The iLink API is intentionally kept behind this module.  The rest of the
application only sees :class:`~xmagents.models.IncomingMessage` and
:class:`~xmagents.models.DeliveryResult`, which makes this adapter usable in
offline tests with a small fake ``aiohttp``-compatible session.

The protocol is the one used by ``reference/wechat-api-reference.py``.  iLink
payloads have changed shape a few times, therefore decoding helpers accept the
known variants rather than assuming one exact response schema.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import mimetypes
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote, urljoin, urlsplit

from xmagents.models import Attachment, DeliveryResult, IncomingMessage
from xmagents.files import MAX_FILE_BYTES

from .base import ChannelAdapter, ChannelError, ChannelHTTPError
from .redaction import redact_text


BASE_URL = "https://ilinkai.weixin.qq.com"
CHANNEL_VERSION = "2.4.3"
ILINK_APP_ID = "bot"
ILINK_CLIENT_VERSION = str((2 << 16) | (4 << 8) | 3)
BOT_AGENT = "xmagents/0.1 (python)"

# iLink clients generally send fairly small text messages.  A lower limit than
# the provider's hard limit also leaves room for future metadata and keeps the
# individual messages useful on mobile clients.
DEFAULT_MAX_TEXT_LENGTH = 2000
DEFAULT_SEND_INTERVAL = 1.5
DOWNLOAD_CHUNK_BYTES = 64 * 1024


def _base_info() -> dict[str, str]:
    return {"channel_version": CHANNEL_VERSION, "bot_agent": BOT_AGENT}


def _headers(token: str | None = None, *, uin: int | str | None = None) -> dict[str, str]:
    """Build iLink headers.

    ``X-WECHAT-UIN`` is a base64-encoded decimal value, not the bot token.  A
    fresh value per request mirrors the reference client and avoids leaking a
    stable identifier into tests or logs.
    """

    value = str(uin if uin is not None else random.randint(0, 0xFFFFFFFF))
    result = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "X-WECHAT-UIN": base64.b64encode(value.encode("ascii")).decode("ascii"),
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": ILINK_CLIENT_VERSION,
    }
    if token:
        result["Authorization"] = f"Bearer {token}"
    return result


def _decode_media_key(value: Any) -> bytes | None:
    """Decode the 128-bit media key used by iLink.

    Servers have returned both standard base64 and hexadecimal keys.  Invalid
    values return ``None`` so unencrypted media can still be consumed; callers
    never silently use a key of the wrong length.
    """

    if not value:
        return None
    if isinstance(value, bytes):
        return value if len(value) == 16 else None
    text = str(value).strip()
    try:
        raw = base64.b64decode(text, validate=True)
        if len(raw) == 16:
            return raw
    except (ValueError, TypeError):
        pass
    try:
        raw = bytes.fromhex(text)
        return raw if len(raw) == 16 else None
    except ValueError:
        return None


def _aes_ecb_encrypt(data: bytes, key: bytes) -> bytes:
    """Encrypt bytes with AES-128-ECB and PKCS#7 padding."""

    if len(key) != 16:
        raise ValueError("iLink AES key must be 16 bytes")
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    pad_length = 16 - (len(data) % 16)
    padded = data + bytes([pad_length]) * pad_length
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(padded) + encryptor.finalize()


def _aes_ecb_decrypt(data: bytes, key: bytes) -> bytes:
    """Decrypt AES-128-ECB bytes and remove valid PKCS#7 padding."""

    if len(key) != 16:
        raise ValueError("iLink AES key must be 16 bytes")
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()
    plain = decryptor.update(data) + decryptor.finalize()
    if plain:
        pad_length = plain[-1]
        if 1 <= pad_length <= 16 and plain.endswith(bytes([pad_length]) * pad_length):
            plain = plain[:-pad_length]
    return plain


def _split_text(text: str, max_length: int = DEFAULT_MAX_TEXT_LENGTH) -> list[str]:
    """Split text into non-empty chunks without dropping any characters.

    Paragraph/sentence boundaries are preferred, but a hard split is used for
    an unusually long token.  This keeps messages ordered and avoids the
    provider rejecting a whole response because one line is too long.
    """

    if max_length < 1:
        raise ValueError("max_length must be positive")
    if not text:
        return [""]
    chunks: list[str] = []
    remaining = str(text)
    separators = ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ")
    while len(remaining) > max_length:
        window = remaining[: max_length + 1]
        cut = -1
        sep_len = 0
        for separator in separators:
            index = window.rfind(separator, 1, max_length + 1)
            if index > cut:
                cut = index
                sep_len = len(separator)
        if cut <= 0:
            cut = max_length
            sep_len = 0
        end = cut + sep_len
        chunks.append(remaining[:end])
        remaining = remaining[end:]
    if remaining or not chunks:
        chunks.append(remaining)
    return [chunk for chunk in chunks if chunk != ""]


def _safe_filename(value: Any, fallback: str) -> str:
    """Return a portable filename with path separators and control chars removed."""

    name = Path(str(value or "")).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return (name or fallback)[:240]


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return default


async def _response_json(response: Any) -> dict[str, Any]:
    """Read JSON from aiohttp or a lightweight test response."""

    value: Any = None
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            value = json_method()
            if hasattr(value, "__await__"):
                value = await value
        except Exception:
            value = None
    if value is None and isinstance(response, Mapping):
        value = response
    if value is None:
        text_method = getattr(response, "text", None)
        if callable(text_method):
            value = text_method()
            if hasattr(value, "__await__"):
                value = await value
        else:
            read_method = getattr(response, "read", None)
            if callable(read_method):
                value = read_method()
                if hasattr(value, "__await__"):
                    value = await value
            elif isinstance(response, (str, bytes, bytearray)):
                value = response.decode() if isinstance(response, (bytes, bytearray)) else response
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return dict(parsed) if isinstance(parsed, Mapping) else {}
        except (TypeError, ValueError):
            return {}
    return {}


async def _response_bytes(response: Any, *, max_bytes: int | None = None) -> bytes:
    """Read a response while enforcing an optional byte ceiling.

    iLink media URLs are provider data and their content length is not always
    present.  Reading their body with a single ``read()`` call would permit an
    unbounded allocation before the size validation runs, so real aiohttp
    streams are consumed in bounded chunks.  Tiny fake response objects used
    in offline tests retain the compatible ``read`` fallback.
    """

    def check_size(value: int) -> None:
        if max_bytes is not None and value > max_bytes:
            raise ChannelError(f"微信附件超过 {max_bytes // 1024 // 1024}MB 限制")

    if isinstance(response, Mapping):
        body = json.dumps(response, ensure_ascii=False).encode("utf-8")
        check_size(len(body))
        return body
    content = getattr(response, "content", None)
    iter_chunked = getattr(content, "iter_chunked", None)
    if callable(iter_chunked):
        chunks: list[bytes] = []
        received = 0
        async for chunk in iter_chunked(DOWNLOAD_CHUNK_BYTES):
            value = bytes(chunk or b"")
            received += len(value)
            check_size(received)
            chunks.append(value)
        return b"".join(chunks)
    read_method = getattr(response, "read", None)
    if callable(read_method):
        value = read_method()
        if hasattr(value, "__await__"):
            value = await value
        if isinstance(value, str):
            body = value.encode("utf-8")
        else:
            body = bytes(value or b"")
        check_size(len(body))
        return body
    content = getattr(response, "content", b"")
    if callable(content):
        content = content()
        if hasattr(content, "__await__"):
            content = await content
    if content:
        body = bytes(content)
        check_size(len(body))
        return body
    text_method = getattr(response, "text", None)
    if callable(text_method):
        value = text_method()
        if hasattr(value, "__await__"):
            value = await value
        if value:
            body = str(value).encode("utf-8")
            check_size(len(body))
            return body
    return b""


class _BufferedResponse:
    """Minimal response object retaining a body after an aiohttp context exits."""

    __slots__ = ("status", "headers", "_body")

    def __init__(self, status: int, headers: Mapping[str, Any] | None, body: bytes) -> None:
        self.status = status
        self.headers = dict(headers or {})
        self._body = bytes(body)

    async def read(self) -> bytes:
        return self._body

    async def text(self) -> str:
        return self._body.decode("utf-8", "replace")

    async def json(self) -> Any:
        try:
            value = json.loads(self._body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return value


def _status(response: Any) -> int:
    try:
        return int(getattr(response, "status", getattr(response, "status_code", 200)))
    except (TypeError, ValueError):
        return 200


def _retry_after(response: Any) -> float | None:
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After") or headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class WeChatIlinkAdapter(ChannelAdapter):
    """Async iLink adapter for one configured WeChat bot account."""

    channel = "wechat"
    # iLink returns one opaque get_updates_buf for the entire response.  It is
    # not an incremental per-message acknowledgement, so the service only
    # persists it after all messages from that response have been processed.
    cursor_checkpoint_per_message = False

    def __init__(
        self,
        account_id: str | None = None,
        bot_token: str | None = None,
        base_url: str = BASE_URL,
        *,
        token: str | None = None,
        session: Any = None,
        min_send_interval: float = DEFAULT_SEND_INTERVAL,
        max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
        cursor: str = "",
        request_timeout: float | None = None,
    ) -> None:
        # Accept ``WeChatIlinkAdapter("token")`` as a convenient shorthand,
        # while retaining the channel contract's account-first form.
        if bot_token is None and token is not None:
            bot_token = token
        elif bot_token is None and account_id:
            bot_token, account_id = account_id, "wechat"
        super().__init__(account_id or "wechat")
        self.bot_token = bot_token or ""
        self.base_url = str(base_url or BASE_URL).rstrip("/")
        self._session = session
        self._owns_session = False
        self.min_send_interval = max(0.0, float(min_send_interval))
        self.max_text_length = max(1, int(max_text_length))
        self.cursor = str(cursor or "")
        self.request_timeout = request_timeout
        self._send_lock = asyncio.Lock()
        self._next_send_at = 0.0
        self._typing_cache: dict[str, tuple[str, float]] = {}

    async def start(self) -> "WeChatIlinkAdapter":
        if self._session is None:
            try:
                import aiohttp
            except ImportError as error:  # pragma: no cover - optional runtime guard
                raise ChannelError("微信渠道需要安装 aiohttp") from error
            kwargs: dict[str, Any] = {}
            if self.request_timeout is not None:
                kwargs["timeout"] = aiohttp.ClientTimeout(total=self.request_timeout)
            self._session = aiohttp.ClientSession(**kwargs)
            self._owns_session = True
        await super().start()
        return self

    async def __aenter__(self) -> "WeChatIlinkAdapter":
        return await self.start()

    async def __aexit__(self, *_args: Any) -> None:
        await self.stop()

    async def stop(self) -> None:
        if self._owns_session and self._session is not None:
            close = getattr(self._session, "close", None)
            if callable(close):
                value = close()
                if hasattr(value, "__await__"):
                    await value
        self._session = None if self._owns_session else self._session
        self._owns_session = False
        await super().stop()

    async def _ensure_session(self) -> Any:
        if self._session is None:
            await self.start()
        if self._session is None:  # pragma: no cover - defensive
            raise ChannelError("微信 HTTP 会话不可用")
        return self._session

    def _url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        return urljoin(self.base_url + "/", path.lstrip("/"))

    def _media_headers(self, media_url: str) -> dict[str, str]:
        """Avoid forwarding the iLink bearer token to a foreign CDN host.

        iLink control-plane paths require the token.  A download URL can be a
        signed external CDN URL, where forwarding it would needlessly expose
        the bot credential.  Same-origin URLs retain the reference client's
        authenticated behaviour for providers that protect media there.
        """

        try:
            target = urlsplit(self._url(media_url))
            origin = urlsplit(self.base_url)
            target_port = target.port or (443 if target.scheme == "https" else 80)
            origin_port = origin.port or (443 if origin.scheme == "https" else 80)
            if (
                target.scheme.lower() == origin.scheme.lower()
                and target.hostname == origin.hostname
                and target_port == origin_port
            ):
                return _headers(self.bot_token)
        except (TypeError, ValueError):
            # A malformed URL is still handled by aiohttp below; it should
            # never receive authentication headers while doing so.
            pass
        return {"Accept": "*/*"}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        data: Any = None,
        headers: Mapping[str, str] | None = None,
        max_bytes: int | None = None,
    ) -> Any:
        session = await self._ensure_session()
        request = getattr(session, method.lower(), None)
        if not callable(request):
            raise ChannelError(f"HTTP session does not support {method.upper()}")
        request_headers = dict(headers or _headers(self.bot_token))
        kwargs: dict[str, Any] = {"headers": request_headers}
        if json_body is not None:
            kwargs["json"] = json_body
        if data is not None:
            kwargs["data"] = data
        try:
            response = request(self._url(path), **kwargs)
            if hasattr(response, "__aenter__"):
                async with response as entered:
                    status = _status(entered)
                    declared_size = getattr(entered, "headers", {}).get("Content-Length") if getattr(entered, "headers", None) else None
                    try:
                        declared = int(declared_size) if declared_size is not None else 0
                    except (TypeError, ValueError):
                        declared = 0
                    if max_bytes is not None and declared > max_bytes:
                        raise ChannelError(f"微信附件超过 {max_bytes // 1024 // 1024}MB 限制")
                    body = await _response_bytes(entered, max_bytes=max_bytes)
                    if max_bytes is not None and len(body) > max_bytes:
                        raise ChannelError(f"微信附件超过 {max_bytes // 1024 // 1024}MB 限制")
                    if not body:
                        payload = await _response_json(entered)
                        if payload:
                            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    buffered = _BufferedResponse(status, getattr(entered, "headers", {}), body)
                    if status >= 400:
                        raise ChannelHTTPError(
                            f"微信请求失败（HTTP {status}）",
                            status=status,
                            retry_after=_retry_after(entered),
                        )
                    return buffered
            if hasattr(response, "__await__"):
                response = await response
            status = _status(response)
            declared_size = getattr(response, "headers", {}).get("Content-Length") if getattr(response, "headers", None) else None
            try:
                declared = int(declared_size) if declared_size is not None else 0
            except (TypeError, ValueError):
                declared = 0
            if max_bytes is not None and declared > max_bytes:
                raise ChannelError(f"微信附件超过 {max_bytes // 1024 // 1024}MB 限制")
            body = await _response_bytes(response, max_bytes=max_bytes)
            if max_bytes is not None and len(body) > max_bytes:
                raise ChannelError(f"微信附件超过 {max_bytes // 1024 // 1024}MB 限制")
            if not body:
                payload = await _response_json(response)
                if payload:
                    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            buffered = _BufferedResponse(status, getattr(response, "headers", {}), body)
            if status >= 400:
                raise ChannelHTTPError(
                    f"微信请求失败（HTTP {status}）",
                    status=status,
                    retry_after=_retry_after(response),
                )
            return buffered
        except ChannelHTTPError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise ChannelError(f"微信网络请求失败：{redact_text(error, secrets=(self.bot_token,))}") from error

    async def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        response = await self._request("POST", path, json_body=dict(body))
        # A context manager response has already been exited by _request; fake
        # responses and non-context responses still need to be read here.
        payload = await _response_json(response)
        self._raise_provider_error(payload)
        return payload

    async def _get(self, path: str) -> dict[str, Any]:
        response = await self._request("GET", path)
        payload = await _response_json(response)
        self._raise_provider_error(payload)
        return payload

    @staticmethod
    def _raise_provider_error(payload: Mapping[str, Any]) -> None:
        """Promote iLink JSON failures into retry-aware channel errors."""

        code = _first(payload, "errcode", "err_code", "error_code", "retcode", "ret_code", default=0)
        try:
            failed = int(code or 0) != 0
        except (TypeError, ValueError):
            failed = bool(code and str(code).lower() not in {"ok", "success"})
        if not failed and payload.get("success") is False:
            failed = True
        if not failed:
            return
        message = str(_first(payload, "errmsg", "err_msg", "error", "message", "retmsg", default="微信 API 返回失败") or "微信 API 返回失败")
        retry_value = _first(payload, "retry_after", "retryAfter", "retry_interval", default=None)
        try:
            retry_after = float(retry_value) if retry_value is not None else None
        except (TypeError, ValueError):
            retry_after = None
        try:
            status = int(code)
        except (TypeError, ValueError):
            status = None
        raise ChannelHTTPError(redact_text(message), status=status, retry_after=retry_after)

    async def _put_bytes(self, url: str, data: bytes) -> Any:
        response = await self._request("PUT", url, data=data, headers={"Content-Type": "application/octet-stream"})
        return response

    async def getupdates(self, cursor: str | None = None) -> dict[str, Any]:
        return await self._post(
            "ilink/bot/getupdates",
            {"get_updates_buf": self.cursor if cursor is None else str(cursor), "base_info": _base_info()},
        )

    @staticmethod
    def _messages_from_response(data: Mapping[str, Any]) -> list[dict[str, Any]]:
        for key in ("msgs", "messages", "msg_list", "items", "updates"):
            value = data.get(key)
            if isinstance(value, list):
                return [_as_dict(item) for item in value if isinstance(item, Mapping)]
        value = data.get("msg") or data.get("message")
        if isinstance(value, Mapping):
            return [dict(value)]
        return []

    @staticmethod
    def _raw_message(value: Mapping[str, Any]) -> dict[str, Any]:
        # Some responses wrap each item in ``{"msg": {...}}``.
        msg = value.get("msg") or value.get("message")
        return dict(msg) if isinstance(msg, Mapping) else dict(value)

    @staticmethod
    def _message_id(msg: Mapping[str, Any], index: int) -> str:
        value = _first(msg, "message_id", "msg_id", "id", "client_id", "new_msg_id")
        if value is not None:
            return str(value)
        encoded = json.dumps(msg, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(f"{index}:{encoded}".encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _item_list(msg: Mapping[str, Any]) -> list[dict[str, Any]]:
        value = msg.get("item_list") or msg.get("items") or msg.get("itemList") or []
        return [_as_dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []

    @staticmethod
    def _attachment_stubs(items: list[dict[str, Any]]) -> list[Attachment]:
        """Expose provider media as placeholders so the service downloads it.

        iLink does not put the bytes directly in a getupdates response.  A
        non-empty attachment list is therefore a capability signal to the
        service, which later calls :meth:`download_attachments` into the
        bound Agent workspace.
        """

        result: list[Attachment] = []
        for index, item in enumerate(items):
            try:
                item_type = int(item.get("type", 0) or 0)
            except (TypeError, ValueError):
                continue
            if item_type not in {2, 4}:
                continue
            kind = "image" if item_type == 2 else "file"
            media = _as_dict(item.get("img_item") or item.get("file_item") or item.get("attach_item") or item.get("media"))
            filename = _safe_filename(
                _first(media, "file_name", "fileName", "filename", "name", "file_id", "fileId", default=f"attachment-{index}"),
                f"attachment-{index}",
            )
            mime_type = str(_first(media, "mime_type", "mimeType", "content_type", "contentType", default="") or "")
            if not mime_type:
                mime_type = mimetypes.guess_type(filename)[0] or ("image/*" if kind == "image" else "application/octet-stream")
            try:
                size = int(_first(media, "file_size", "fileSize", "size", default=0) or 0)
            except (TypeError, ValueError):
                size = 0
            result.append(Attachment(path="", filename=filename, kind=kind, mime_type=mime_type, size=size, source=media))
        return result

    @classmethod
    def decode_message(cls, value: Mapping[str, Any], *, account_id: str, next_cursor: str = "") -> IncomingMessage:
        msg = cls._raw_message(value)
        items = cls._item_list(msg)
        text = str(_first(msg, "text", "content", default="") or "")
        if not text:
            for item in items:
                try:
                    item_type = int(item.get("type", 0) or 0)
                except (TypeError, ValueError):
                    item_type = 0
                if item_type == 1:
                    text_item = _as_dict(item.get("text_item") or item.get("textItem"))
                    text = str(_first(text_item, "text", "content", default="") or "")
                    if text:
                        break
        sender_id = str(_first(msg, "from_user_id", "fromUserId", "from", "sender_id", "user_id", default="") or "")
        # ``to_user_id`` is normally the bot account in inbound iLink
        # messages.  Route replies to the sender first; group/forwarded
        # payloads may provide an explicit chat id which takes precedence.
        peer_id = str(_first(msg, "chat_id", "conversation_id", "peer_id", default=sender_id) or sender_id)
        if not peer_id:
            peer_id = str(_first(msg, "to_user_id", "toUserId", default="") or "")
        context = _first(msg, "context_token", "contextToken", "context", default=None)
        kind = str(_first(msg, "chat_type", "conversation_type", "kind", default="private") or "private")
        metadata = {
            "next_cursor": next_cursor,
            "message_type": msg.get("message_type", msg.get("messageType", 1)),
            "item_list": items,
        }
        return IncomingMessage(
            channel="wechat",
            account_id=account_id,
            peer_id=peer_id,
            external_id=cls._message_id(msg, 0),
            text=text,
            sender_id=sender_id or None,
            sender_name=str(_first(msg, "sender_name", "from_user_name", "nickname", default="") or ""),
            kind=kind,
            context_token=str(context) if context is not None else None,
            attachments=cls._attachment_stubs(items),
            raw=dict(msg),
            metadata=metadata,
        )

    @classmethod
    def decode_messages(cls, data: Mapping[str, Any], *, account_id: str, next_cursor: str = "") -> list[IncomingMessage]:
        """Decode every message in one raw ``getupdates`` response."""

        return [
            cls.decode_message(item, account_id=account_id, next_cursor=next_cursor)
            for item in cls._messages_from_response(data)
            if cls._raw_message(item)
        ]

    async def poll_once(self, cursor: str | None = None) -> list[IncomingMessage]:
        data = await self.getupdates(cursor)
        next_cursor = str(_first(data, "get_updates_buf", "next_cursor", "cursor", "buf", default=self.cursor) or "")
        self.cursor = next_cursor
        result: list[IncomingMessage] = []
        for index, raw in enumerate(self._messages_from_response(data)):
            msg = self._raw_message(raw)
            if not msg:
                continue
            decoded = self.decode_message(msg, account_id=self.account_id, next_cursor=next_cursor)
            # Include the batch index in the deterministic fallback so two
            # identical provider payloads in one response remain distinct.
            decoded.external_id = self._message_id(msg, index)
            result.append(decoded)
        return result

    async def _throttle(self) -> None:
        async with self._send_lock:
            now = time.monotonic()
            delay = self._next_send_at - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_send_at = time.monotonic() + self.min_send_interval

    async def _send_message(self, peer_id: str, context_token: str | None, item_list: list[dict[str, Any]]) -> dict[str, Any]:
        await self._throttle()
        client_id = f"xmagents-wechat-{random.randint(0, 0xFFFFFFFF):08x}"
        return await self._post(
            "ilink/bot/sendmessage",
            {
                "msg": {
                    "from_user_id": "",
                    "to_user_id": str(peer_id),
                    "client_id": client_id,
                    "message_type": 2,
                    "message_state": 2,
                    "context_token": context_token or "",
                    "item_list": item_list,
                },
                "base_info": _base_info(),
            },
        )

    async def send_text(self, peer_id: str, text: str, **kwargs: Any) -> DeliveryResult:
        context_token = kwargs.get("context_token") or kwargs.get("context") or ""
        chunks = _split_text(str(text), int(kwargs.get("max_length", self.max_text_length)))
        raw_results: list[dict[str, Any]] = []
        try:
            for chunk in chunks:
                raw_results.append(await self._send_message(str(peer_id), str(context_token), [{"type": 1, "text_item": {"text": chunk}}]))
        except ChannelError as error:
            return DeliveryResult.failure(
                str(error), raw=raw_results, attempts=len(raw_results) + 1,
                status=getattr(error, "status", None), retry_after=getattr(error, "retry_after", None),
            )
        external_id = None
        if raw_results:
            external_id = str(_first(raw_results[-1], "message_id", "msg_id", "client_id", "id", default="")) or None
        return DeliveryResult.success(raw=raw_results if len(raw_results) > 1 else (raw_results[0] if raw_results else {}), external_id=external_id)

    # Reference-client spellings retained for integrations that need the raw
    # iLink operation.  New gateway code should use ``send_text``/``send_file``
    # so delivery is represented by a provider-neutral result model.
    async def sendmessage(self, user_id: str, context_token: str, text: str) -> dict[str, Any]:
        return await self._send_message(str(user_id), str(context_token or ""), [{"type": 1, "text_item": {"text": str(text)}}])

    async def _upload_cdn(self, encrypted: bytes) -> dict[str, Any]:
        info = await self._post("ilink/bot/getuploadurl", {"file_size": len(encrypted), "base_info": _base_info()})
        upload_url = _first(info, "upload_url", "uploadUrl", "url")
        if not upload_url:
            raise ChannelError("微信 CDN 未返回上传地址")
        response = await self._put_bytes(str(upload_url), encrypted)
        status = _status(response)
        if status not in (200, 201, 202, 204):
            raise ChannelHTTPError(f"微信 CDN 上传失败（HTTP {status}）", status=status, retry_after=_retry_after(response))
        return {key: value for key, value in info.items() if key not in {"upload_url", "uploadUrl", "url"}}

    async def send_file(self, peer_id: str, path: str, **kwargs: Any) -> DeliveryResult:
        file_path = Path(path)
        context_token = kwargs.get("context_token") or kwargs.get("context") or ""
        try:
            if not file_path.is_file():
                return DeliveryResult.failure(f"文件不存在: {path}")
            declared_size = file_path.stat().st_size
            if declared_size > MAX_FILE_BYTES:
                return DeliveryResult.failure(f"文件超过 {MAX_FILE_BYTES // 1024 // 1024}MB 限制")
            data = file_path.read_bytes()
            if len(data) > MAX_FILE_BYTES:
                return DeliveryResult.failure(f"文件超过 {MAX_FILE_BYTES // 1024 // 1024}MB 限制")
            name = _safe_filename(file_path.name, "attachment")
            key = os.urandom(16)
            encrypted = _aes_ecb_encrypt(data, key)
            cdn_info = await self._upload_cdn(encrypted)
            mime = str(kwargs.get("mime_type") or mimetypes.guess_type(name)[0] or "application/octet-stream")
            is_image = bool(kwargs.get("kind") == "image" or mime.startswith("image/"))
            media = {
                **cdn_info,
                "aes_key": base64.b64encode(key).decode("ascii"),
            }
            if is_image:
                item = {"type": 2, "img_item": media}
            else:
                media.update({"file_name": name, "file_size": len(data)})
                item = {"type": 4, "attach_item": media}
            raw = await self._send_message(str(peer_id), str(context_token), [item])
            external_id = str(_first(raw, "message_id", "msg_id", "client_id", "id", default="")) or None
            return DeliveryResult.success(raw=raw, external_id=external_id)
        except (OSError, ValueError, ChannelError) as error:
            return DeliveryResult.failure(
                str(error), status=getattr(error, "status", None), retry_after=getattr(error, "retry_after", None),
            )

    async def sendfile(self, user_id: str, context_token: str, file_path: str) -> dict[str, Any]:
        result = await self.send_file(user_id, file_path, context_token=context_token)
        if not result.ok:
            raise ChannelError(result.error or "微信文件发送失败")
        return result.raw if isinstance(result.raw, Mapping) else {"result": result.raw}

    async def sendimage(self, user_id: str, context_token: str, image_path: str) -> dict[str, Any]:
        result = await self.send_file(user_id, image_path, context_token=context_token, kind="image")
        if not result.ok:
            raise ChannelError(result.error or "微信图片发送失败")
        return result.raw if isinstance(result.raw, Mapping) else {"result": result.raw}

    async def reply(self, user_id: str, context_token: str, text: str, *, show_typing: bool = True) -> dict[str, Any]:
        ticket = None
        if show_typing:
            config = await self.getconfig(user_id)
            ticket = _first(config, "typing_ticket", "typingTicket")
            if ticket:
                await self.sendtyping(user_id, str(ticket), 1)
        try:
            return await self.sendmessage(user_id, context_token, text)
        finally:
            if show_typing and ticket:
                try:
                    await self.sendtyping(user_id, str(ticket), 2)
                except ChannelError:
                    # A typing notification is best-effort and should not turn
                    # a successfully delivered message into a failed reply.
                    pass

    async def getconfig(self, user_id: str) -> dict[str, Any]:
        cached = self._typing_cache.get(str(user_id))
        if cached and time.time() < cached[1]:
            return {"typing_ticket": cached[0]}
        result = await self._post("ilink/bot/getconfig", {"to_user_id": str(user_id), "base_info": _base_info()})
        ticket = _first(result, "typing_ticket", "typingTicket")
        if ticket:
            self._typing_cache[str(user_id)] = (str(ticket), time.time() + 23 * 3600)
        return result

    async def sendtyping(self, user_id: str, ticket: str, status: int) -> dict[str, Any]:
        return await self._post(
            "ilink/bot/sendtyping",
            {"to_user_id": str(user_id), "typing_ticket": ticket, "status": int(status), "base_info": _base_info()},
        )

    async def download_attachments(self, message: IncomingMessage | Any, destination: str, account_id: str | None = None) -> list[Attachment]:
        """Download and decrypt image/file items into an isolated directory."""

        raw = _as_dict(getattr(message, "raw", None))
        items = getattr(message, "metadata", {}).get("item_list") if hasattr(message, "metadata") else None
        if not isinstance(items, list):
            items = self._item_list(raw)
        if not items:
            items = list(getattr(message, "item_list", []) or [])
        user_id = str(
            getattr(message, "sender_id", None)
            or getattr(message, "user_id", None)
            or getattr(message, "from_user", None)
            or getattr(message, "peer_id", "unknown")
        )
        effective_account_id = str(account_id or self.account_id)
        account_segment = hashlib.sha256(effective_account_id.encode("utf-8")).hexdigest()[:16]
        user_segment = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
        root = Path(destination).resolve() / account_segment / user_segment
        root.mkdir(parents=True, exist_ok=True)
        result: list[Attachment] = []
        used_names: set[str] = set()
        for index, value in enumerate(items):
            item = _as_dict(value)
            try:
                item_type = int(item.get("type", 0) or 0)
            except (TypeError, ValueError):
                item_type = 0
            kind = "image" if item_type == 2 else "file" if item_type == 4 else ""
            if not kind:
                continue
            media = _as_dict(item.get("img_item") or item.get("file_item") or item.get("attach_item") or item.get("media"))
            url = _first(media, "download_url", "downloadUrl", "url") or _first(item, "download_url", "downloadUrl", "url")
            if not url:
                request_body = dict(media)
                request_body.setdefault("file_id", _first(media, "file_id", "fileId", "id") or _first(item, "file_id", "fileId", "id"))
                request_body["base_info"] = _base_info()
                info = await self._post("ilink/bot/getdownloadurl", request_body)
                media.update(info)
                url = _first(info, "download_url", "downloadUrl", "url")
            if not url:
                raise ChannelError("微信媒体未提供可下载地址")
            response = await self._request(
                "GET",
                str(url),
                headers=self._media_headers(str(url)),
                max_bytes=MAX_FILE_BYTES,
            )
            data = await _response_bytes(response)
            headers = getattr(response, "headers", {}) or {}
            content_type = str(headers.get("Content-Type", "")).split(";", 1)[0].strip()
            key = _decode_media_key(_first(media, "aes_key", "aesKey", "encrypt_key", "encryptKey"))
            if key:
                data = _aes_ecb_decrypt(data, key)
            if len(data) > MAX_FILE_BYTES:
                raise ChannelError(f"微信附件解密后超过 {MAX_FILE_BYTES // 1024 // 1024}MB 限制")
            raw_name = _first(media, "file_name", "fileName", "filename", "name", default=f"attachment-{index}")
            filename = _safe_filename(raw_name, f"attachment-{index}")
            if "." not in filename and content_type:
                extension = mimetypes.guess_extension(content_type) or ("." + content_type.split("/", 1)[-1])
                filename += extension
            original = filename
            suffix = 1
            while filename in used_names or (root / filename).exists():
                stem, extension = os.path.splitext(original)
                filename = f"{stem}-{suffix}{extension}"
                suffix += 1
            used_names.add(filename)
            path = root / filename
            path.write_bytes(data)
            result.append(
                Attachment(
                    path=str(path),
                    filename=filename,
                    kind=kind,
                    mime_type=content_type or ("image/*" if kind == "image" else "application/octet-stream"),
                    size=len(data),
                    source=media,
                )
            )
        return result


class WeChatQRLoginSession:
    """Step-wise iLink QR login session suitable for a WebUI poll endpoint."""

    def __init__(self, base_url: str = BASE_URL, *, session: Any = None) -> None:
        self.base_url = str(base_url or BASE_URL).rstrip("/")
        self.current_base = self.base_url
        self.qrcode_val: str | None = None
        self.qrcode_img_url: str = ""
        self.qrcode_png_b64: str | None = None
        self.state = "init"
        self.bot_token: str | None = None
        self.result_base_url: str | None = None
        self.error: str | None = None
        self._session = session
        self._owns_session = False
        self._pending_verify: str | None = None

    async def _ensure_session(self) -> Any:
        if self._session is None:
            try:
                import aiohttp
            except ImportError as error:  # pragma: no cover
                raise ChannelError("微信 QR 登录需要安装 aiohttp") from error
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def _request(self, method: str, url: str, *, json_body: Any = None) -> dict[str, Any]:
        session = await self._ensure_session()
        request = getattr(session, method.lower(), None)
        if not callable(request):
            raise ChannelError(f"HTTP session does not support {method.upper()}")
        kwargs: dict[str, Any] = {"headers": _headers()}
        if json_body is not None:
            kwargs["json"] = json_body
        response = request(url, **kwargs)
        if hasattr(response, "__aenter__"):
            async with response as entered:
                status = _status(entered)
                if status >= 400:
                    raise ChannelHTTPError(f"微信 QR 请求失败（HTTP {status}）", status=status, retry_after=_retry_after(entered))
                payload = await _response_json(entered)
                WeChatIlinkAdapter._raise_provider_error(payload)
                return payload
        if hasattr(response, "__await__"):
            response = await response
        status = _status(response)
        if status >= 400:
            raise ChannelHTTPError(f"微信 QR 请求失败（HTTP {status}）", status=status, retry_after=_retry_after(response))
        payload = await _response_json(response)
        WeChatIlinkAdapter._raise_provider_error(payload)
        return payload

    async def start(self) -> dict[str, Any]:
        url = f"{self.current_base}/ilink/bot/get_bot_qrcode?bot_type=3"
        try:
            data = await self._request("POST", url, json_body={"local_token_list": []})
            if not _first(data, "qrcode", "qr_code"):
                data = await self._request("GET", url)
            self.qrcode_val = str(_first(data, "qrcode", "qr_code", default="") or "") or None
            self.qrcode_img_url = str(_first(data, "qrcode_img_content", "qrcode_img_url", "qrcode_url", default="") or "")
            if not self.qrcode_val:
                self.state = "error"
                self.error = "获取登录二维码失败，请检查网络"
            else:
                self.state = "waiting"
                self.qrcode_png_b64 = _render_qrcode_png(self.qrcode_img_url or self.qrcode_val)
        except Exception as error:
            self.state = "error"
            self.error = str(error)
        return self.snapshot()

    async def submit_verify_code(self, code: str) -> None:
        value = str(code or "").strip()
        if not value:
            raise ValueError("配对码不能为空")
        self._pending_verify = value

    async def poll_once(self) -> dict[str, Any]:
        if self.state in {"confirmed", "expired", "error"}:
            return self.snapshot()
        if not self.qrcode_val:
            self.state = "error"
            self.error = self.error or "二维码尚未生成"
            return self.snapshot()
        if self.state == "need_verifycode" and not self._pending_verify:
            return self.snapshot()
        url = f"{self.current_base}/ilink/bot/get_qrcode_status?qrcode={quote(self.qrcode_val, safe='')}"
        if self._pending_verify:
            url += f"&verify_code={quote(self._pending_verify, safe='')}"
        submitted = self._pending_verify
        try:
            status = await self._request("GET", url)
        except Exception as error:
            self.error = f"轮询失败：{error}"
            return self.snapshot()
        state = str(_first(status, "status", "state", default="") or "").lower()
        if _first(status, "bot_token", "token") or state in {"confirmed", "confirm", "success"}:
            self.bot_token = str(_first(status, "bot_token", "token", default="") or "") or None
            self.result_base_url = str(_first(status, "baseurl", "base_url", "baseUrl", default=self.current_base) or self.current_base)
            self.current_base = self.result_base_url
            self.state = "confirmed"
            self.error = None
        elif state in {"scaned", "scanned", "scan"}:
            self.state = "scanned"
            self._pending_verify = None
        elif state in {"need_verifycode", "need_verify_code", "verifycode"}:
            if submitted and _first(status, "retry_verifycode", "retry_verify_code"):
                self.error = "配对码不匹配，请重新输入"
            self._pending_verify = None
            self.state = "need_verifycode"
        elif state in {"verify_code_blocked", "verifycode_blocked"}:
            self.state = "error"
            self.error = "配对码多次错误，请重新获取二维码"
        elif state in {"scaned_but_redirect", "scanned_but_redirect"}:
            redirect = _first(status, "redirect_host", "redirectHost")
            if redirect:
                self.current_base = str(redirect)
                if not self.current_base.startswith("http"):
                    self.current_base = f"https://{self.current_base}"
        elif state in {"expired", "expire", "timeout"}:
            self.state = "expired"
            self.error = "二维码已过期，请重新获取"
        elif state in {"binded_redirect", "bound_redirect", "already_bound"}:
            self.state = "error"
            self.error = "检测到该微信已绑定其他连接"
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "qrcode": self.qrcode_val,
            "qrcode_img_url": self.qrcode_img_url,
            "qrcode_png_b64": self.qrcode_png_b64,
            "bot_token": self.bot_token,
            "base_url": self.result_base_url,
            "error": self.error,
        }

    async def close(self) -> None:
        if self._owns_session and self._session is not None:
            close = getattr(self._session, "close", None)
            if callable(close):
                value = close()
                if hasattr(value, "__await__"):
                    await value
        self._session = None if self._owns_session else self._session
        self._owns_session = False


def _render_qrcode_png(data: str) -> str | None:
    """Render a QR payload locally and return a base64 encoded PNG."""

    try:
        import qrcode

        qr = qrcode.QRCode(border=2, box_size=8)
        qr.add_data(data)
        qr.make(fit=True)
        image = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None


# Compatibility names used by integrations and by the reference client.
WeChatAdapter = WeChatIlinkAdapter
QRLoginSession = WeChatQRLoginSession
RelayHubClient = WeChatIlinkAdapter

# Public aliases for tests/extensions that prefer names without a leading
# underscore.  The underscored forms remain the canonical protocol helpers.
aes_ecb_encrypt = _aes_ecb_encrypt
aes_ecb_decrypt = _aes_ecb_decrypt
decode_media_key = _decode_media_key
build_headers = _headers
split_text = _split_text


__all__ = [
    "BASE_URL",
    "CHANNEL_VERSION",
    "ILINK_APP_ID",
    "ILINK_CLIENT_VERSION",
    "BOT_AGENT",
    "WeChatIlinkAdapter",
    "WeChatAdapter",
    "WeChatQRLoginSession",
    "QRLoginSession",
    "RelayHubClient",
    "_headers",
    "_base_info",
    "_decode_media_key",
    "_aes_ecb_encrypt",
    "_aes_ecb_decrypt",
    "_split_text",
    "aes_ecb_encrypt",
    "aes_ecb_decrypt",
    "decode_media_key",
    "build_headers",
    "split_text",
]
