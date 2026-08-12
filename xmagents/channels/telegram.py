"""Telegram Bot API adapter.

The adapter uses raw Bot API calls rather than a large framework so one bot
account can be managed by the same lifecycle as the WeChat account.  The
client is injectable, making parsing, proxy configuration and streaming easy
to test without Telegram network access.
"""

from __future__ import annotations

import asyncio
import mimetypes
import re
import time
from urllib.parse import quote
from collections.abc import AsyncIterable, Iterable, Mapping
from pathlib import Path
from typing import Any

import httpx

from xmagents.models import Attachment, DeliveryResult, IncomingMessage
from xmagents.files import MAX_FILE_BYTES

from .base import ChannelAdapter, ChannelError, ChannelHTTPError
from .redaction import redact_text


TELEGRAM_API = "https://api.telegram.org"
TELEGRAM_FILE_API = "https://api.telegram.org/file"
MAX_TEXT_LENGTH = 4096
MAX_CAPTION_LENGTH = 1024
DOWNLOAD_CHUNK_BYTES = 64 * 1024


def _safe_filename(value: str, fallback: str = "attachment") -> str:
    value = Path(str(value or "")).name
    value = re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip(" ._")
    return value[:240] or fallback


def _safe_path_component(value: str, fallback: str = "attachment") -> str:
    """Reject URL path traversal before using a Bot API file path locally."""

    return _safe_filename(Path(str(value or "")).name, fallback)


def split_text(text: str, limit: int = MAX_TEXT_LENGTH) -> list[str]:
    """Split text into Telegram-sized chunks without changing the text.

    A previous implementation discarded the separator chosen as a convenient
    split point.  That made a long reply lose newlines and sometimes spaces,
    which is especially visible in code blocks.  Keeping the separator at the
    start of the following chunk preserves the exact payload while retaining
    the provider's hard character limit.
    """

    if limit < 1:
        raise ValueError("limit 必须大于 0")
    text = str(text or "")
    if not text:
        return [""]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit + 1)
        if cut < max(1, limit // 3):
            cut = remaining.rfind(" ", 0, limit + 1)
        if cut < 1:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]
    chunks.append(remaining)
    return chunks


def _coerce_size(value: Any) -> int:
    """Return a non-negative provider-declared size without trusting it."""

    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _message_from_update(update: Mapping[str, Any], account_id: str) -> IncomingMessage | None:
    """Decode a Telegram update (message, edited message, or channel post)."""

    update_id = update.get("update_id")
    message: Mapping[str, Any] | None = None
    event_kind = "message"
    for key in ("message", "edited_message", "channel_post", "edited_channel_post"):
        candidate = update.get(key)
        if isinstance(candidate, Mapping):
            message = candidate
            event_kind = key
            break
    if not message:
        return None
    chat = message.get("chat") if isinstance(message.get("chat"), Mapping) else {}
    sender = message.get("from") if isinstance(message.get("from"), Mapping) else {}
    chat_id = chat.get("id")
    if chat_id is None:
        return None
    chat_type = str(chat.get("type") or "private")
    kind = "private" if chat_type == "private" else "group"
    sender_id = str(sender.get("id")) if sender.get("id") is not None else str(chat_id)
    display = " ".join(str(part) for part in (sender.get("first_name"), sender.get("last_name")) if part) or str(chat.get("title") or "")
    text = str(message.get("text") or message.get("caption") or "")
    attachments: list[Attachment] = []
    media: Mapping[str, Any] | None = None
    media_kind = "file"
    if isinstance(message.get("document"), Mapping):
        media, media_kind = message["document"], "file"
    elif isinstance(message.get("photo"), list) and message["photo"]:
        photos = [item for item in message["photo"] if isinstance(item, Mapping)]
        if photos:
            media, media_kind = max(photos, key=lambda item: _coerce_size(item.get("file_size"))), "image"
    elif isinstance(message.get("video"), Mapping):
        media, media_kind = message["video"], "video"
    elif isinstance(message.get("audio"), Mapping):
        media, media_kind = message["audio"], "audio"
    elif isinstance(message.get("voice"), Mapping):
        media, media_kind = message["voice"], "audio"
    elif isinstance(message.get("animation"), Mapping):
        media, media_kind = message["animation"], "video"
    elif isinstance(message.get("sticker"), Mapping):
        media, media_kind = message["sticker"], "image"
    if media:
        raw_name = media.get("file_name") or media.get("file_unique_id") or media.get("file_id") or f"telegram-{update_id or 'attachment'}"
        mime = str(media.get("mime_type") or mimetypes.guess_type(str(raw_name))[0] or "application/octet-stream")
        attachments.append(Attachment(path="", filename=_safe_filename(str(raw_name)), kind=media_kind, mime_type=mime, size=_coerce_size(media.get("file_size")), source=dict(media)))
    # update_id is globally unique for a bot and is the durable inbox key.
    # A message_id is only unique within a chat and can collide across peers.
    external_id = str(update_id if update_id is not None else message.get("message_id") or "")
    return IncomingMessage(
        channel="telegram",
        account_id=account_id,
        peer_id=str(chat_id),
        external_id=external_id,
        text=text,
        sender_id=sender_id,
        sender_name=display,
        kind=kind,
        attachments=attachments,
        raw=dict(update),
        metadata={
            "update_id": update_id,
            "chat_type": chat_type,
            "event_kind": event_kind,
            "chat_title": chat.get("title", ""),
            "username": sender.get("username") or chat.get("username"),
            "bot_mention": bool(message.get("entities")),
        },
    )


def _update_message(update: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("message", "edited_message", "channel_post", "edited_channel_post"):
        value = update.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _utf16_entity_text(text: str, offset: Any, length: Any) -> str:
    """Extract a Telegram entity, whose offsets are UTF-16 code units."""

    try:
        start = int(offset)
        end = start + int(length)
    except (TypeError, ValueError):
        return ""
    if start < 0 or end < start:
        return ""
    encoded = text.encode("utf-16-le")
    start_byte, end_byte = start * 2, end * 2
    if end_byte > len(encoded):
        return ""
    try:
        return encoded[start_byte:end_byte].decode("utf-16-le")
    except UnicodeDecodeError:
        return ""


def _group_triggered(message: Mapping[str, Any], bot: Mapping[str, Any]) -> bool:
    text = str(message.get("text") or message.get("caption") or "")
    username = str(bot.get("username") or "").lower()
    # Telegram commands are intentionally accepted in a whitelisted group.
    # With privacy mode disabled, however, a bot may receive commands
    # addressed to another bot.  Only accept unqualified commands or ones
    # explicitly addressed to this configured bot.
    command = re.match(r"^\s*/[A-Za-z0-9_]+(?:@([A-Za-z0-9_]+))?(?:\s|$)", text)
    if command:
        target = command.group(1)
        return target is None or bool(username and target.lower() == username)
    reply = message.get("reply_to_message")
    if isinstance(reply, Mapping) and isinstance(reply.get("from"), Mapping):
        if str(reply["from"].get("id")) == str(bot.get("id")):
            return True
    for entity in message.get("entities") or message.get("caption_entities") or []:
        if not isinstance(entity, Mapping) or entity.get("type") != "mention":
            continue
        mention = _utf16_entity_text(text, entity.get("offset", 0), entity.get("length", 0)).lstrip("@").lower()
        if username and mention == username:
            return True
    return False


def _normalize_group_command(text: str, bot: Mapping[str, Any]) -> str:
    """Remove Telegram's ``@bot`` command suffix before runtime parsing.

    Telegram delivers group commands such as ``/status@my_bot``.  The group
    gate understands that form, but the provider-neutral command parser should
    receive the same command syntax as private chats.  Only remove a suffix
    addressed to this configured bot; commands for another bot are left alone.
    """

    username = str(bot.get("username") or "").strip().lower()
    if not username:
        return text
    match = re.match(r"^(\s*/[A-Za-z0-9_]+)@([A-Za-z0-9_]+)(?=\s|$)(.*)$", str(text or ""), re.S)
    if not match or match.group(2).lower() != username:
        return text
    return f"{match.group(1)}{match.group(3)}"


def _timeout_for_poll(request_timeout: float | httpx.Timeout, poll_timeout: int) -> float | httpx.Timeout:
    """Ensure long polling is not cut off by the HTTP client's read timeout.

    ``getUpdates(timeout=25)`` can legally hold the request for 25 seconds.
    A user supplied scalar timeout below that value otherwise creates a tight
    timeout/retry loop.  Detailed ``httpx.Timeout`` values are left intact so
    callers retain their explicitly configured connect/write pools.
    """

    minimum = float(poll_timeout) + 10.0
    if isinstance(request_timeout, httpx.Timeout):
        # ``None`` explicitly means no read timeout, which already satisfies
        # long polling.  Preserve all other timeout facets while extending an
        # accidentally too-short read timeout.
        if request_timeout.read is None or request_timeout.read >= minimum:
            return request_timeout
        return httpx.Timeout(
            connect=request_timeout.connect,
            read=minimum,
            write=request_timeout.write,
            pool=request_timeout.pool,
        )
    try:
        return max(float(request_timeout), minimum)
    except (TypeError, ValueError):
        return minimum


class TelegramAdapter(ChannelAdapter):
    channel = "telegram"
    # Telegram update ids are monotonically increasing.  Once an update has
    # been handled, its next offset is a safe durable checkpoint even if a
    # later update from the same long-poll batch fails.
    cursor_checkpoint_per_message = True

    def __init__(
        self,
        account_id: str,
        bot_token: str,
        *,
        proxy: str | None = None,
        base_url: str = TELEGRAM_API,
        file_base_url: str | None = None,
        poll_timeout: int = 25,
        request_timeout: float | httpx.Timeout = 40.0,
        client: httpx.AsyncClient | None = None,
        max_text_length: int = MAX_TEXT_LENGTH,
        group_allowlist: Iterable[str] | None = None,
    ):
        super().__init__(account_id)
        if not bot_token or not str(bot_token).strip():
            raise ValueError("Telegram bot token 不能为空")
        self.bot_token = str(bot_token).strip()
        self.proxy = proxy or None
        self.base_url = base_url.rstrip("/")
        self.file_base_url = (file_base_url or f"{self.base_url}/file").rstrip("/")
        self.poll_timeout = max(0, int(poll_timeout))
        self.request_timeout = _timeout_for_poll(request_timeout, self.poll_timeout)
        self.max_text_length = max(1, int(max_text_length))
        self.group_allowlist = {str(value) for value in (group_allowlist or ())}
        self.offset: int | None = None
        self.bot: dict[str, Any] = {}
        self._client = client
        self._owns_client = client is None
        self._send_lock = asyncio.Lock()

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/bot{self.bot_token}"

    def _new_client(self) -> httpx.AsyncClient:
        kwargs: dict[str, Any] = {
            "timeout": self.request_timeout,
            "trust_env": False,
            "follow_redirects": True,
        }
        if self.proxy:
            # httpx >=0.28 uses ``proxy``.  Keep a fallback for older clients
            # used by downstream deployments.
            kwargs["proxy"] = self.proxy
        try:
            return httpx.AsyncClient(**kwargs)
        except TypeError:
            kwargs.pop("proxy", None)
            if self.proxy:
                kwargs["proxies"] = self.proxy
            return httpx.AsyncClient(**kwargs)

    async def start(self) -> "TelegramAdapter":
        if self._client is None:
            self._client = self._new_client()
        self._started = True
        if not self.bot:
            try:
                self.bot = await self._api("getMe")
            except Exception:
                await self.stop()
                raise
        return self

    async def stop(self) -> None:
        client, owns = self._client, self._owns_client
        self._started = False
        if client is not None and owns:
            await client.aclose()
            self._client = None

    async def _api(self, method: str, payload: Mapping[str, Any] | None = None, *, files: Any = None, data: Any = None) -> Any:
        if self._client is None:
            self._client = self._new_client()
        # Telegram rejects explicit JSON null fields, most importantly
        # ``offset: null`` during the first long-poll request.
        body = {key: value for key, value in dict(payload or {}).items() if value is not None}
        try:
            if files is not None:
                response = await self._client.post(f"{self.api_url}/{method}", data=data or body, files=files)
            else:
                response = await self._client.post(f"{self.api_url}/{method}", json=body)
        except (httpx.HTTPError, OSError) as exc:
            raise ChannelHTTPError(f"Telegram 请求失败: {redact_text(exc, secrets=(self.bot_token,))}") from exc
        try:
            raw = response.json()
        except (ValueError, TypeError) as exc:
            raise ChannelHTTPError(f"Telegram 返回无效 JSON（HTTP {response.status_code}）", status=response.status_code) from exc
        if not isinstance(raw, Mapping):
            raise ChannelHTTPError(f"Telegram 返回无效 JSON（HTTP {response.status_code}）", status=response.status_code)
        if response.status_code >= 400 or not raw.get("ok", False):
            params = raw.get("parameters") if isinstance(raw, Mapping) else {}
            retry_after = params.get("retry_after") if isinstance(params, Mapping) else None
            description = raw.get("description") if isinstance(raw, Mapping) else None
            try:
                parsed_retry_after = float(retry_after) if retry_after is not None else None
            except (TypeError, ValueError):
                parsed_retry_after = None
            raise ChannelHTTPError(
                redact_text(description or f"Telegram API HTTP {response.status_code}", secrets=(self.bot_token,)),
                status=response.status_code,
                retry_after=parsed_retry_after,
            )
        return raw.get("result", {})

    @staticmethod
    def _mapping_result(result: Any, operation: str) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise ChannelError(f"Telegram {operation} 返回格式无效")
        return dict(result)

    async def get_me(self) -> dict[str, Any]:
        result = await self._api("getMe")
        self.bot = self._mapping_result(result, "getMe")
        return dict(self.bot)

    async def get_updates(self, *, timeout: int | None = None, limit: int = 100, allowed_updates: list[str] | None = None) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"offset": self.offset, "timeout": self.poll_timeout if timeout is None else max(0, int(timeout)), "limit": min(100, max(1, int(limit)))}
        if allowed_updates is not None:
            payload["allowed_updates"] = allowed_updates
        updates = await self._api("getUpdates", payload)
        return updates if isinstance(updates, list) else []

    async def poll_once(self, cursor: str | None = None) -> list[IncomingMessage]:
        if cursor is not None and str(cursor):
            try:
                self.offset = int(cursor)
            except ValueError:
                pass
        updates = await self.get_updates()
        messages: list[IncomingMessage] = []
        for update in updates:
            if not isinstance(update, Mapping):
                continue
            update_id = update.get("update_id")
            if update_id is not None:
                try:
                    self.offset = int(update_id) + 1
                except (TypeError, ValueError):
                    pass
            message = _message_from_update(update, self.account_id)
            if message is None:
                continue
            # Group chats are opt-in: the allowlist must explicitly contain
            # the chat id, otherwise an account can only receive private DMs.
            if message.kind == "group" and message.peer_id not in self.group_allowlist:
                message.metadata["approved"] = False
                continue
            if message.kind == "group" and not _group_triggered(_update_message(update), self.bot):
                continue
            if message.kind == "group":
                message.text = _normalize_group_command(message.text, self.bot)
            # Capture the checkpoint for this update, rather than the final
            # batch offset.  The service can then safely persist every handled
            # Telegram update without skipping a later failed update.
            message.metadata["next_cursor"] = str(self.offset or "")
            messages.append(message)
        return messages

    async def send_text(self, peer_id: str, text: str, *, reply_to: str | int | None = None, parse_mode: str | None = None, disable_web_page_preview: bool = True, **_: Any) -> DeliveryResult:
        chunks = split_text(text, self.max_text_length)
        last_id: str | None = None
        try:
            async with self._send_lock:
                for chunk in chunks:
                    payload: dict[str, Any] = {"chat_id": str(peer_id), "text": chunk, "disable_web_page_preview": disable_web_page_preview}
                    if parse_mode:
                        payload["parse_mode"] = parse_mode
                    if reply_to is not None:
                        payload["reply_to_message_id"] = int(reply_to)
                    result = await self._api("sendMessage", payload)
                    result = self._mapping_result(result, "sendMessage")
                    last_id = str(result.get("message_id")) if result.get("message_id") is not None else last_id
                    reply_to = None
            return DeliveryResult.success(raw={"chunks": len(chunks)}, external_id=last_id)
        except ChannelError as exc:
            return DeliveryResult.failure(str(exc), status=getattr(exc, "status", None), retry_after=getattr(exc, "retry_after", None))

    async def send_file(self, peer_id: str, path: str, *, caption: str | None = None, reply_to: str | int | None = None, **_: Any) -> DeliveryResult:
        file_path = Path(path)
        if not file_path.is_file():
            return DeliveryResult.failure(f"文件不存在: {path}")
        try:
            if file_path.stat().st_size > MAX_FILE_BYTES:
                return DeliveryResult.failure(f"文件超过 {MAX_FILE_BYTES // 1024 // 1024}MB 限制")
        except OSError as exc:
            return DeliveryResult.failure(f"无法读取文件: {exc}")
        caption = (caption or "")[:MAX_CAPTION_LENGTH]
        try:
            with file_path.open("rb") as handle:
                files = {"document": (_safe_filename(file_path.name), handle, mimetypes.guess_type(file_path.name)[0] or "application/octet-stream")}
                data: dict[str, Any] = {"chat_id": str(peer_id)}
                if caption:
                    data["caption"] = caption
                if reply_to is not None:
                    data["reply_to_message_id"] = int(reply_to)
                result = await self._api("sendDocument", files=files, data=data)
                result = self._mapping_result(result, "sendDocument")
            message_id = result.get("message_id") if isinstance(result, Mapping) else None
            return DeliveryResult.success(raw=result, external_id=str(message_id) if message_id is not None else None)
        except (OSError, ChannelError) as exc:
            return DeliveryResult.failure(str(exc), status=getattr(exc, "status", None), retry_after=getattr(exc, "retry_after", None))

    async def download_attachments(self, message: IncomingMessage, destination: str) -> list[Attachment]:
        if self._client is None:
            self._client = self._new_client()
        target = Path(destination).resolve()
        target.mkdir(parents=True, exist_ok=True)
        result: list[Attachment] = []
        for index, item in enumerate(message.attachments):
            if item.size and item.size > MAX_FILE_BYTES:
                raise ChannelError(f"Telegram 附件超过 {MAX_FILE_BYTES // 1024 // 1024}MB 限制")
            file_id = item.source.get("file_id")
            if not file_id:
                continue
            info = await self.get_file(str(file_id))
            file_path = info.get("file_path")
            if not file_path:
                raise ChannelError("Telegram getFile 未返回 file_path")
            # File paths are supplied by Telegram but are still untrusted
            # input. Quote components so a malicious/fake response cannot
            # alter the token-bearing URL path.
            raw_parts = str(file_path).split("/")
            if any(part in {"", ".", ".."} for part in raw_parts):
                raise ChannelError("Telegram 返回了无效 file_path")
            safe_parts = "/".join(quote(_safe_path_component(part, "file"), safe="._-") for part in raw_parts)
            if not safe_parts:
                raise ChannelError("Telegram 返回了无效 file_path")
            url = f"{self.file_base_url}/bot{self.bot_token}/{safe_parts}"
            try:
                async with self._client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_length = response.headers.get("Content-Length")
                    try:
                        declared_size = int(content_length) if content_length is not None else 0
                    except (TypeError, ValueError):
                        declared_size = 0
                    if declared_size > MAX_FILE_BYTES:
                        raise ChannelError(f"Telegram 附件超过 {MAX_FILE_BYTES // 1024 // 1024}MB 限制")
                    chunks: list[bytes] = []
                    received = 0
                    async for chunk in response.aiter_bytes(DOWNLOAD_CHUNK_BYTES):
                        received += len(chunk)
                        if received > MAX_FILE_BYTES:
                            raise ChannelError(f"Telegram 附件超过 {MAX_FILE_BYTES // 1024 // 1024}MB 限制")
                        chunks.append(chunk)
                    content = b"".join(chunks)
            except (httpx.HTTPError, OSError) as exc:
                raise ChannelHTTPError(f"Telegram 文件下载失败: {redact_text(exc, secrets=(self.bot_token,))}") from exc
            filename = _safe_filename(item.filename or Path(file_path).name, f"attachment-{index}")
            destination_path = target / filename
            if destination_path.exists():
                destination_path = target / f"{destination_path.stem}-{int(time.time() * 1000)}{destination_path.suffix}"
            destination_path.write_bytes(content)
            result.append(Attachment(path=str(destination_path), filename=destination_path.name, kind=item.kind, mime_type=item.mime_type, size=len(content), source={**item.source, "file_path": file_path}))
        return result

    async def edit_text(self, peer_id: str, message_id: str | int, text: str, *, parse_mode: str | None = None) -> DeliveryResult:
        # Telegram message edits have the same 4096 character limit, but an
        # edit cannot represent multiple chunks; retain the first chunk.
        payload: dict[str, Any] = {"chat_id": str(peer_id), "message_id": int(message_id), "text": split_text(text, self.max_text_length)[0]}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            result = await self._api("editMessageText", payload)
            result = self._mapping_result(result, "editMessageText")
            return DeliveryResult.success(raw=result, external_id=str(result.get("message_id", message_id)))
        except ChannelError as exc:
            return DeliveryResult.failure(str(exc), status=getattr(exc, "status", None), retry_after=getattr(exc, "retry_after", None))

    # Small provider-shaped aliases are useful to callers that also use the
    # raw Bot API naming, while the gateway itself uses the common methods.
    async def send_message(self, peer_id: str, text: str, **kwargs: Any) -> DeliveryResult:
        return await self.send_text(peer_id, text, **kwargs)

    async def send_document(self, peer_id: str, path: str, **kwargs: Any) -> DeliveryResult:
        return await self.send_file(peer_id, path, **kwargs)

    async def edit_message_text(self, peer_id: str, message_id: str | int, text: str, **kwargs: Any) -> DeliveryResult:
        return await self.edit_text(peer_id, message_id, text, **kwargs)

    async def get_file(self, file_id: str) -> dict[str, Any]:
        result = await self._api("getFile", {"file_id": file_id})
        return self._mapping_result(result, "getFile")

    def open_text_stream(self, peer_id: str, *, parse_mode: str | None = None,
                         update_interval: float = 0.7, reply_to: str | int | None = None) -> "TelegramTextStream":
        """Create a push-based stream for a live Agent response.

        ``stream_text`` remains useful for an async iterator.  This variant is
        intended for SDK callbacks where chunks arrive one at a time and must
        be rendered before the final result is known.
        """

        return TelegramTextStream(self, peer_id, parse_mode=parse_mode, update_interval=update_interval, reply_to=reply_to)

    async def stream_text(self, peer_id: str, chunks: AsyncIterable[str] | Iterable[str], *, parse_mode: str | None = None, update_interval: float = 0.7, reply_to: str | int | None = None) -> DeliveryResult:
        """Send an initial message and periodically edit it as chunks arrive."""

        iterator = chunks.__aiter__() if hasattr(chunks, "__aiter__") else None
        sync_iterator = None if iterator is not None else iter(chunks)
        text = ""
        message_id: str | None = None
        last_update = 0.0
        try:
            while True:
                try:
                    chunk = await iterator.__anext__() if iterator is not None else next(sync_iterator)  # type: ignore[arg-type]
                except (StopAsyncIteration, StopIteration):
                    break
                text += str(chunk)
                now = time.monotonic()
                if message_id is None:
                    # Only render the editable first message.  ``send_text``
                    # would split a long early delta and leave duplicate
                    # overflow messages before the final flush.
                    sent = await self.send_text(peer_id, split_text(text, self.max_text_length)[0], parse_mode=parse_mode, reply_to=reply_to)
                    if not sent.ok:
                        return sent
                    message_id = sent.external_id
                    last_update = now
                elif now - last_update >= max(0.05, update_interval) and message_id:
                    edited = await self.edit_text(peer_id, message_id, text, parse_mode=parse_mode)
                    if not edited.ok:
                        return edited
                    last_update = now
            if message_id is None:
                return await self.send_text(peer_id, text, parse_mode=parse_mode, reply_to=reply_to)
            chunks_out = split_text(text, self.max_text_length)
            final = await self.edit_text(peer_id, message_id, chunks_out[0], parse_mode=parse_mode)
            if not final.ok:
                return final
            last_id = final.external_id or message_id
            for chunk in chunks_out[1:]:
                sent = await self.send_text(peer_id, chunk, parse_mode=parse_mode)
                if not sent.ok:
                    return sent
                last_id = sent.external_id or last_id
            return DeliveryResult.success(raw={"chunks": len(chunks_out)}, external_id=last_id)
        except (ChannelError, TypeError, OSError) as exc:
            return DeliveryResult.failure(str(exc))


class TelegramTextStream:
    """Incremental Telegram message editor with a final chunk-safe flush."""

    def __init__(self, adapter: TelegramAdapter, peer_id: str, *, parse_mode: str | None = None,
                 update_interval: float = 0.7, reply_to: str | int | None = None):
        self.adapter = adapter
        self.peer_id = str(peer_id)
        self.parse_mode = parse_mode
        self.update_interval = max(0.05, float(update_interval))
        self.reply_to = reply_to
        self.text = ""
        self.message_id: str | None = None
        self._last_update = 0.0
        self._closed = False
        self._lock = asyncio.Lock()

    async def push(self, chunk: str) -> DeliveryResult:
        """Append a delta and, when due, render the first Telegram chunk."""

        async with self._lock:
            if self._closed:
                return DeliveryResult.failure("Telegram 流已结束")
            self.text += str(chunk or "")
            if not self.text:
                return DeliveryResult.success(external_id=self.message_id)
            now = time.monotonic()
            first = split_text(self.text, self.adapter.max_text_length)[0]
            if self.message_id is None:
                result = await self.adapter.send_text(self.peer_id, first, parse_mode=self.parse_mode, reply_to=self.reply_to)
                if result.ok:
                    self.message_id = result.external_id
                    self._last_update = now
                return result
            if now - self._last_update < self.update_interval:
                return DeliveryResult.success(external_id=self.message_id)
            result = await self.adapter.edit_text(self.peer_id, self.message_id, first, parse_mode=self.parse_mode)
            if result.ok:
                self._last_update = now
            return result

    async def finish(self, final_text: str | None = None) -> DeliveryResult:
        """Write the authoritative final response and any overflow chunks."""

        async with self._lock:
            if self._closed:
                return DeliveryResult.failure("Telegram 流已结束")
            self._closed = True
            text = self.text if final_text is None else str(final_text)
            chunks = split_text(text, self.adapter.max_text_length)
            if self.message_id is None:
                return await self.adapter.send_text(self.peer_id, text, parse_mode=self.parse_mode, reply_to=self.reply_to)
            first = await self.adapter.edit_text(self.peer_id, self.message_id, chunks[0], parse_mode=self.parse_mode)
            if not first.ok:
                return first
            last_id = first.external_id or self.message_id
            for chunk in chunks[1:]:
                sent = await self.adapter.send_text(self.peer_id, chunk, parse_mode=self.parse_mode)
                if not sent.ok:
                    return sent
                last_id = sent.external_id or last_id
            return DeliveryResult.success(raw={"chunks": len(chunks)}, external_id=last_id)


TelegramBotAdapter = TelegramAdapter
TelegramChannelAdapter = TelegramAdapter

__all__ = ["TelegramAdapter", "TelegramBotAdapter", "TelegramChannelAdapter", "TelegramTextStream", "split_text", "_message_from_update", "_group_triggered", "_normalize_group_command", "MAX_TEXT_LENGTH"]
