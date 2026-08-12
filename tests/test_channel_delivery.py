from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from xmagents.channels.telegram import TelegramAdapter, _timeout_for_poll
from xmagents.channels.wechat import WeChatIlinkAdapter
from xmagents.config import AppPaths
from xmagents.files import MAX_FILE_BYTES
from xmagents.models import DeliveryResult, IncomingMessage
from xmagents.service import AppService


class FakeAdapter:
    channel = "telegram"

    def __init__(self, results: list[DeliveryResult] | None = None):
        self.results = list(results or [DeliveryResult.success(external_id="42")])
        self.calls: list[tuple[str, str, dict]] = []

    async def send_text(self, peer_id: str, text: str, **kwargs):
        self.calls.append((peer_id, text, kwargs))
        return self.results.pop(0) if self.results else DeliveryResult.success(external_id="next")

    async def send_file(self, peer_id: str, path: str, **kwargs):
        self.calls.append((peer_id, path, kwargs))
        return self.results.pop(0) if self.results else DeliveryResult.success(external_id="file")


def _service(tmp_path: Path) -> AppService:
    return AppService(AppPaths.from_root(tmp_path, tmp_path / "data"))


@pytest.mark.asyncio
async def test_outbox_success_is_persisted_and_sent(tmp_path: Path):
    service = _service(tmp_path)
    account = service.create_channel("telegram", "bot", token="not-a-real-token")
    adapter = FakeAdapter()
    service.channels[account["id"]] = adapter

    result = await service.send_message(account["id"], "100", "hello")

    assert result.ok
    assert adapter.calls == [("100", "hello", {"context_token": None})]
    row = service.db.fetchone("SELECT state,attempts,sent_at FROM outbox")
    assert row and row["state"] == "sent" and row["attempts"] == 1 and row["sent_at"]


@pytest.mark.asyncio
async def test_outbox_429_is_requeued_without_duplicate_immediate_send(tmp_path: Path):
    service = _service(tmp_path)
    account = service.create_channel("telegram", "bot", token="not-a-real-token")
    adapter = FakeAdapter([DeliveryResult.failure("Too Many Requests", status=429, retry_after=30)])
    service.channels[account["id"]] = adapter

    result = await service.send_message(account["id"], "100", "hello")

    assert not result.ok
    assert len(adapter.calls) == 1
    row = service.db.fetchone("SELECT state,attempts,available_at,last_error FROM outbox")
    assert row and row["state"] == "pending" and row["attempts"] == 1
    assert row["last_error"] == "Too Many Requests"


@pytest.mark.asyncio
async def test_offline_channel_does_not_consume_outbox_attempt_budget(tmp_path: Path):
    service = _service(tmp_path)
    account = service.create_channel("telegram", "bot", token="not-a-real-token")

    result = await service.send_message(account["id"], "100", "hello")

    assert not result.ok
    row = service.db.fetchone("SELECT state,attempts FROM outbox")
    assert row and row["state"] == "pending" and row["attempts"] == 0


@pytest.mark.asyncio
async def test_wechat_deferred_delivery_needs_fresh_context(tmp_path: Path):
    service = _service(tmp_path)
    account = service.create_channel("wechat", "wechat", token="not-a-real-token")
    adapter = FakeAdapter()
    adapter.channel = "wechat"
    service.channels[account["id"]] = adapter
    delivery_id = service.enqueue_outbox(
        account["id"], "wx-user", {"text": "scheduled result", "context_token": None},
        kind="wechat_deferred", state="deferred",
    )

    assert await service._flush_wechat_deferred(account["id"], "wx-user", None) == 0
    assert service.db.fetchone("SELECT state FROM outbox WHERE id=?", (delivery_id,))["state"] == "deferred"
    assert await service._flush_wechat_deferred(account["id"], "wx-user", "new-context") == 1
    assert adapter.calls == [("wx-user", "scheduled result", {"context_token": "new-context"})]
    row = service.db.fetchone("SELECT state,kind FROM outbox WHERE id=?", (delivery_id,))
    assert row and row["state"] == "sent" and row["kind"] == "text"


@pytest.mark.asyncio
async def test_telegram_push_stream_edits_then_sends_overflow():
    adapter = TelegramAdapter("account", "123456:abcdefghijklmnopqrstuv", max_text_length=5)
    sent: list[str] = []
    edits: list[str] = []

    async def send_text(peer_id: str, text: str, **kwargs):
        sent.append(text)
        return DeliveryResult.success(external_id=str(len(sent)))

    async def edit_text(peer_id: str, message_id: str, text: str, **kwargs):
        edits.append(text)
        return DeliveryResult.success(external_id=message_id)

    adapter.send_text = send_text  # type: ignore[method-assign]
    adapter.edit_text = edit_text  # type: ignore[method-assign]
    stream = adapter.open_text_stream("100", update_interval=0)
    assert (await stream.push("abc")).ok
    assert (await stream.push("def")).ok
    result = await stream.finish("abcdefgh")

    assert result.ok
    assert sent == ["abc", "fgh"]
    assert edits[-1] == "abcde"


@pytest.mark.asyncio
async def test_telegram_poll_omits_null_offset_and_extends_read_timeout():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "result": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = TelegramAdapter("account", "123456:abcdefghijklmnopqrstuv", poll_timeout=25, request_timeout=1, client=client)
    await adapter.get_updates()
    assert calls == [{"timeout": 25, "limit": 100}]
    assert adapter.request_timeout == 35
    timeout = _timeout_for_poll(httpx.Timeout(1), 25)
    assert isinstance(timeout, httpx.Timeout) and timeout.read == 35
    await client.aclose()


@pytest.mark.asyncio
async def test_telegram_stream_text_flushes_overflow_once():
    adapter = TelegramAdapter("account", "123456:abcdefghijklmnopqrstuv", max_text_length=5)
    sent: list[str] = []
    edits: list[str] = []

    async def send_text(peer_id: str, text: str, **kwargs):
        sent.append(text)
        return DeliveryResult.success(external_id=str(len(sent)))

    async def edit_text(peer_id: str, message_id: str, text: str, **kwargs):
        edits.append(text)
        return DeliveryResult.success(external_id=message_id)

    adapter.send_text = send_text  # type: ignore[method-assign]
    adapter.edit_text = edit_text  # type: ignore[method-assign]
    result = await adapter.stream_text("100", ["abcdefgh"])

    assert result.ok
    assert sent == ["abcde", "fgh"]
    assert edits == ["abcde"]


def test_wechat_external_media_does_not_receive_bot_authorization():
    adapter = WeChatIlinkAdapter("account", "super-secret-token", base_url="https://ilinkai.weixin.qq.com")
    external_headers = adapter._media_headers("https://cdn.example.invalid/object")
    same_origin_headers = adapter._media_headers("https://ilinkai.weixin.qq.com/media/object")

    assert "Authorization" not in external_headers
    assert same_origin_headers["Authorization"] == "Bearer super-secret-token"


@pytest.mark.asyncio
async def test_channel_file_upload_rejects_oversized_file_without_network(tmp_path: Path):
    oversized = tmp_path / "too-large.bin"
    oversized.write_bytes(b"x")
    # Sparse files exercise the stat-before-read fast path without allocating
    # a 100MB test fixture.
    with oversized.open("r+b") as handle:
        handle.truncate(MAX_FILE_BYTES + 1)

    telegram = TelegramAdapter("account", "123456:abcdefghijklmnopqrstuv")
    wechat = WeChatIlinkAdapter("account", "token")
    telegram_result = await telegram.send_file("100", str(oversized))
    wechat_result = await wechat.send_file("100", str(oversized))

    assert not telegram_result.ok and "超过" in (telegram_result.error or "")
    assert not wechat_result.ok and "超过" in (wechat_result.error or "")


@pytest.mark.asyncio
async def test_outbox_file_delivery_rechecks_workspace_after_symlink_swap(tmp_path: Path):
    service = _service(tmp_path)
    account = service.create_channel("telegram", "bot", token="not-a-real-token")
    agent = service.create_agent("workspace")
    peer = service.upsert_peer(account["id"], "100", approved=True)
    service.approve_peer(peer["id"], agent_id=agent["id"])
    adapter = FakeAdapter()
    service.channels[account["id"]] = adapter

    target = Path(agent["workspace"]) / "report.txt"
    target.write_text("safe")
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    delivery_id = service.enqueue_outbox(account["id"], "100", {"path": str(target), "caption": None}, kind="file")
    target.unlink()
    target.symlink_to(outside)

    result = await service._deliver_outbox(delivery_id)

    assert not result.ok
    assert adapter.calls == []
    row = service.db.fetchone("SELECT state,last_error FROM outbox WHERE id=?", (delivery_id,))
    assert row and row["state"] == "failed" and "工作区" in row["last_error"]


@pytest.mark.asyncio
async def test_start_channel_replaces_errored_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service = _service(tmp_path)
    account = service.create_channel("telegram", "bot", token="not-a-real-token")
    service.update_channel(account["id"], {"status": "error"})
    old_adapter = FakeAdapter()

    async def stop_old_adapter():
        return None

    old_adapter.stop = stop_old_adapter  # type: ignore[attr-defined]
    service.channels[account["id"]] = old_adapter

    class ReplacementAdapter(FakeAdapter):
        async def start(self):
            return self

        async def stop(self):
            return None

        async def poll_once(self, _cursor):
            await asyncio.sleep(60)
            return []

    replacement = ReplacementAdapter()
    monkeypatch.setattr(service, "_build_adapter", lambda _row: replacement)

    await service.start_channel(account["id"])
    assert service.channels[account["id"]] is replacement
    await service.stop_channel(account["id"])


def _inbound(account_id: str, external_id: str, next_cursor: str, *, channel: str = "telegram") -> IncomingMessage:
    return IncomingMessage(
        channel=channel,
        account_id=account_id,
        peer_id="peer",
        external_id=external_id,
        text="hello",
        metadata={"next_cursor": next_cursor},
    )


@pytest.mark.asyncio
async def test_telegram_cursor_checkpoints_only_completed_messages(tmp_path: Path):
    """A failing later update must not make polling skip that update on retry."""

    service = _service(tmp_path)
    account = service.create_channel("telegram", "bot", token="not-a-real-token")
    service.update_channel(account["id"], {"cursor": "100"})
    first = _inbound(account["id"], "101", "102")
    second = _inbound(account["id"], "102", "103")

    class TelegramBatch:
        cursor_checkpoint_per_message = True
        offset = 103

        async def poll_once(self, cursor: str):
            assert cursor == "100"
            return [first, second]

    failed = asyncio.Event()

    async def handle(message: IncomingMessage) -> None:
        if message.external_id == "102":
            failed.set()
            raise RuntimeError("second update failed")

    service.handle_incoming = handle  # type: ignore[method-assign]
    task = asyncio.create_task(service._poll_channel(account["id"], TelegramBatch()))
    try:
        await asyncio.wait_for(failed.wait(), timeout=0.5)
        await asyncio.sleep(0)
        row = service.db.fetchone("SELECT cursor FROM channel_accounts WHERE id=?", (account["id"],))
        assert row and row["cursor"] == "102"
        states = {
            row["external_event_id"]: row["state"]
            for row in service.db.fetchall("SELECT external_event_id,state FROM inbox ORDER BY external_event_id")
        }
        assert states == {"101": "processed", "102": "received"}
        # The durable received state is deliberately retryable; it is not a
        # de-duplication hit until Agent handling has completed.
        assert service._record_inbox(account["id"], second) is True
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_wechat_batch_cursor_waits_for_entire_batch(tmp_path: Path):
    """iLink's opaque cursor is committed only after every item succeeds."""

    service = _service(tmp_path)
    account = service.create_channel("wechat", "wechat", token="not-a-real-token")
    first = _inbound(account["id"], "one", "opaque-next", channel="wechat")
    second = _inbound(account["id"], "two", "opaque-next", channel="wechat")

    class WeChatBatch:
        cursor_checkpoint_per_message = False
        cursor = "opaque-next"

        async def poll_once(self, cursor: str):
            assert cursor == ""
            return [first, second]

    failed = asyncio.Event()

    async def handle(message: IncomingMessage) -> None:
        if message.external_id == "two":
            failed.set()
            raise RuntimeError("second iLink message failed")

    service.handle_incoming = handle  # type: ignore[method-assign]
    task = asyncio.create_task(service._poll_channel(account["id"], WeChatBatch()))
    try:
        await asyncio.wait_for(failed.wait(), timeout=0.5)
        await asyncio.sleep(0)
        row = service.db.fetchone("SELECT cursor FROM channel_accounts WHERE id=?", (account["id"],))
        assert row and row["cursor"] == ""
        states = {
            row["external_event_id"]: row["state"]
            for row in service.db.fetchall("SELECT external_event_id,state FROM inbox ORDER BY external_event_id")
        }
        assert states == {"one": "processed", "two": "received"}
        assert service._record_inbox(account["id"], second) is True
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_telegram_split_text_preserves_whitespace_at_chunk_boundaries():
    from xmagents.channels.telegram import split_text

    text = "first line\nsecond line and more"
    assert "".join(split_text(text, 12)) == text
