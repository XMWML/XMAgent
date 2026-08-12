from __future__ import annotations

import asyncio
import os
import stat
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from xmagents.agents.runtime import AnthropicProvider
from xmagents.channels.base import ChannelAdapter
from xmagents.config import AppPaths
from xmagents.control import ControlClient, ControlError, controlled_environment
from xmagents.models import DeliveryResult
from xmagents.service import AppService


class FakeChannel(ChannelAdapter):
    channel = "telegram"

    def __init__(self, account_id: str):
        super().__init__(account_id)
        self.files: list[tuple[str, str, str | None]] = []

    async def poll_once(self, cursor=None):
        return []

    async def send_text(self, peer_id: str, text: str, **kwargs):
        return DeliveryResult.success()

    async def send_file(self, peer_id: str, path: str, **kwargs):
        self.files.append((peer_id, path, kwargs.get("caption")))
        return DeliveryResult.success(external_id="file-1")


async def _scenario() -> None:
    with TemporaryDirectory() as directory:
        service = AppService(AppPaths.from_root(Path.cwd(), Path(directory) / "data"))
        account = service.create_channel("telegram", "fake", token="token")
        agent = service.create_agent("controlled")
        peer = service.upsert_peer(account["id"], "remote-user", approved=True)
        service.approve_peer(peer["id"], agent_id=agent["id"])
        adapter = FakeChannel(account["id"])
        await adapter.start()
        service.channels[account["id"]] = adapter
        upload = Path(agent["workspace"]) / "uploads" / "report.txt"
        upload.write_text("report", encoding="utf-8")
        await service.start()
        assert service.control_socket_path.exists()
        assert stat.S_IMODE(service.control_socket_path.stat().st_mode) == 0o600
        client = ControlClient(service.control_socket_path)
        common = {"agent_id": agent["id"], "secret": service.control_secret_for_agent(agent["id"])}

        sent = await asyncio.to_thread(client.request, {**common, "action": "send_file", "relative_path": "uploads/report.txt", "caption": "done"})
        assert sent["delivery"]["ok"] is True
        assert adapter.files == [("remote-user", str(upload.resolve()), "done")]

        try:
            await asyncio.to_thread(client.request, {**common, "action": "send_file", "relative_path": "../outside.txt"})
        except ControlError as error:
            assert "工作区" in str(error)
        else:
            raise AssertionError("workspace traversal was accepted")

        try:
            await asyncio.to_thread(client.request, {**common, "agent_id": "other", "action": "schedule_list"})
        except ControlError as error:
            assert "权限" in str(error)
        else:
            raise AssertionError("capability was accepted for another Agent")

        created = await asyncio.to_thread(client.request, {
            **common,
            "action": "schedule_create",
            "prompt": "remind me",
            "expression_type": "every",
            "expression": "3600",
            "peer_id": peer["id"],
            "timezone": "Asia/Shanghai",
        })
        schedule_id = created["schedule"]["id"]
        listed = await asyncio.to_thread(client.request, {**common, "action": "schedule_list"})
        assert [item["id"] for item in listed["schedules"]] == [schedule_id]
        cancelled = await asyncio.to_thread(client.request, {**common, "action": "schedule_cancel", "schedule_id": schedule_id})
        assert cancelled["cancelled"] is True

        settings = service._agent_settings(service._row("agents", agent["id"]) or {})
        environment = AnthropicProvider(settings)._workspace_cli_environment()
        assert environment["XMAGENTS_AGENT_ID"] == agent["id"]
        assert environment["XMAGENTS_CONTROL_SOCKET"] == str(service.control_socket_path)
        assert environment["XMAGENTS_CONTROL_SECRET"] == common["secret"]
        blocked_env = {**os.environ, **environment}
        listed_from_cli = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-m", "xmagents.main", "schedule", "list"],
            cwd=Path.cwd(),
            env=blocked_env,
            capture_output=True,
            text=True,
        )
        assert listed_from_cli.returncode == 0
        assert schedule_id in listed_from_cli.stdout
        sent_from_cli = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-m", "xmagents.main", "agent-send-file", "uploads/report.txt"],
            cwd=Path.cwd(),
            env=blocked_env,
            capture_output=True,
            text=True,
        )
        assert sent_from_cli.returncode == 0
        assert "已提交" in sent_from_cli.stdout
        assert len(adapter.files) == 2
        blocked = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-m", "xmagents.main", "doctor"],
            cwd=Path.cwd(),
            env=blocked_env,
            capture_output=True,
            text=True,
        )
        assert blocked.returncode == 2
        assert "只允许" in blocked.stderr
        await service.stop()
        assert not service.control_socket_path.exists()


def test_control_socket_scopes_agent_operations() -> None:
    asyncio.run(_scenario())


def test_partial_control_scope_fails_closed(monkeypatch) -> None:
    monkeypatch.delenv("XMAGENTS_CONTROL_SOCKET", raising=False)
    monkeypatch.delenv("XMAGENTS_CONTROL_SECRET", raising=False)
    monkeypatch.setenv("XMAGENTS_AGENT_ID", "agent-only")
    try:
        controlled_environment()
    except ControlError as error:
        assert "不完整" in str(error)
    else:
        raise AssertionError("partial control environment was accepted")
