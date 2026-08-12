from __future__ import annotations

import asyncio
from pathlib import Path
import uuid

import pytest
from fastapi.testclient import TestClient

from xmagents.config import AppPaths
from xmagents.main import build_app
from xmagents.service import AppService
from xmagents.agents.runtime import AnthropicProvider


def _service(tmp_path: Path) -> AppService:
    return AppService(AppPaths.from_root(tmp_path, tmp_path / "data"))


@pytest.mark.asyncio
async def test_wechat_qr_confirmation_creates_one_channel_when_polls_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions: list[FakeQRSession] = []

    class FakeQRSession:
        def __init__(self, base_url: str):
            self.base_url = base_url
            self.closed = 0
            sessions.append(self)

        async def start(self) -> dict[str, str]:
            return {"state": "waiting", "base_url": self.base_url}

        async def poll_once(self) -> dict[str, str]:
            await asyncio.sleep(0.01)
            return {"state": "confirmed", "bot_token": "wechat-token", "base_url": self.base_url}

        async def close(self) -> None:
            self.closed += 1

    monkeypatch.setattr("xmagents.channels.wechat.WeChatQRLoginSession", FakeQRSession)
    service = _service(tmp_path)
    started: list[str] = []

    async def start_channel(account_id: str) -> dict[str, object]:
        started.append(account_id)
        return service._row("channel_accounts", account_id) or {}

    service.start_channel = start_channel  # type: ignore[method-assign]
    login = await service.begin_wechat_qr()

    first, second, third = await asyncio.gather(*[service.poll_wechat_qr(login["login_id"]) for _ in range(3)])

    assert {item["account_id"] for item in (first, second, third)} == {first["account_id"]}
    assert "bot_token" not in first
    assert started == [first["account_id"]]
    assert service.db.fetchone("SELECT COUNT(*) AS count FROM channel_accounts WHERE channel='wechat'")["count"] == 1
    assert sessions[0].closed == 1
    assert (await service.poll_wechat_qr(login["login_id"]))["account_id"] == first["account_id"]


def test_peer_binding_rebinds_one_route_and_exposes_channel_context(tmp_path: Path) -> None:
    service = _service(tmp_path)
    account = service.create_channel("telegram", "support-bot", token="token")
    first_agent = service.create_agent("客服")
    second_agent = service.create_agent("售后")
    peer = service.upsert_peer(account["id"], "12345", display_name="Alice", approved=False)

    first = service.bind_peer(peer["id"], first_agent["id"])

    assert first["agent_name"] == "客服"
    assert first["account_name"] == "support-bot"
    assert first["channel"] == "telegram"
    assert first["display_name"] == "Alice"
    assert first["approved"] == 1
    assert service.binding_for_peer(peer["id"])["id"] == first_agent["id"]

    rebound = service.bind_peer(peer["id"], second_agent["id"])

    assert rebound["id"] == first["id"]
    assert rebound["agent_id"] == second_agent["id"]
    assert service.binding_for_peer(peer["id"])["id"] == second_agent["id"]
    assert len(service.list_bindings(peer_id=peer["id"])) == 1
    assert service.unbind_peer(peer["id"])
    assert service.binding_for_peer(peer["id"]) is None


def test_binding_api_lists_rebinds_and_unbinds(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.initialize_admin("1234567890")
    account = service.create_channel("telegram", "ops-bot", token="token")
    first_agent = service.create_agent("一号")
    second_agent = service.create_agent("二号")
    peer = service.upsert_peer(account["id"], "remote-user")
    service.bind_peer(peer["id"], first_agent["id"])
    # macOS has a small AF_UNIX pathname limit; pytest's generated path can
    # exceed it before the TestClient lifespan starts the control socket.
    service.control_socket_path = Path("/tmp") / f"xma-routing-{uuid.uuid4().hex[:12]}.sock"
    csrf = "test-csrf-token"

    with TestClient(build_app(service)) as client:
        client.cookies.set("xmagents_session", service.create_session())
        client.cookies.set("xmagents_csrf", csrf)
        headers = {"X-XMAgent-CSRF": csrf}

        listed = client.get("/api/bindings").json()
        assert [(item["peer_id"], item["agent_id"]) for item in listed] == [(peer["id"], first_agent["id"])]

        rebound = client.post(f"/api/bindings/{peer['id']}", json={"agent_id": second_agent["id"]}, headers=headers)
        assert rebound.status_code == 200
        assert rebound.json()["agent_id"] == second_agent["id"]

        removed = client.delete(f"/api/bindings/{peer['id']}", headers=headers)
        assert removed.json() == {"ok": True}
        assert client.get("/api/bindings").json() == []

        invalid_name = client.post("/api/agents", json={"name": "bad/name"}, headers=headers)
        assert invalid_name.status_code == 400
        duplicate_name = client.post("/api/agents", json={"name": "一号"}, headers=headers)
        assert duplicate_name.status_code == 400


def test_agent_workspace_uses_unique_name_and_moves_on_rename(tmp_path: Path) -> None:
    service = _service(tmp_path)
    agent = service.create_agent("我的 Agent")
    workspace = tmp_path / "data" / "workspaces" / "我的 Agent"

    assert Path(agent["workspace"]) == workspace
    assert (workspace / "uploads").is_dir()
    (workspace / "uploads" / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="名称已存在"):
        service.create_agent("我的 agent")

    updated = service.update_agent(agent["id"], {"name": "新的 Agent"})
    renamed = tmp_path / "data" / "workspaces" / "新的 Agent"

    assert Path(updated["workspace"]) == renamed
    assert (renamed / "uploads" / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not workspace.exists()
    with pytest.raises(ValueError, match="路径分隔符"):
        service.create_agent("../not-a-workspace")


def test_builtin_claude_code_profile_uses_local_login_without_api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    profiles = service.list_api_profiles()
    builtin = next(profile for profile in profiles if profile["id"] == "builtin-claude-code")

    assert builtin["builtin"] is True
    assert builtin["secret_configured"] is False
    with pytest.raises(ValueError, match="不能编辑"):
        service.update_api_profile("builtin-claude-code", {"name": "other"})
    with pytest.raises(ValueError, match="不能删除"):
        service.delete_api_profile("builtin-claude-code")

    agent = service.create_agent("本机登录", api_profile_id="builtin-claude-code")
    settings = service._agent_settings(service._row("agents", agent["id"]) or {})
    assert settings.extra["use_local_claude_code_login"] is True
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "should-not-be-injected")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://should-not-be-injected.example")

    class FakeOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSDK:
        ClaudeAgentOptions = FakeOptions

    provider = AnthropicProvider(settings)
    provider._sdk_module = FakeSDK
    options = provider._options()
    env = options.kwargs.get("env", {})
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "ANTHROPIC_BASE_URL" not in env
    assert "CLAUDE_CONFIG_DIR" not in env
    assert not (Path(agent["workspace"]) / ".claude-config").exists()


def test_dashboard_counts_all_inbox_messages_and_returns_recent_summaries(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.initialize_admin("1234567890")
    account = service.create_channel("telegram", "metrics-bot", token="token")
    peer = service.upsert_peer(account["id"], "inbox-user", display_name="Metrics user", approved=True)
    first_id = uuid.uuid4().hex
    second_id = uuid.uuid4().hex
    service.db.execute(
        "INSERT INTO inbox(id,account_id,external_event_id,peer_id,payload_json,state,received_at) VALUES(?,?,?,?,?,?,?)",
        (first_id, account["id"], "evt-1", peer["external_id"], service.db.json({"text": "first inbound"}), "processed", "2026-08-12T00:00:00+00:00"),
    )
    service.db.execute(
        "INSERT INTO inbox(id,account_id,external_event_id,peer_id,payload_json,state,received_at) VALUES(?,?,?,?,?,?,?)",
        (second_id, account["id"], "evt-2", peer["external_id"], service.db.json({"text": "second inbound"}), "received", "2026-08-12T00:01:00+00:00"),
    )
    overview = service.inbox_overview(limit=8)
    assert overview["total"] == 2
    assert overview["pending"] == 1
    assert [item["summary"] for item in overview["recent"]] == ["second inbound", "first inbound"]

    service.control_socket_path = Path("/tmp") / f"xma-dashboard-{uuid.uuid4().hex[:12]}.sock"
    csrf = "dashboard-csrf"
    with TestClient(build_app(service)) as client:
        client.cookies.set("xmagents_session", service.create_session())
        client.cookies.set("xmagents_csrf", csrf)
        payload = client.get("/api/dashboard").json()
    assert payload["inbox_total"] == 2
    assert payload["inbox_pending"] == 1
    assert payload["inbox"][0]["summary"] == "second inbound"


def test_agent_create_api_returns_validation_error_for_duplicate_name(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.initialize_admin("1234567890")
    service.create_agent("already-here")
    service.control_socket_path = Path("/tmp") / f"xma-agent-name-{uuid.uuid4().hex[:12]}.sock"
    csrf = "agent-name-csrf"
    with TestClient(build_app(service)) as client:
        client.cookies.set("xmagents_session", service.create_session())
        client.cookies.set("xmagents_csrf", csrf)
        response = client.post("/api/agents", json={"name": "Already-Here"}, headers={"X-XMAgent-CSRF": csrf})
    assert response.status_code == 400
    assert "名称已存在" in response.json()["detail"]
