from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from xmagents.agents.runtime import AgentRuntime, AgentSettings, AnthropicProvider, Provider, RuntimeEvent, TurnContext, parse_command
from xmagents.channels.telegram import TelegramAdapter, _group_triggered, _message_from_update, split_text
from xmagents.channels.wechat import WeChatIlinkAdapter, _aes_ecb_decrypt, _aes_ecb_encrypt
from xmagents.config import AppPaths
from xmagents.files import FileSafetyError, validate_send_file
from xmagents.knowledge import KnowledgeService
from xmagents.main import build_app, main
from xmagents.plugins import PluginLoader
from xmagents.service import AppService


def test_database_auth_and_webui():
    with TemporaryDirectory() as directory:
        service = AppService(AppPaths.from_root(Path.cwd(), directory))
        app = build_app(service)
        with TestClient(app) as client:
            assert client.get("/api/health").json()["configured"] is False
            page = client.get("/")
            csrf = client.cookies.get("xmagents_csrf")
            assert csrf
            assert client.post("/setup", data={
                "password": "1234567890", "password_confirm": "1234567890", "csrf_token": csrf,
            }, follow_redirects=False).status_code == 303
            assert client.post("/login", data={"password": "1234567890", "csrf_token": csrf}, follow_redirects=False).status_code == 303

            def mutation(method, path, **kwargs):
                headers = {"X-XMAgent-CSRF": csrf, **kwargs.pop("headers", {})}
                return getattr(client, method)(path, headers=headers, **kwargs)

            changed = mutation("post", "/api/auth/password", json={
                "current_password": "1234567890",
                "new_password": "abcdefghij",
                "new_password_confirm": "abcdefghij",
            })
            assert changed.status_code == 200
            assert changed.json() == {"ok": True}
            assert client.get("/api/agents").status_code == 200
            assert mutation("post", "/api/auth/password", json={
                "current_password": "1234567890",
                "new_password": "klmnopqrst",
                "new_password_confirm": "klmnopqrst",
            }).status_code == 401
            # Password changes revoke the previous session. Restore a session
            # in the test browser before exercising the remaining management
            # API, while retaining its existing CSRF cookie.
            client.cookies.set("xmagents_session", service.create_session())
            created = mutation("post", "/api/agents", json={"name": "demo"}).json()
            assert created["permission_mode"] == "bypassPermissions"
            assert client.get("/api/agents").json()[0]["name"] == "demo"

            configured = mutation("post", "/api/agents", json={
                "name": "configured",
                "provider": "openai",
                "model": "gpt-test",
                "permission_mode": "plan",
                "effort": "high",
                "memory_enabled": False,
                "knowledge_base_id": "kb-1",
            }).json()
            assert configured["provider"] == "openai"
            assert configured["model"] == "gpt-test"
            assert configured["permission_mode"] == "plan"
            assert configured["effort"] == "high"
            assert configured["memory_enabled"] == 0
            assert configured["knowledge_base_id"] == "kb-1"
            assert configured["config"]["permission_mode"] == "plan"

            updated = mutation("patch", f"/api/agents/{configured['id']}", json={
                "provider": "anthropic",
                "model": "claude-test",
                "permission_mode": "acceptEdits",
                "effort": "max",
                "memory_enabled": True,
                "knowledge_base_id": "kb-2",
                "config": {"system_prompt": "test prompt"},
            }).json()
            assert updated["provider"] == "anthropic"
            assert updated["model"] == "claude-test"
            assert updated["permission_mode"] == "acceptEdits"
            assert updated["effort"] == "max"
            assert updated["memory_enabled"] == 1
            assert updated["knowledge_base_id"] == "kb-2"
            assert updated["config"] == {"system_prompt": "test prompt"}

            listed = {item["id"]: item for item in client.get("/api/agents").json()}
            assert listed[configured["id"]]["effort"] == "max"
            assert listed[configured["id"]]["config"]["system_prompt"] == "test prompt"
            assert mutation("delete", f"/api/agents/{configured['id']}").json() == {"ok": True}
            assert configured["id"] not in {item["id"] for item in client.get("/api/agents").json()}
            assert mutation("patch", "/api/agents/missing", json={"effort": "low"}).status_code == 404
            assert mutation("patch", "/api/plugins/not-installed", json={"enabled": True}).status_code == 404

            first = mutation("post", f"/api/memory/{created['id']}", json={"content": "private memory"}).json()
            second_agent = mutation("post", "/api/agents", json={"name": "other"}).json()
            assert mutation("delete", f"/api/memory/{second_agent['id']}/{first['id']}").status_code == 404
            assert client.get(f"/api/memory/{created['id']}").json()[0]["id"] == first["id"]

            channel = mutation("post", "/api/channels", json={"channel": "telegram", "name": "remove", "token": "test"}).json()
            assert mutation("delete", f"/api/channels/{channel['id']}").json() == {"ok": True}
            assert mutation("delete", f"/api/channels/{channel['id']}").status_code == 404


def test_webui_setup_login_and_mutation_require_csrf():
    with TemporaryDirectory() as directory:
        service = AppService(AppPaths.from_root(Path.cwd(), directory))
        app = build_app(service)
        with TestClient(app) as client:
            # Setup is protected even before a password exists. A page visit
            # supplies the matching browser cookie and hidden form value.
            assert client.post("/setup", data={"password": "1234567890", "password_confirm": "1234567890"}).status_code == 422
            page = client.get("/")
            assert page.status_code == 200
            csrf = client.cookies.get("xmagents_csrf")
            assert csrf and f'name="csrf_token" value="{csrf}"' in page.text
            setup = client.post("/setup", data={
                "password": "1234567890",
                "password_confirm": "1234567890",
                "csrf_token": csrf,
            }, follow_redirects=False)
            assert setup.status_code == 303
            assert client.cookies.get("xmagents_session")

            # A configured browser session refuses state changes without the
            # header, then accepts the dashboard's custom header.
            assert client.post("/api/agents", json={"name": "blocked"}).status_code == 403
            created = client.post("/api/agents", json={"name": "allowed"}, headers={"X-XMAgent-CSRF": csrf})
            assert created.status_code == 200
            assert created.json()["name"] == "allowed"
            assert client.post("/logout", data={}).status_code == 422
            assert client.post("/logout", data={"csrf_token": csrf}, follow_redirects=False).status_code == 303


def test_database_wal_sidecars_are_owner_only():
    with TemporaryDirectory() as directory:
        service = AppService(AppPaths.from_root(Path.cwd(), Path(directory) / "data"))
        service.db.execute("INSERT INTO settings(key,value,updated_at) VALUES('mode-check','true','now')")
        for suffix in ("", "-wal", "-shm"):
            path = service.db.path.with_name(service.db.path.name + suffix)
            if path.exists():
                assert path.stat().st_mode & 0o077 == 0


def test_serve_defaults_to_dual_stack_host(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_serve(_service, args):
        seen["host"] = args.host
        return 0

    monkeypatch.setattr("xmagents.main.command_serve", fake_serve)
    assert main(["serve"]) == 0
    assert seen["host"] == "both"


def test_plugins_and_commands():
    class FakeProvider(Provider):
        async def stream(self, prompt, history):
            yield RuntimeEvent("text", "ok")

    async def run():
        settings = AgentSettings(agent_id="agent-1")
        runtime = AgentRuntime(settings, provider=FakeProvider(settings), plugin_loader=PluginLoader(Path.cwd() / "plugins"))
        assert (await runtime.submit("/test text1"))[0].content == "输入了text1"
        assert (await runtime.submit("/effort high"))[0].content == "思考强度已设置为 high。"
        assert (await runtime.submit("/status"))[0].content.find("effort: high") >= 0
        await runtime.close()

    asyncio.run(run())


def test_channel_helpers_and_group_gate():
    assert parse_command(" /test hello ") == ("test", "hello")
    assert len(split_text("x" * 9000, 4096)) == 3
    key = b"0123456789abcdef"
    assert _aes_ecb_decrypt(_aes_ecb_encrypt(b"payload", key), key) == b"payload"
    update = {"update_id": 1, "message": {"message_id": 2, "chat": {"id": -1, "type": "group"}, "from": {"id": 3}, "text": "hello"}}
    message = _message_from_update(update, "account")
    assert message and message.kind == "group"
    assert not _group_triggered(update["message"], {"id": 99, "username": "bot"})
    mention = {**update["message"], "text": "@bot hi", "entities": [{"type": "mention", "offset": 0, "length": 4}]}
    assert _group_triggered(mention, {"id": 99, "username": "bot"})
    emoji_mention = {**update["message"], "text": "😀 @bot hi", "entities": [{"type": "mention", "offset": 3, "length": 4}]}
    assert _group_triggered(emoji_mention, {"id": 99, "username": "bot"})
    assert not _group_triggered({**update["message"], "text": "/status@another_bot"}, {"id": 99, "username": "bot"})


def test_telegram_group_command_suffix_is_normalized_before_runtime() -> None:
    async def run() -> None:
        adapter = TelegramAdapter("account", "123456:abcdefghijklmnopqrstuv", group_allowlist=["-1"])
        adapter.bot = {"id": 99, "username": "bot"}

        async def updates(*_args, **_kwargs):
            return [{
                "update_id": 1,
                "message": {
                    "message_id": 2,
                    "chat": {"id": -1, "type": "group"},
                    "from": {"id": 3},
                    "text": "/help@bot extra",
                },
            }]

        adapter.get_updates = updates  # type: ignore[method-assign]
        messages = await adapter.poll_once()
        assert [message.text for message in messages] == ["/help extra"]

    asyncio.run(run())


def test_knowledge_and_workspace_safety():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.sqlite3"
        with sqlite3.connect(source) as connection:
            connection.execute("CREATE TABLE documents(id TEXT PRIMARY KEY,title TEXT,content TEXT NOT NULL,metadata_json TEXT,updated_at TEXT)")
            connection.execute("INSERT INTO documents VALUES('1','Guide','XMAgent knowledge','{}','2026-08-12T00:00:00Z')")
        service = AppService(AppPaths.from_root(Path.cwd(), root / "data"))
        imported = KnowledgeService(service).import_database(source, "guide")
        assert KnowledgeService(service).search(imported["id"], "knowledge")[0]["id"] == "1"
        agent = service.create_agent("safe")
        path = Path(agent["workspace"]) / "uploads" / "ok.txt"
        path.write_text("ok")
        assert validate_send_file(agent["workspace"], "uploads/ok.txt") == path.resolve()
        try:
            validate_send_file(agent["workspace"], "../source.sqlite3")
        except FileSafetyError:
            pass
        else:
            raise AssertionError("workspace traversal was accepted")

        invalid = root / "invalid.sqlite3"
        with sqlite3.connect(invalid) as connection:
            connection.execute("CREATE TABLE documents(id TEXT PRIMARY KEY,content TEXT NOT NULL)")
        try:
            KnowledgeService(service).import_database(invalid, "invalid")
        except ValueError as error:
            assert "title" in str(error)
        else:
            raise AssertionError("incomplete knowledge schema was accepted")


def test_anthropic_history_restore_and_stream_deduplication():
    class ResultMessage:
        result = ""
        subtype = "success"
        is_error = False
        session_id = "session"
        usage = None
        total_cost_usd = None
        errors = None

    class FakeClaudeClient:
        def __init__(self):
            self.connected = 0
            self.queries: list[str] = []

        async def connect(self):
            self.connected += 1

        async def query(self, prompt):
            self.queries.append(prompt)

        async def receive_response(self):
            yield ResultMessage()

        async def disconnect(self):
            pass

    class StreamThenFinalProvider(Provider):
        async def stream(self, prompt, history):
            yield RuntimeEvent("text", "partial ", {"stream": True})
            yield RuntimeEvent("text", "complete")

    async def run():
        settings = AgentSettings(agent_id="agent-restore")
        client = FakeClaudeClient()
        provider = AnthropicProvider(settings, sdk_client=client)
        history = [{"role": "user", "content": "earlier question"}, {"role": "assistant", "content": "earlier answer"}]
        await anext(provider.stream("next question", history))
        assert "earlier question" in client.queries[-1]
        await anext(provider.stream("later question", history))
        assert "<context>" not in client.queries[-1]

        runtime = AgentRuntime(AgentSettings(agent_id="agent-stream"), provider=StreamThenFinalProvider(AgentSettings()))
        await runtime.submit("hello")
        assert runtime.history[-1]["content"] == "complete"
        await runtime.close()

    asyncio.run(run())


def test_conversation_session_is_persisted_and_reused():
    class SessionProvider(Provider):
        def __init__(self, settings):
            super().__init__(settings)
            self.session_ids: list[str] = []

        async def stream(self, prompt, history, *, session_id="default"):
            self.session_ids.append(session_id)
            yield RuntimeEvent("text", "reply")
            yield RuntimeEvent("result", metadata={"session_id": "sdk-session-42"})

    async def run():
        with TemporaryDirectory() as directory:
            service = AppService(AppPaths.from_root(Path.cwd(), Path(directory) / "data"))
            agent = service.create_agent("sessions")
            runtime = service.runtime_for(agent["id"])
            provider = SessionProvider(runtime.settings)
            runtime.provider = provider

            await service.chat_agent(agent["id"], "first", peer_id="webui-peer")
            row = service.db.fetchone(
                "SELECT session_id FROM conversations WHERE agent_id=? AND route_key=?",
                (agent["id"], "webui-peer"),
            )
            assert row and row["session_id"] == "sdk-session-42"

            await service.chat_agent(agent["id"], "second", peer_id="webui-peer")
            assert provider.session_ids == ["default", "sdk-session-42"]

            await service.chat_agent(agent["id"], "/new", peer_id="webui-peer")
            cleared = service.db.fetchone(
                "SELECT session_id FROM conversations WHERE agent_id=? AND route_key=?",
                (agent["id"], "webui-peer"),
            )
            assert cleared and cleared["session_id"] is None
            await runtime.close()

    asyncio.run(run())


def test_effort_persistence_keeps_the_current_runtime_alive():
    class FakeProvider(Provider):
        async def stream(self, prompt, history):
            yield RuntimeEvent("text", "ok")

    async def run():
        with TemporaryDirectory() as directory:
            service = AppService(AppPaths.from_root(Path.cwd(), Path(directory) / "data"))
            agent = service.create_agent("effort")
            runtime = service.runtime_for(agent["id"])
            runtime.provider = FakeProvider(runtime.settings)

            result = await runtime.submit("/effort high")

            assert result[0].content == "思考强度已设置为 high。"
            assert service._runtimes[agent["id"]] is runtime
            assert runtime.mailbox._closed is False
            assert "effort: high" in (await runtime.submit("/status"))[0].content
            row = service._row("agents", agent["id"])
            assert row and row["effort"] == "high"
            await runtime.close()

    asyncio.run(run())


def test_api_profile_update_and_delete_invalidate_bound_runtime():
    with TemporaryDirectory() as directory:
        service = AppService(AppPaths.from_root(Path.cwd(), Path(directory) / "data"))
        profile = service.create_api_profile(name="provider", provider="openai", secret="old-key")
        agent = service.create_agent("bound", provider="openai", api_profile_id=profile["id"])

        original = service.runtime_for(agent["id"])
        service.update_api_profile(profile["id"], {"api_key": "new-key"})
        assert agent["id"] not in service._runtimes
        refreshed = service.runtime_for(agent["id"])
        assert refreshed is not original
        assert refreshed.settings.api_key == "new-key"

        service.delete_api_profile(profile["id"])
        assert agent["id"] not in service._runtimes


def test_memory_routes_reject_cross_agent_access():
    with TemporaryDirectory() as directory:
        service = AppService(AppPaths.from_root(Path.cwd(), Path(directory) / "data"))
        service.initialize_admin("1234567890")
        app = build_app(service)
        first = service.create_agent("first")
        second = service.create_agent("second")
        from xmagents.memory import MemoryStore
        memory_id = MemoryStore(service).add(first["id"], "isolated")
        with TestClient(app) as client:
            client.cookies.set("xmagents_session", service.create_session())
            client.cookies.set("xmagents_csrf", "test-csrf")
            assert client.get(f"/api/memory/{second['id']}").status_code == 200
            headers = {"X-XMAgent-CSRF": "test-csrf"}
            assert client.delete(f"/api/memory/{second['id']}/{memory_id}", headers=headers).status_code == 404
            assert client.patch(f"/api/memory/{second['id']}/{memory_id}", json={"content": "overwrite"}, headers=headers).status_code == 404
