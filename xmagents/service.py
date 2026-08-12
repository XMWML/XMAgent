"""Application orchestration shared by the WebUI, channels, CLI and scheduler."""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
import inspect
import asyncio
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import AppPaths
from .database import Database, utcnow
from .security import hash_password, new_session_token, session_expiry, verify_password
from .agents.runtime import AgentRuntime, AgentSettings, RuntimeEvent, TurnContext
from .memory import KnowledgeContext, MemoryStore
from .plugins import PluginLoader
from .scheduler import Scheduler
from .channels.telegram import TelegramAdapter
from .channels.wechat import WeChatIlinkAdapter
from .channels.redaction import redact_mapping, redact_secret, redact_text
from .control import ControlError, ControlServer, derive_agent_secret
from .files import FileSafetyError, validate_send_file
from .models import DeliveryResult


OUTBOX_LEASE_SECONDS = 120
OUTBOX_MAX_ATTEMPTS = 8
OUTBOX_IDLE_SECONDS = 1.0


class AppService:
    def __init__(self, paths: AppPaths | None = None, database: Database | None = None):
        self.paths = paths or AppPaths.from_root()
        self.paths.ensure()
        self.db = database or Database(self.paths.database)
        self.db.initialize()
        self.runtime_manager: Any = None
        self.scheduler: Any = Scheduler(self, self._run_schedule)
        self.channels: dict[str, Any] = {}
        self.plugin_loader = PluginLoader(self.paths.root / "plugins")
        self._runtimes: dict[str, AgentRuntime] = {}
        self._channel_tasks: dict[str, asyncio.Task[Any]] = {}
        self._qr_sessions: dict[str, Any] = {}
        self._outbox_task: asyncio.Task[Any] | None = None
        self._outbox_wakeup = asyncio.Event()
        # Capability values intentionally live only for this service process.
        # A fresh service restart invalidates any secret inherited by an old
        # Claude subprocess, so it cannot keep using a replacement service.
        self._control_master_secret = secrets.token_bytes(32)
        self.control_socket_path = self.paths.runtime / "control.sock"
        self.control_server: ControlServer | None = None

    async def start(self) -> None:
        self.restore_plugins()
        self._recover_outbox_leases()
        if self.control_server is None:
            server = ControlServer(self.control_socket_path, self._handle_control_request)
            try:
                await server.start()
            except Exception:
                await server.stop()
                raise
            self.control_server = server
        try:
            if self._outbox_task is None or self._outbox_task.done():
                self._outbox_wakeup.clear()
                self._outbox_task = asyncio.create_task(self._run_outbox(), name="xmagents-outbox")
            await self.scheduler.start()
            # Only accounts explicitly marked running are resumed after a restart.
            # This prevents a partially configured token from causing a retry loop.
            for row in self.db.fetchall("SELECT * FROM channel_accounts WHERE status='running'"):
                try:
                    await self.start_channel(row["id"])
                except Exception as error:
                    self.update_channel(row["id"], {"status": "error"})
                    self.db.audit("channel_start_failed", target=row["id"], detail={"error": redact_text(error)})
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        # Stop polling before cancelling deliveries; a running inbound turn may
        # enqueue a response, while an outbox lease is intentionally retained
        # as ``unknown`` if shutdown races a provider request.
        if self.control_server is not None:
            await self.control_server.stop()
            self.control_server = None
        for task in self._channel_tasks.values():
            task.cancel()
        for task in self._channel_tasks.values():
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._channel_tasks.clear()
        if self._outbox_task:
            self._outbox_task.cancel()
            try:
                await self._outbox_task
            except asyncio.CancelledError:
                pass
            self._outbox_task = None
        for adapter in list(self.channels.values()):
            try:
                await adapter.stop()
            except Exception as error:
                self.db.audit("channel_stop_failed", detail={"error": redact_text(error)})
        self.channels.clear()
        for session in list(self._qr_sessions.values()):
            try:
                await session.close()
            except Exception:
                pass
        self._qr_sessions.clear()
        for runtime in list(self._runtimes.values()):
            await runtime.close()
        self._runtimes.clear()
        await self.scheduler.stop()

    def control_secret_for_agent(self, agent_id: str) -> str:
        """Return the current-process local capability for one Agent."""

        return derive_agent_secret(self._control_master_secret, str(agent_id))

    def _control_workspace(self, agent: dict[str, Any]) -> Path:
        """Resolve an Agent workspace while rejecting symlinked escape roots."""

        try:
            workspace = Path(str(agent["workspace"])).resolve(strict=True)
            workspaces_root = self.paths.workspaces.resolve(strict=True)
        except (KeyError, FileNotFoundError, OSError) as error:
            raise ControlError("Agent 工作区不可用") from error
        if workspace == workspaces_root or workspaces_root not in workspace.parents:
            raise ControlError("Agent 工作区不在 XMAgent 管理目录内")
        return workspace

    def _control_peer(self, agent_id: str, peer_id: Any = None) -> dict[str, Any]:
        """Resolve a currently active, approved binding for a scoped Agent."""

        rows = [dict(row) for row in self.db.fetchall(
            "SELECT p.* FROM remote_peers p JOIN agent_bindings b ON b.peer_id=p.id "
            "WHERE b.agent_id=? AND b.active=1 AND p.approved=1 ORDER BY p.created_at",
            (agent_id,),
        )]
        requested = str(peer_id or "").strip()
        if requested:
            for row in rows:
                if str(row["id"]) == requested:
                    return row
            raise ControlError("渠道用户未绑定到此 Agent")
        if not rows:
            raise ControlError("Agent 没有已批准的渠道用户绑定")
        if len(rows) != 1:
            raise ControlError("Agent 绑定多个渠道用户，请指定 peer_id")
        return rows[0]

    def _control_context_token(self, agent_id: str, peer: dict[str, Any]) -> str | None:
        route_key = str(peer.get("chat_id") or peer.get("external_id") or "")
        row = self.db.fetchone(
            "SELECT context_token FROM conversations WHERE agent_id=? AND route_key=?",
            (agent_id, route_key),
        )
        return str(row["context_token"]) if row and row["context_token"] else None

    async def _handle_control_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Authorize a workspace CLI request and execute its narrow action."""

        agent_id = str(payload.get("agent_id") or "").strip()
        supplied_secret = str(payload.get("secret") or "")
        if not agent_id or not supplied_secret:
            raise ControlError("控制请求缺少 Agent 身份")
        if not hmac.compare_digest(supplied_secret, self.control_secret_for_agent(agent_id)):
            raise ControlError("控制请求没有权限")
        agent = self._row("agents", agent_id)
        if not agent:
            raise ControlError("Agent 不存在或已删除")
        action = str(payload.get("action") or "")
        try:
            if action == "send_file":
                relative_path = payload.get("relative_path", payload.get("path"))
                if not isinstance(relative_path, str) or not relative_path.strip():
                    raise ControlError("文件路径不能为空")
                if len(relative_path) > 4096:
                    raise ControlError("文件路径过长")
                caption = payload.get("caption")
                if caption is not None and not isinstance(caption, str):
                    raise ControlError("文件说明必须是文本")
                if isinstance(caption, str) and len(caption) > 4096:
                    raise ControlError("文件说明过长")
                workspace = self._control_workspace(agent)
                # Validate before enqueueing, including realpath containment,
                # symlink escape and the global attachment byte limit.
                path = validate_send_file(workspace, relative_path)
                peer = self._control_peer(agent_id, payload.get("peer_id"))
                context_token = self._control_context_token(agent_id, peer)
                result = await self.send_file(
                    str(peer["account_id"]),
                    str(peer.get("chat_id") or peer["external_id"]),
                    str(workspace),
                    str(path.relative_to(workspace)),
                    caption=caption,
                    context_token=context_token,
                )
                self.db.audit("control_send_file", target=agent_id, detail={"peer_id": peer["id"], "path": str(path.relative_to(workspace))})
                return {"delivery": result.to_dict(), "peer_id": str(peer["id"])}

            if action == "schedule_create":
                prompt = payload.get("prompt")
                expression = payload.get("expression")
                expression_type = payload.get("expression_type")
                timezone = payload.get("timezone", "Asia/Shanghai")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ControlError("任务内容不能为空")
                if len(prompt) > 16_000:
                    raise ControlError("任务内容过长")
                if not isinstance(expression, str) or not expression.strip():
                    raise ControlError("定时表达式不能为空")
                if len(expression) > 512:
                    raise ControlError("定时表达式过长")
                if expression_type not in {"at", "every", "cron"}:
                    raise ControlError("定时类型必须是 at、every 或 cron")
                if not isinstance(timezone, str) or len(timezone) > 128:
                    raise ControlError("时区无效")
                peer = self._control_peer(agent_id, payload.get("peer_id"))
                schedule = self.scheduler.create(
                    agent_id,
                    prompt.strip(),
                    expression.strip(),
                    str(expression_type),
                    peer_id=str(peer["id"]),
                    timezone=timezone,
                )
                self.db.audit("control_schedule_created", target=str(schedule["id"]), detail={"agent_id": agent_id, "peer_id": peer["id"]})
                return {"schedule": schedule}

            if action == "schedule_list":
                return {"schedules": self.scheduler.list(agent_id)}

            if action == "schedule_cancel":
                schedule_id = payload.get("schedule_id")
                if not isinstance(schedule_id, str) or not schedule_id.strip():
                    raise ControlError("任务 ID 不能为空")
                row = self._row("schedules", schedule_id)
                if not row or str(row["agent_id"]) != agent_id:
                    raise ControlError("定时任务不存在或无权访问")
                self.scheduler.cancel(schedule_id)
                self.db.audit("control_schedule_cancelled", target=schedule_id, detail={"agent_id": agent_id})
                return {"schedule_id": schedule_id, "cancelled": True}
        except ControlError:
            raise
        except (FileSafetyError, ValueError) as error:
            raise ControlError(str(error)) from error
        raise ControlError("不支持的控制操作")

    def _build_adapter(self, row: dict[str, Any]) -> Any:
        config = self.db.loads(row.get("config_json"), {})
        if row["channel"] == "telegram":
            return TelegramAdapter(
                row["id"], row.get("token") or "", proxy=row.get("proxy"),
                base_url=row.get("base_url") or "https://api.telegram.org",
                poll_timeout=int(config.get("poll_timeout", 25)),
                group_allowlist=config.get("group_allowlist", []),
            )
        if row["channel"] == "wechat":
            return WeChatIlinkAdapter(
                row["id"], row.get("token") or "", base_url=row.get("base_url") or "https://ilinkai.weixin.qq.com",
                min_send_interval=float(config.get("min_send_interval", 1.5)),
                max_text_length=int(config.get("max_text_length", 2000)), cursor=row.get("cursor") or "",
            )
        raise ValueError(f"不支持的渠道：{row['channel']}")

    async def start_channel(self, account_id: str) -> dict[str, Any]:
        row = self._row("channel_accounts", account_id)
        if not row:
            raise KeyError(account_id)
        if not row.get("token"):
            raise ValueError("渠道尚未配置 token")
        if account_id in self.channels:
            # A failed poll leaves the adapter object present so outbox
            # delivery can still see it.  Starting an errored account must
            # replace that stale adapter and restart its polling task.
            task = self._channel_tasks.get(account_id)
            if str(row.get("status") or "") == "running" and task is not None and not task.done():
                return row
            await self.stop_channel(account_id)
            row = self._row("channel_accounts", account_id) or row
        adapter = self._build_adapter(row)
        await adapter.start()
        self.channels[account_id] = adapter
        self.update_channel(account_id, {"status": "running"})
        self._channel_tasks[account_id] = asyncio.create_task(self._poll_channel(account_id, adapter), name=f"xmagents-channel-{account_id}")
        self._wake_outbox()
        return self._row("channel_accounts", account_id) or row

    async def stop_channel(self, account_id: str) -> None:
        task = self._channel_tasks.pop(account_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        adapter = self.channels.pop(account_id, None)
        if adapter:
            await adapter.stop()
        if self._row("channel_accounts", account_id):
            self.update_channel(account_id, {"status": "stopped"})

    async def _dispatch_runtime_event(self, account_id: str, peer_id: str, event: RuntimeEvent,
                                      *, context_token: str | None) -> None:
        """Forward concise live tool activity without duplicating final text.

        Telegram's edit API can stream assistant text, but the runtime emits
        both deltas and a final complete message.  The durable outbox carries
        the final response, while this callback relays only tool state to both
        channels.  It provides real-time visibility without risking duplicate
        content or a second concurrent iLink text stream.
        """

        if event.kind != "tool":
            return
        names: list[str] = []
        for item in event.metadata.get("tools", []) or []:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
        if not names and event.content:
            names = [str(event.content)]
        if not names:
            return
        await self.send_message(account_id, peer_id, "正在使用工具：" + "、".join(names[:3]), context_token=context_token)

    async def _poll_channel(self, account_id: str, adapter: Any) -> None:
        cursor = (self._row("channel_accounts", account_id) or {}).get("cursor", "")
        while True:
            try:
                messages = await adapter.poll_once(cursor)
                # Telegram gives a monotonic offset for each update. iLink's
                # cursor instead acknowledges a complete response batch, so
                # saving it after the first message could lose later messages
                # if their handling fails. Concrete adapters declare which
                # form they provide; unknown adapters retain the safer common
                # per-message behaviour used by the original contract.
                checkpoint_per_message = bool(getattr(adapter, "cursor_checkpoint_per_message", True))
                batch_cursor = cursor
                for message in messages:
                    raw_next_cursor = getattr(message, "metadata", {}).get("next_cursor", cursor)
                    next_cursor = str(raw_next_cursor) if raw_next_cursor not in (None, "") else str(cursor or "")
                    if self._record_inbox(account_id, message):
                        try:
                            await self.handle_incoming(message)
                        except Exception as error:
                            self._mark_inbox_failed(account_id, message.external_id, error)
                            raise
                        self._mark_inbox_processed(account_id, message.external_id)
                    batch_cursor = next_cursor
                    if checkpoint_per_message:
                        cursor = next_cursor
                        self.update_channel(account_id, {"cursor": cursor})
                if messages and not checkpoint_per_message:
                    # Prefer the adapter's opaque response cursor. Falling
                    # back to message metadata keeps test/downgraded adapters
                    # compatible while preserving batch acknowledgement.
                    adapter_cursor = getattr(adapter, "cursor", None)
                    if adapter_cursor in (None, ""):
                        adapter_cursor = batch_cursor
                    if adapter_cursor not in (None, ""):
                        cursor = str(adapter_cursor)
                        self.update_channel(account_id, {"cursor": cursor})
                # Adapters also advance their cursor for updates filtered out
                # by a channel gate (for example Telegram group chatter) or
                # provider-only events. Persist it even when no user message
                # reached the Agent, otherwise every restart redownloads the
                # same ignored update batch.
                adapter_cursor = getattr(adapter, "cursor", None)
                if adapter_cursor in (None, ""):
                    adapter_cursor = getattr(adapter, "offset", None)
                # An opaque batch cursor (iLink) must not be persisted after
                # a mid-batch failure. Its response object has already moved
                # to the batch's end, whereas ``cursor`` still points at the
                # last durable acknowledgement.
                if adapter_cursor not in (None, "") and (checkpoint_per_message or not messages):
                    adapter_cursor = str(adapter_cursor)
                    if adapter_cursor != cursor:
                        cursor = adapter_cursor
                        self.update_channel(account_id, {"cursor": cursor})
                if not messages:
                    await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.update_channel(account_id, {"status": "error"})
                self.db.audit("channel_poll_failed", target=account_id, detail={"error": redact_text(error)})
                await asyncio.sleep(2)

    async def _reload_channel(self, account_id: str, was_running: bool) -> None:
        """Replace a live adapter after credentials or transport settings change."""

        try:
            await self.stop_channel(account_id)
            if was_running and self._row("channel_accounts", account_id):
                await self.start_channel(account_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._row("channel_accounts", account_id):
                self.update_channel(account_id, {"status": "error"})
            self.db.audit("channel_reload_failed", target=account_id, detail={"error": redact_text(error)})

    def _peer_id_for_message(self, message: Any) -> str:
        row = self.db.fetchone("SELECT id FROM remote_peers WHERE account_id=? AND external_id=? ORDER BY created_at DESC LIMIT 1", (message.account_id, message.peer_id))
        return row["id"] if row else ""

    async def begin_wechat_qr(self, base_url: str | None = None) -> dict[str, Any]:
        from .channels.wechat import WeChatQRLoginSession

        session = WeChatQRLoginSession(base_url or "https://ilinkai.weixin.qq.com")
        snapshot = await session.start()
        login_id = uuid.uuid4().hex
        self._qr_sessions[login_id] = session
        snapshot["login_id"] = login_id
        return snapshot

    async def poll_wechat_qr(self, login_id: str) -> dict[str, Any]:
        session = self._qr_sessions.get(login_id)
        if not session:
            raise KeyError(login_id)
        snapshot = await session.poll_once()
        if snapshot.get("state") == "confirmed" and snapshot.get("bot_token"):
            account_id = uuid.uuid4().hex
            self.create_channel("wechat", f"微信 {account_id[:8]}", token=snapshot["bot_token"], base_url=snapshot.get("base_url"), account_id=account_id)
            try:
                await self.start_channel(account_id)
            except Exception as error:
                self.update_channel(account_id, {"status": "error"})
                self.db.audit("wechat_qr_channel_start_failed", target=account_id, detail={"error": redact_text(error)})
                snapshot["channel_error"] = "微信已绑定，但自动启动失败；请在渠道页重新启动。"
            snapshot["account_id"] = account_id
            # QR snapshots are rendered in the administrator browser and may
            # be retained by a proxy/browser cache. The durable channel row
            # has the token; never expose it through this management API.
            snapshot.pop("bot_token", None)
        else:
            snapshot.pop("bot_token", None)
        if snapshot.get("state") in {"confirmed", "expired", "error"}:
            await session.close()
            self._qr_sessions.pop(login_id, None)
        snapshot["login_id"] = login_id
        return snapshot

    async def submit_wechat_verify(self, login_id: str, code: str) -> dict[str, Any]:
        session = self._qr_sessions.get(login_id)
        if not session:
            raise KeyError(login_id)
        await session.submit_verify_code(code)
        return session.snapshot()

    def _record_inbox(self, account_id: str, message: Any) -> bool:
        external_id = str(message.external_id)
        existing = self.db.fetchone(
            "SELECT state FROM inbox WHERE account_id=? AND external_event_id=?",
            (account_id, external_id),
        )
        if existing:
            # A process can stop after durably recording an inbound event but
            # before its Agent turn completes. Treat that event as retryable;
            # only a confirmed processed state is a true de-duplication hit.
            if str(existing["state"] or "") == "processed":
                return False
            self.db.execute(
                "UPDATE inbox SET state='received',error=NULL WHERE account_id=? AND external_event_id=?",
                (account_id, external_id),
            )
            return True
        try:
            self.db.execute("INSERT INTO inbox(id,account_id,external_event_id,payload_json,received_at) VALUES(?,?,?,?,?)",
                            (uuid.uuid4().hex, account_id, external_id, self.db.json(message.to_dict()), utcnow()))
            return True
        except sqlite3.IntegrityError:
            # A concurrent/rare duplicate can win between the check and
            # insert. Re-read the durable state rather than silently dropping
            # an event that was recorded but never processed.
            existing = self.db.fetchone(
                "SELECT state FROM inbox WHERE account_id=? AND external_event_id=?",
                (account_id, external_id),
            )
            if not existing or str(existing["state"] or "") == "processed":
                return False
            self.db.execute(
                "UPDATE inbox SET state='received',error=NULL WHERE account_id=? AND external_event_id=?",
                (account_id, external_id),
            )
            return True

    def _mark_inbox_processed(self, account_id: str, external_id: str) -> None:
        self.db.execute("UPDATE inbox SET state='processed',processed_at=? WHERE account_id=? AND external_event_id=?", (utcnow(), account_id, str(external_id)))

    def _mark_inbox_failed(self, account_id: str, external_id: str, error: Exception) -> None:
        """Keep a retryable durable record when an Agent turn raises."""

        self.db.execute(
            "UPDATE inbox SET state='received',error=? WHERE account_id=? AND external_event_id=?",
            (redact_text(error), account_id, str(external_id)),
        )

    def _agent_settings(self, row: dict[str, Any]) -> AgentSettings:
        config = self.db.loads(row.get("config_json"), {})
        profile = None
        if row.get("api_profile_id"):
            profile = self.db.fetchone("SELECT * FROM api_profiles WHERE id=?", (row["api_profile_id"],))
        if profile:
            config = {**self.db.loads(profile["options_json"], {}), **config}
            config.setdefault("api_key", profile["secret"])
            config.setdefault("api_url", profile["base_url"])
        # MCP configuration is stored independently so administrators can
        # start/stop a server without hand-editing a large Agent JSON blob.
        # Runtime-only SDK MCP configs must never be written back to the
        # serialisable agent JSON (the local knowledge server contains Python
        # objects), so we build this mapping fresh for every runtime.
        mcp_servers: dict[str, Any] = {}
        for item in self.db.fetchall("SELECT name,config_json FROM mcp_servers WHERE agent_id=? AND enabled=1", (row["id"],)):
            value = self.db.loads(item["config_json"], {})
            if isinstance(value, dict):
                mcp_servers[str(item["name"])] = value
        saved_mcp = config.get("mcp_servers")
        if isinstance(saved_mcp, dict):
            mcp_servers = {**saved_mcp, **mcp_servers}
        knowledge_id = str(row.get("knowledge_base_id") or "")
        if knowledge_id and bool(row.get("provider", "anthropic").lower() == "anthropic"):
            try:
                mcp_servers["xmagents_knowledge"] = self._knowledge_mcp_server(str(row["id"]), knowledge_id)
            except Exception as error:
                self.db.audit("knowledge_mcp_build_failed", target=str(row["id"]), detail={"error": redact_text(error)})
        if mcp_servers:
            config["mcp_servers"] = mcp_servers
            config.setdefault("strict_mcp_config", True)
        config.update({"agent_id": row["id"], "provider": row.get("provider", "anthropic"), "model": row.get("model"),
                       "permission_mode": row.get("permission_mode", "bypassPermissions"), "effort": row.get("effort", "medium"),
                       "workspace": row.get("workspace"), "memory_enabled": bool(row.get("memory_enabled", 1)),
                       "knowledge_enabled": bool(knowledge_id),
                       # These are passed only to the Claude SDK child
                       # environment.  The master capability never leaves
                       # this service process.
                       "control_socket": str(self.control_socket_path),
                       "control_secret": self.control_secret_for_agent(str(row["id"]))})
        return AgentSettings.from_value(config)

    def _knowledge_mcp_server(self, agent_id: str, knowledge_id: str) -> Any:
        """Build read-only in-process MCP tools for one Agent knowledge base.

        It deliberately resolves the knowledge id on every call rather than
        capturing a database connection.  That keeps tools process-safe and
        makes a disabled/deleted base fail closed.
        """

        import claude_agent_sdk as sdk

        @sdk.tool(
            "knowledge_schema",
            "查看此 Agent 已启用知识库的 documents 表结构和文档数量。仅用于只读检索。",
            {},
        )
        async def knowledge_schema(_: dict[str, Any]) -> dict[str, Any]:
            from .knowledge import KnowledgeService

            try:
                result = KnowledgeService(self).schema(knowledge_id)
                return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}
            except Exception as error:
                return {"content": [{"type": "text", "text": f"知识库不可用：{redact_text(error)}"}], "isError": True}

        @sdk.tool(
            "knowledge_search",
            "在此 Agent 已启用知识库的 documents 中只读搜索。输入 query 和可选 limit。",
            {"query": str, "limit": int},
        )
        async def knowledge_search(args: dict[str, Any]) -> dict[str, Any]:
            from .knowledge import KnowledgeService

            query = str(args.get("query") or "").strip()
            if not query:
                return {"content": [{"type": "text", "text": "query 不能为空"}], "isError": True}
            try:
                limit = max(1, min(int(args.get("limit", 5) or 5), 20))
            except (TypeError, ValueError):
                limit = 5
            try:
                results = KnowledgeService(self).search(knowledge_id, query, limit)
                return {"content": [{"type": "text", "text": json.dumps(results, ensure_ascii=False)}]}
            except Exception as error:
                return {"content": [{"type": "text", "text": f"知识库检索失败：{redact_text(error)}"}], "isError": True}

        return sdk.create_sdk_mcp_server(
            name=f"xmagents-knowledge-{agent_id[:8]}",
            version="1.0.0",
            tools=[knowledge_schema, knowledge_search],
        )

    def runtime_for(self, agent_id: str) -> AgentRuntime:
        if agent_id in self._runtimes:
            return self._runtimes[agent_id]
        row = self._row("agents", agent_id)
        if not row:
            raise KeyError(agent_id)
        settings = self._agent_settings(row)
        self._restore_agent_plugins(agent_id)
        runtime = AgentRuntime(
            settings,
            plugin_loader=self.plugin_loader,
            memory=MemoryStore(self),
            knowledge=KnowledgeContext(self),
            save_message=lambda direction, content, context=None: self._save_runtime_message(agent_id, direction, content, context),
            save_settings=lambda value: self._save_runtime_settings(agent_id, value),
            reset_conversation=lambda context: self._reset_runtime_conversation(agent_id, context),
            history_loader=lambda context: self._runtime_history(agent_id, context),
        )
        if runtime.settings.extra.get("advanced_python"):
            try:
                compile(str(runtime.settings.extra["advanced_python"]), f"<agent:{agent_id}:options>", "exec")
            except SyntaxError as error:
                self.db.audit(
                    "advanced_config_invalid",
                    target=agent_id,
                    detail={"line": error.lineno, "error": redact_text(error)},
                )
        self._runtimes[agent_id] = runtime
        return runtime

    def _invalidate_runtime(self, agent_id: str) -> None:
        runtime = self._runtimes.pop(agent_id, None)
        if runtime is None:
            return
        try:
            asyncio.get_running_loop().create_task(runtime.close())
        except RuntimeError:
            # A management mutation can occur in a synchronous CLI command;
            # there is no active SDK process in that path to await here.
            pass

    def _restore_agent_plugins(self, agent_id: str) -> None:
        """Load global and per-agent plugin switches before a runtime starts."""

        global_rows = self.db.fetchall("SELECT name,enabled FROM plugins")
        for item in global_rows:
            self.plugin_loader.enabled[str(item["name"]).lower()] = bool(item["enabled"])
        rows = self.db.fetchall("SELECT plugin_name,enabled FROM agent_plugins WHERE agent_id=?", (agent_id,))
        if rows:
            effective = {plugin.command for plugin in self.plugin_loader._plugins.values()}
            for item in rows:
                name = str(item["plugin_name"]).lstrip("/").lower()
                if bool(item["enabled"]):
                    effective.add(name)
                else:
                    effective.discard(name)
            self.plugin_loader.set_agent_commands(agent_id, effective)

    def restore_plugins(self) -> None:
        """Restore persisted plugin state without constructing all runtimes."""

        global_rows = self.db.fetchall("SELECT name,enabled FROM plugins")
        for item in global_rows:
            self.plugin_loader.enabled[str(item["name"]).lower()] = bool(item["enabled"])
        for agent in self.db.fetchall("SELECT DISTINCT agent_id FROM agent_plugins"):
            self._restore_agent_plugins(str(agent["agent_id"]))

    def list_plugins(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        """Return discovered plugins plus durable global/per-agent switches."""

        rows = {str(row["name"]).lower(): dict(row) for row in self.db.fetchall("SELECT * FROM plugins")}
        agent_rows: dict[str, bool] = {}
        if agent_id:
            agent_rows = {str(row["plugin_name"]).lower(): bool(row["enabled"])
                          for row in self.db.fetchall("SELECT plugin_name,enabled FROM agent_plugins WHERE agent_id=?", (agent_id,))}
        result = []
        for plugin in self.plugin_loader._plugins.values():
            persisted = rows.get(plugin.command, {})
            global_enabled = bool(persisted.get("enabled", self.plugin_loader.enabled.get(plugin.command, True)))
            item = {
                "name": plugin.name,
                "command": plugin.command,
                "description": plugin.description,
                "usage": plugin.usage,
                "error": plugin.error,
                "enabled": global_enabled,
                "config": redact_mapping(self.db.loads(persisted.get("config_json"), {})),
            }
            if agent_id:
                item["agent_enabled"] = agent_rows.get(plugin.command, global_enabled)
            result.append(item)
        return result

    def set_plugin_enabled(self, command: str, enabled: bool, config: dict[str, Any] | None = None) -> dict[str, Any]:
        name = command.lstrip("/").lower()
        if name not in self.plugin_loader._plugins:
            raise KeyError(name)
        self.plugin_loader.enabled[name] = bool(enabled)
        now = utcnow()
        self.db.execute(
            "INSERT INTO plugins(name,enabled,config_json,updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET enabled=excluded.enabled,config_json=excluded.config_json,updated_at=excluded.updated_at",
            (name, int(enabled), self.db.json(config or {}), now),
        )
        return next(item for item in self.list_plugins() if item["command"] == name)

    def set_agent_plugin_enabled(self, agent_id: str, command: str, enabled: bool) -> dict[str, Any]:
        if not self._row("agents", agent_id):
            raise KeyError(agent_id)
        name = command.lstrip("/").lower()
        if name not in self.plugin_loader._plugins:
            raise KeyError(name)
        self.db.execute(
            "INSERT INTO plugins(name,enabled,config_json,updated_at) VALUES(?,?,?,?) ON CONFLICT(name) DO NOTHING",
            (name, 1, self.db.json({}), utcnow()),
        )
        self.db.execute(
            "INSERT INTO agent_plugins(agent_id,plugin_name,enabled) VALUES(?,?,?) "
            "ON CONFLICT(agent_id,plugin_name) DO UPDATE SET enabled=excluded.enabled",
            (agent_id, name, int(enabled)),
        )
        self.plugin_loader.set_agent_enabled(agent_id, name, bool(enabled))
        return {"ok": True, "agent_id": agent_id, "command": name, "enabled": bool(enabled)}

    async def _save_runtime_message(self, agent_id: str, direction: str, content: str, context: Any = None) -> str:
        route_key = getattr(context, "peer_id", None) or "default"
        conversation = self.conversation(agent_id, route_key)
        return self.add_message(conversation["id"], direction, content, sender=getattr(context, "user_id", "agent"))

    def _runtime_history(self, agent_id: str, context: TurnContext) -> list[dict[str, str]]:
        """Load one route's durable transcript for a newly created runtime.

        A single Agent can intentionally be bound to several peers.  Provider
        sessions and application-level history must therefore both be keyed by
        the same route, especially for OpenAI-compatible providers where the
        full transcript is sent on every request.
        """

        route_key = context.peer_id or "default"
        rows = self.db.fetchall(
            "SELECT m.direction,m.content FROM messages m JOIN conversations c ON c.id=m.conversation_id "
            "WHERE c.agent_id=? AND c.route_key=? AND m.direction IN ('user','assistant') "
            "ORDER BY m.created_at DESC LIMIT 200",
            (agent_id, route_key),
        )
        return [
            {"role": "assistant" if row["direction"] == "assistant" else "user", "content": str(row["content"])}
            for row in reversed(rows)
        ]

    async def _reset_runtime_conversation(self, agent_id: str, context: TurnContext) -> None:
        """Clear one route's provider state without deleting its audit history."""

        route_key = context.peer_id or "default"
        conversation = self.db.fetchone(
            "SELECT id FROM conversations WHERE agent_id=? AND route_key=?",
            (agent_id, route_key),
        )
        self.db.execute(
            "UPDATE conversations SET session_id=NULL,context_token=NULL,status='ready',updated_at=? WHERE agent_id=? AND route_key=?",
            (utcnow(), agent_id, route_key),
        )
        # A response may already have been queued by a transport callback
        # while /new was waiting for the mailbox. It belongs to the discarded
        # context, so prevent a later outbox worker from delivering it. Rows
        # already leased/sent remain visible for operator inspection because a
        # remote provider may have accepted them.
        if conversation:
            changed = self.db.execute(
                "UPDATE outbox SET state='cancelled',last_error=?,lease_until=NULL "
                "WHERE conversation_id=? AND state IN ('pending','deferred')",
                ("cleared by /new", conversation["id"]),
            )
            if changed:
                self.db.audit("conversation_outbox_cancelled", target=agent_id, detail={"route_key": route_key, "count": changed})

    def _save_conversation_session(self, conversation_id: str, events: list[RuntimeEvent]) -> None:
        """Persist the SDK-issued session id without treating it as a secret.

        The SDK sends the id in a result event.  Keeping it in the already
        scoped conversation row makes the relationship inspectable and lets a
        future session-store integration resume it, while normal in-process
        ClaudeSDKClient turns continue through the same transport directly.
        """

        for event in reversed(events):
            session_id = event.metadata.get("session_id") if event.metadata else None
            if session_id:
                self.db.execute(
                    "UPDATE conversations SET session_id=?,updated_at=? WHERE id=?",
                    (str(session_id), utcnow(), conversation_id),
                )
                return

    async def _save_runtime_settings(self, agent_id: str, settings: AgentSettings) -> None:
        """Persist an in-turn setting without closing its own mailbox.

        ``/effort`` invokes this callback from the active runtime.  Calling
        ``update_agent`` here would invalidate and close that same runtime,
        making every later message fail with ``agent mailbox is closed``.
        Management-page edits still go through ``update_agent`` and rebuild a
        runtime before its next turn.
        """
        row = self._row("agents", agent_id)
        if row:
            # ``settings.extra`` may contain ephemeral objects injected at
            # runtime (notably the in-process knowledge MCP server). Persist
            # only administrator-supplied JSON configuration.
            saved_config = self.db.loads(row.get("config_json"), {})
            saved_extra = dict(saved_config.get("extra") or {})
            for key, value in settings.extra.items():
                if key in {"mcp_servers", "strict_mcp_config", "control_socket", "control_secret"}:
                    continue
                try:
                    json.dumps(value)
                except (TypeError, ValueError):
                    continue
                saved_extra[key] = value
            config = {**saved_config, "extra": saved_extra}
            self.db.execute(
                "UPDATE agents SET effort=?,config_json=?,updated_at=? WHERE id=?",
                (settings.effort, self.db.json(config), utcnow(), agent_id),
            )
            self.db.audit("agent_effort_updated", target=agent_id, detail={"effort": settings.effort})

    def list_mcp_servers(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM mcp_servers"
        params: tuple[Any, ...] = ()
        if agent_id:
            query += " WHERE agent_id=?"
            params = (agent_id,)
        query += " ORDER BY created_at DESC"
        rows: list[dict[str, Any]] = []
        for row in self.db.fetchall(query, params):
            item = self._decode_row(dict(row))
            item["config"] = redact_mapping(item.get("config") or {})
            rows.append(item)
        return rows

    def create_mcp_server(self, agent_id: str, name: str, config: dict[str, Any], *, enabled: bool = True) -> dict[str, Any]:
        if not self._row("agents", agent_id):
            raise KeyError(agent_id)
        server_id = uuid.uuid4().hex
        now = utcnow()
        self.db.execute(
            "INSERT INTO mcp_servers(id,agent_id,name,config_json,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
            (server_id, agent_id, name.strip() or server_id[:8], self.db.json(config), int(enabled), now, now),
        )
        self._invalidate_runtime(agent_id)
        return next(item for item in self.list_mcp_servers(agent_id) if item["id"] == server_id)

    def update_mcp_server(self, server_id: str, values: dict[str, Any]) -> dict[str, Any]:
        row = self._row("mcp_servers", server_id)
        if not row:
            raise KeyError(server_id)
        updates: dict[str, Any] = {key: values[key] for key in ("name", "enabled") if key in values}
        if "config" in values:
            updates["config_json"] = self.db.json(values["config"] or {})
        if updates:
            updates["updated_at"] = utcnow()
            assignments = ",".join(f"{key}=?" for key in updates)
            self.db.execute(f"UPDATE mcp_servers SET {assignments} WHERE id=?", (*updates.values(), server_id))
        self._invalidate_runtime(str(row["agent_id"]))
        return next(item for item in self.list_mcp_servers(str(row["agent_id"])) if item["id"] == server_id)

    def delete_mcp_server(self, server_id: str) -> None:
        row = self._row("mcp_servers", server_id)
        if not row:
            return
        self.db.execute("DELETE FROM mcp_servers WHERE id=?", (server_id,))
        self._invalidate_runtime(str(row["agent_id"]))

    async def mcp_status(self, agent_id: str) -> dict[str, Any]:
        runtime = self._runtimes.get(agent_id)
        if runtime is None or not hasattr(runtime.provider, "client"):
            return {"connected": False, "mcpServers": []}
        client = getattr(runtime.provider, "client", None)
        if client is None or not getattr(runtime.provider, "_connected", False):
            return {"connected": False, "mcpServers": []}
        getter = getattr(client, "get_mcp_status", None)
        if getter is None:
            return {"connected": False, "mcpServers": []}
        try:
            result = getter()
            if inspect.isawaitable(result):
                result = await result
            return dict(result or {"mcpServers": []})
        except Exception as error:
            return {"connected": True, "mcpServers": [], "error": redact_text(error)}

    async def control_mcp_server(self, agent_id: str, name: str, *, enabled: bool | None = None, reconnect: bool = False) -> dict[str, Any]:
        runtime = self.runtime_for(agent_id)
        client = getattr(runtime.provider, "client", None)
        if client is None or not getattr(runtime.provider, "_connected", False):
            raise RuntimeError("Agent 尚未建立 Claude 会话")
        if reconnect:
            action = getattr(client, "reconnect_mcp_server", None)
            if action is None:
                raise RuntimeError("当前 SDK 不支持 MCP 重连")
            result = action(name)
        else:
            action = getattr(client, "toggle_mcp_server", None)
            if action is None or enabled is None:
                raise RuntimeError("当前 SDK 不支持 MCP 启停")
            result = action(name, bool(enabled))
        if inspect.isawaitable(result):
            await result
        return await self.mcp_status(agent_id)

    async def handle_incoming(self, message: Any, *, on_event: Any = None) -> list[RuntimeEvent]:
        account_id = str(message.account_id)
        channel = str(message.channel)
        is_allowed_group = bool(
            channel == "telegram"
            and getattr(message, "kind", "private") == "group"
            and self._telegram_group_allowed(account_id, str(message.peer_id))
        )
        peer = self.upsert_peer(account_id, str(message.peer_id), chat_id=str(message.peer_id) if message.kind != "private" else None,
                                display_name=getattr(message, "sender_name", ""), kind=getattr(message, "kind", "private"),
                                approved=channel == "wechat" or is_allowed_group)
        agent_binding = self.binding_for_peer(peer["id"])
        if not agent_binding and (channel == "wechat" or is_allowed_group):
            label = getattr(message, "sender_name", "") or (f"Telegram 群组-{message.peer_id}" if is_allowed_group else f"微信-{message.peer_id}")
            agent = self.create_agent(label)
            self.db.execute("UPDATE remote_peers SET approved=1 WHERE id=?", (peer["id"],))
            binding_id = uuid.uuid4().hex
            self.db.execute("INSERT INTO agent_bindings(id,agent_id,peer_id,active,created_at) VALUES(?,?,?,?,?)", (binding_id, agent["id"], peer["id"], 1, utcnow()))
            agent_binding = self.binding_for_peer(peer["id"])
        # All attachment paths injected into prompts must be files verified in
        # the bound workspace.  This matters for a first WeChat message: its
        # Agent is created just above, so downloading earlier would stage
        # bytes under an unassigned shared directory.
        needs_download = any(not str(getattr(item, "path", "") or "") for item in getattr(message, "attachments", []) or [])
        if needs_download and agent_binding and int(agent_binding.get("approved", 0)):
            adapter = self.channels.get(account_id)
            if adapter is not None:
                try:
                    message.attachments = await adapter.download_attachments(
                        message, str(Path(agent_binding["workspace"]) / "uploads"),
                    )
                except Exception as error:
                    self.db.audit("attachment_download_failed", target=account_id, detail={"error": redact_text(error)})
                    message.attachments = []
        # iLink context tokens are tied to an inbound turn.  Scheduled WeChat
        # results stay deferred until a later message yields a fresh token.
        if channel == "wechat":
            await self._flush_wechat_deferred(account_id, str(message.peer_id), getattr(message, "context_token", None))
        if not agent_binding or not int(agent_binding.get("approved", 0)):
            if channel == "telegram":
                await self.send_message(account_id, str(message.peer_id), "此 Telegram 用户尚未获管理员批准。", context_token=getattr(message, "context_token", None))
            return []
        runtime = self.runtime_for(agent_binding["id"])
        conversation = self.conversation(agent_binding["id"], str(message.peer_id))
        context = TurnContext(channel=channel, user_id=getattr(message, "sender_id", None) or str(message.peer_id), peer_id=str(message.peer_id),
                              metadata={"account_id": account_id, "context_token": getattr(message, "context_token", None)},
                              attachments=list(getattr(message, "attachments", []) or []),
                              session_id=str(conversation.get("session_id") or "default"))
        context.send_message = lambda text, **kwargs: self.send_message(account_id, str(message.peer_id), text, context_token=getattr(message, "context_token", None))
        context.send_file = lambda relative, **kwargs: self.send_file(account_id, str(message.peer_id), agent_binding["workspace"], relative, context_token=getattr(message, "context_token", None), **kwargs)
        if getattr(message, "context_token", None):
            self.db.execute("UPDATE conversations SET context_token=?,updated_at=? WHERE id=?", (message.context_token, utcnow(), conversation["id"]))
        adapter = self.channels.get(account_id)
        telegram_stream = None
        stream_failed = False

        async def channel_event(event: RuntimeEvent) -> None:
            nonlocal telegram_stream, stream_failed
            if (
                channel == "telegram"
                and event.kind == "text"
                and event.metadata.get("stream")
                and not stream_failed
                and adapter is not None
                and hasattr(adapter, "open_text_stream")
            ):
                if telegram_stream is None:
                    telegram_stream = adapter.open_text_stream(str(message.peer_id), update_interval=0.7)
                pushed = await telegram_stream.push(event.content)
                if not pushed.ok:
                    stream_failed = True
            await self._dispatch_runtime_event(
                account_id,
                str(message.peer_id),
                event,
                context_token=getattr(message, "context_token", None),
            )
            if on_event is not None:
                result = on_event(event)
                if inspect.isawaitable(result):
                    await result

        events = await runtime.submit(getattr(message, "text", ""), context, on_event=channel_event)
        self._save_conversation_session(conversation["id"], events)
        # A final non-stream message supersedes partial SDK deltas.  This
        # keeps persisted history and outbound replies free of duplicates.
        final_parts = [event.content for event in events if event.kind == "text" and not event.metadata.get("stream")]
        stream_parts = [event.content for event in events if event.kind == "text" and event.metadata.get("stream")]
        response = "".join(final_parts or stream_parts)
        if response:
            if telegram_stream is not None and not stream_failed:
                rendered = await telegram_stream.finish(response)
                if not rendered.ok:
                    # The stream may fail after Telegram accepted its first
                    # message.  Keep the durable fallback visible as unknown
                    # rather than blindly duplicating a potentially delivered
                    # answer.
                    self.db.audit("telegram_stream_finish_failed", target=account_id, detail={"error": redact_text(rendered.error)})
            elif telegram_stream is not None:
                self.db.audit("telegram_stream_delivery_unknown", target=account_id, detail={"peer_id": str(message.peer_id)})
            else:
                await self.send_message(account_id, str(message.peer_id), response, context_token=getattr(message, "context_token", None))
        return events

    def _telegram_group_allowed(self, account_id: str, peer_id: str) -> bool:
        """Return whether a Telegram group was explicitly allowed by admin."""

        account = self._row("channel_accounts", account_id)
        if not account or account.get("channel") != "telegram":
            return False
        config = self.db.loads(account.get("config_json"), {})
        values = config.get("group_allowlist", [])
        return str(peer_id) in {str(value) for value in values if value is not None}

    async def chat_agent(self, agent_id: str, text: str, *, peer_id: str = "webui") -> list[RuntimeEvent]:
        """Run a local WebUI turn through the same mailbox and history path."""

        runtime = self.runtime_for(agent_id)
        conversation = self.conversation(agent_id, peer_id)
        events = await runtime.submit(
            text,
            TurnContext(channel="webui", user_id="admin", peer_id=peer_id, session_id=str(conversation.get("session_id") or "default")),
        )
        self._save_conversation_session(conversation["id"], events)
        return events

    async def stream_agent_chat(self, agent_id: str, text: str, *, peer_id: str = "webui"):
        """Yield real-time tool/text events while preserving the final turn."""

        queue: asyncio.Queue[RuntimeEvent | None] = asyncio.Queue()

        async def receive(event: RuntimeEvent) -> None:
            await queue.put(event)

        async def run() -> None:
            try:
                await self.chat_agent_with_callback(agent_id, text, peer_id=peer_id, on_event=receive)
            except Exception as error:
                await queue.put(RuntimeEvent("error", redact_text(error)))
            finally:
                await queue.put(None)

        task = asyncio.create_task(run(), name=f"xmagents-webui-stream-{agent_id}")
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            if not task.done():
                task.cancel()

    async def chat_agent_with_callback(self, agent_id: str, text: str, *, peer_id: str, on_event: Any) -> list[RuntimeEvent]:
        runtime = self.runtime_for(agent_id)
        conversation = self.conversation(agent_id, peer_id)
        events = await runtime.submit(
            text,
            TurnContext(channel="webui", user_id="admin", peer_id=peer_id, session_id=str(conversation.get("session_id") or "default")),
            on_event=on_event,
        )
        self._save_conversation_session(conversation["id"], events)
        return events

    def agent_status(self, agent_id: str) -> dict[str, Any]:
        runtime = self.runtime_for(agent_id)
        value = runtime.settings.status()
        value.update({
            "mcp_configured": bool(runtime.settings.extra.get("mcp_servers")),
            "plugins": self.list_plugins(agent_id),
            "outbox": self.db.fetchone("SELECT COUNT(*) AS count FROM outbox WHERE state IN ('pending','leased')") ["count"],
        })
        return value

    async def send_message(self, account_id: str, peer_id: str, text: str, *, context_token: str | None = None) -> DeliveryResult:
        delivery_id = self.enqueue_outbox(account_id, peer_id, {"text": text, "context_token": context_token}, kind="text")
        return await self._deliver_outbox(delivery_id)

    async def send_file(self, account_id: str, peer_id: str, workspace: str, relative_path: str, *, context_token: str | None = None, caption: str | None = None) -> DeliveryResult:
        from .files import validate_send_file

        path = validate_send_file(workspace, relative_path)
        delivery_id = self.enqueue_outbox(account_id, peer_id, {"path": str(path), "caption": caption, "context_token": context_token}, kind="file")
        return await self._deliver_outbox(delivery_id)

    def _outbox_workspace(self, row: dict[str, Any]) -> Path | None:
        """Resolve the Agent workspace owning a persisted file delivery."""

        conversation_id = str(row.get("conversation_id") or "")
        if conversation_id:
            owner = self.db.fetchone(
                "SELECT a.workspace FROM conversations c JOIN agents a ON a.id=c.agent_id WHERE c.id=?",
                (conversation_id,),
            )
            if owner and owner["workspace"]:
                return Path(str(owner["workspace"])).resolve()
        account_id = str(row.get("account_id") or "")
        peer_id = str(row.get("peer_id") or "")
        if account_id and peer_id:
            owner = self.db.fetchone(
                "SELECT a.workspace FROM remote_peers p "
                "JOIN agent_bindings b ON b.peer_id=p.id AND b.active=1 "
                "JOIN agents a ON a.id=b.agent_id "
                "WHERE p.account_id=? AND (p.id=? OR p.external_id=? OR p.chat_id=?) LIMIT 1",
                (account_id, peer_id, peer_id, peer_id),
            )
            if owner and owner["workspace"]:
                return Path(str(owner["workspace"])).resolve()
        return None

    async def _run_schedule(self, row: dict[str, Any]) -> str:
        runtime = self.runtime_for(row["agent_id"])
        events = await runtime.submit(row["prompt"], TurnContext(channel="schedule", peer_id=row.get("peer_id") or ""))
        response = "".join(event.content for event in events if event.kind == "text")
        if row.get("peer_id"):
            peer = self._row("remote_peers", row["peer_id"])
            if peer:
                account = self._row("channel_accounts", peer["account_id"])
                if account and account.get("channel") == "wechat":
                    # iLink replies are context-bound; calling it without a
                    # fresh token is unreliable and can be rate-limited.  The
                    # next inbound WeChat message atomically releases this.
                    self.enqueue_outbox(
                        peer["account_id"],
                        peer["chat_id"] or peer["external_id"],
                        {"text": response, "context_token": None, "scheduled": True},
                        kind="wechat_deferred",
                        state="deferred",
                    )
                    self.db.audit("wechat_schedule_deferred", target=row["id"], detail={"peer_id": peer["id"]})
                else:
                    await self.send_message(peer["account_id"], peer["chat_id"] or peer["external_id"], response, context_token=None)
        return response

    # ---- setup/auth -----------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self.db.setting("admin_password_hash"))

    def initialize_admin(self, password: str) -> None:
        self.db.set_setting("admin_password_hash", hash_password(password))
        self.db.set_setting("app_version", "0.1.0")
        self.db.audit("admin_initialized")

    def change_admin_password(self, current_password: str, new_password: str) -> None:
        """Replace the administrator credential and revoke existing sessions.

        Password changes are deliberately checked in the service layer so the
        WebUI and any future management client share the same validation.  A
        new session can be issued by the caller after this method succeeds.
        """

        if not self.authenticate(current_password):
            raise ValueError("当前管理员密码错误")
        password_hash = hash_password(new_password)
        self.db.set_setting("admin_password_hash", password_hash)
        self.db.execute("DELETE FROM sessions")
        self.db.audit("admin_password_changed")

    def authenticate(self, password: str) -> bool:
        stored = self.db.setting("admin_password_hash")
        return bool(stored and verify_password(password, stored))

    def create_session(self) -> str:
        token = new_session_token()
        now = utcnow()
        self.db.execute(
            "INSERT INTO sessions(token,expires_at,created_at,last_seen_at) VALUES(?,?,?,?)",
            (token, session_expiry(), now, now),
        )
        return token

    def valid_session(self, token: str | None) -> bool:
        if not token:
            return False
        row = self.db.fetchone("SELECT token,expires_at FROM sessions WHERE token=?", (token,))
        if not row:
            return False
        from datetime import UTC, datetime

        try:
            expires = datetime.fromisoformat(row["expires_at"])
            valid = expires > datetime.now(UTC)
        except ValueError:
            valid = False
        if valid:
            self.db.execute("UPDATE sessions SET last_seen_at=? WHERE token=?", (utcnow(), token))
        else:
            self.db.execute("DELETE FROM sessions WHERE token=?", (token,))
        return valid

    # ---- generic row helpers ------------------------------------------
    def _row(self, table: str, identifier: str) -> dict[str, Any] | None:
        row = self.db.fetchone(f"SELECT * FROM {table} WHERE id=?", (identifier,))
        return dict(row) if row else None

    def _decode_row(self, row: dict[str, Any]) -> dict[str, Any]:
        for key in tuple(row):
            if key.endswith("_json"):
                row[key[:-5]] = self.db.loads(row.pop(key), {})
        return row

    def list_agents(self) -> list[dict[str, Any]]:
        return [self._decode_row(dict(row)) for row in self.db.fetchall("SELECT * FROM agents ORDER BY created_at DESC")]

    def list_channels(self) -> list[dict[str, Any]]:
        # Management responses must never put channel credentials into a page,
        # browser devtools, or a JSON export.  Writes still accept a token.
        result = []
        for row in self.db.fetchall("SELECT * FROM channel_accounts ORDER BY created_at DESC"):
            item = self._decode_row(dict(row))
            token = item.pop("token", None)
            item["token_configured"] = bool(token)
            item["token_masked"] = redact_secret(token)
            if item.get("proxy"):
                item["proxy_configured"] = True
                item["proxy_masked"] = redact_secret(item.pop("proxy"))
            else:
                item["proxy_configured"] = False
                item.pop("proxy", None)
            item["config"] = redact_mapping(item.get("config") or {})
            result.append(item)
        return result

    def channel_for_admin(self, account_id: str) -> dict[str, Any] | None:
        """Return one channel with only masked secrets for edit forms."""

        for item in self.list_channels():
            if item["id"] == account_id:
                return item
        return None

    def create_channel(self, channel: str, name: str, *, token: str | None = None, base_url: str | None = None,
                       proxy: str | None = None, config: dict[str, Any] | None = None,
                       account_id: str | None = None) -> dict[str, Any]:
        account_id = account_id or uuid.uuid4().hex
        now = utcnow()
        self.db.execute(
            "INSERT INTO channel_accounts(id,channel,name,status,token,base_url,proxy,config_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (account_id, channel, name, "stopped", token, base_url, proxy, self.db.json(config or {}), now, now),
        )
        return self.channel_for_admin(account_id) or {}

    def update_channel(self, account_id: str, values: dict[str, Any]) -> dict[str, Any]:
        existing = self._row("channel_accounts", account_id)
        if not existing:
            return {}
        was_running = str(existing.get("status") or "") == "running"
        allowed = {"name", "status", "token", "base_url", "proxy", "cursor"}
        updates = {
            key: value for key, value in values.items()
            if key in allowed and not (key in {"token", "proxy"} and value in (None, "", "***"))
        }
        if "config" in values:
            updates["config_json"] = self.db.json(values["config"])
        if updates:
            updates["updated_at"] = utcnow()
            assignments = ",".join(f"{key}=?" for key in updates)
            self.db.execute(f"UPDATE channel_accounts SET {assignments} WHERE id=?", (*updates.values(), account_id))
            connection_changed = bool(set(updates) & {"token", "base_url", "proxy", "config_json"})
            if connection_changed and account_id in self.channels:
                try:
                    asyncio.get_running_loop().create_task(self._reload_channel(account_id, was_running))
                except RuntimeError:
                    # Synchronous maintenance commands persist the new
                    # settings; the next explicit start will build them.
                    pass
        return self.channel_for_admin(account_id) or {}

    async def delete_channel(self, account_id: str) -> None:
        """Stop and remove an account plus its channel-scoped routing data."""

        if not self._row("channel_accounts", account_id):
            raise KeyError(account_id)
        await self.stop_channel(account_id)
        self.db.execute("DELETE FROM channel_accounts WHERE id=?", (account_id,))
        self.db.audit("channel_deleted", target=account_id)

    def list_api_profiles(self) -> list[dict[str, Any]]:
        rows = []
        for row in self.db.fetchall("SELECT id,name,provider,base_url,models_json,options_json,enabled,created_at,updated_at FROM api_profiles ORDER BY name"):
            item = dict(row)
            item["models"] = self.db.loads(item.pop("models_json"), [])
            item["options"] = redact_mapping(self.db.loads(item.pop("options_json"), {}))
            item["secret_configured"] = bool(self.db.fetchone("SELECT secret FROM api_profiles WHERE id=? AND secret IS NOT NULL AND secret != ''", (item["id"],)))
            rows.append(item)
        return rows

    def create_api_profile(self, *, name: str, provider: str, base_url: str | None = None,
                           models: list[str] | None = None, secret: str | None = None,
                           options: dict[str, Any] | None = None, enabled: bool = True) -> dict[str, Any]:
        profile_id = uuid.uuid4().hex
        now = utcnow()
        self.db.execute(
            "INSERT INTO api_profiles(id,name,provider,base_url,models_json,secret,options_json,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (profile_id, name.strip() or profile_id[:8], provider, base_url or None, self.db.json(models or []), secret or None,
             self.db.json(options or {}), int(enabled), now, now),
        )
        self.db.audit("api_profile_created", target=profile_id)
        return next(item for item in self.list_api_profiles() if item["id"] == profile_id)

    def update_api_profile(self, profile_id: str, values: dict[str, Any]) -> dict[str, Any]:
        row = self._row("api_profiles", profile_id)
        if not row:
            raise KeyError(profile_id)
        allowed = {"name", "provider", "base_url", "enabled"}
        updates = {key: values[key] for key in allowed if key in values}
        if "models" in values:
            updates["models_json"] = self.db.json(values["models"] or [])
        if "options" in values:
            updates["options_json"] = self.db.json(values["options"] or {})
        # A masked/blank secret means keep the existing value.  This avoids a
        # common settings-page mistake where showing a mask wipes credentials.
        if values.get("api_key") not in (None, "", "***"):
            updates["secret"] = str(values["api_key"])
        if updates:
            updates["updated_at"] = utcnow()
            assignments = ",".join(f"{key}=?" for key in updates)
            self.db.execute(f"UPDATE api_profiles SET {assignments} WHERE id=?", (*updates.values(), profile_id))
            for item in self.db.fetchall("SELECT id FROM agents WHERE api_profile_id=?", (profile_id,)):
                self._invalidate_runtime(str(item["id"]))
        self.db.audit("api_profile_updated", target=profile_id)
        return next(item for item in self.list_api_profiles() if item["id"] == profile_id)

    def delete_api_profile(self, profile_id: str) -> None:
        agent_ids = [str(item["id"]) for item in self.db.fetchall("SELECT id FROM agents WHERE api_profile_id=?", (profile_id,))]
        self.db.execute("DELETE FROM api_profiles WHERE id=?", (profile_id,))
        for agent_id in agent_ids:
            self._invalidate_runtime(agent_id)
        self.db.audit("api_profile_deleted", target=profile_id)

    # ---- agents and peers ---------------------------------------------
    def create_agent(self, name: str, *, provider: str = "anthropic", model: str | None = None,
                     api_profile_id: str | None = None, config: dict[str, Any] | None = None,
                     agent_id: str | None = None) -> dict[str, Any]:
        agent_id = agent_id or uuid.uuid4().hex
        workspace = self.paths.workspaces / agent_id
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "uploads").mkdir(exist_ok=True)
        (workspace / "CLAUDE.md").touch(exist_ok=True)
        try:
            workspace.chmod(0o700)
        except OSError:
            pass
        now = utcnow()
        config = dict(config or {})
        permission_mode = str(config.get("permission_mode") or "bypassPermissions")
        effort = str(config.get("effort") or "medium")
        memory_enabled = int(bool(config.get("memory_enabled", True)))
        knowledge_base_id = config.get("knowledge_base_id") or None
        self.db.execute(
            "INSERT INTO agents(id,name,provider,api_profile_id,model,permission_mode,effort,workspace,config_json,memory_enabled,knowledge_base_id,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (agent_id, name, provider, api_profile_id, model, permission_mode, effort,
             str(workspace), self.db.json(config), memory_enabled, knowledge_base_id, now, now),
        )
        self.db.audit("agent_created", target=agent_id, detail={"name": name})
        # Keep create and update responses identical for the WebUI: JSON
        # configuration is decoded under ``config`` instead of leaking an
        # implementation-only ``config_json`` field on only one code path.
        return self.public_agent(agent_id) or {}

    def public_agent(self, agent_id: str) -> dict[str, Any] | None:
        row = self._row("agents", agent_id)
        if not row:
            return None
        return self._decode_row(row)

    def update_agent(self, agent_id: str, values: dict[str, Any]) -> dict[str, Any]:
        allowed = {"name", "provider", "api_profile_id", "model", "permission_mode", "effort", "memory_enabled", "knowledge_base_id"}
        updates = {key: value for key, value in values.items() if key in allowed}
        row = self._row("agents", agent_id)
        if not row:
            raise KeyError(agent_id)
        if "config" in values:
            advanced = (values["config"] or {}).get("advanced_python")
            if advanced:
                try:
                    compile(str(advanced), f"<agent:{agent_id}:options>", "exec")
                except SyntaxError as error:
                    raise ValueError(f"高级 Python 配置语法错误，第 {error.lineno} 行：{error.msg}") from error
            updates["config_json"] = self.db.json(values["config"])
        if updates:
            updates["updated_at"] = utcnow()
            assignments = ",".join(f"{key}=?" for key in updates)
            self.db.execute(f"UPDATE agents SET {assignments} WHERE id=?", (*updates.values(), agent_id))
            self._invalidate_runtime(agent_id)
        return self.public_agent(agent_id) or {}

    def delete_agent(self, agent_id: str) -> None:
        row = self._row("agents", agent_id)
        if not row:
            return
        if self.runtime_manager:
            self.runtime_manager.close_agent(agent_id)
        self.db.execute("DELETE FROM agents WHERE id=?", (agent_id,))
        workspace = Path(row["workspace"])
        if workspace.is_relative_to(self.paths.workspaces) and workspace.exists():
            shutil.rmtree(workspace)
        self.db.audit("agent_deleted", target=agent_id)

    def upsert_peer(self, account_id: str, external_id: str, *, chat_id: str | None = None,
                    display_name: str | None = None, kind: str = "private", approved: bool = False) -> dict[str, Any]:
        account = self.db.fetchone("SELECT id FROM channel_accounts WHERE id=?", (account_id,))
        if not account:
            now = utcnow()
            self.db.execute(
                "INSERT INTO channel_accounts(id,channel,name,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (account_id, "unknown", account_id, "discovered", now, now),
            )
        row = self.db.fetchone(
            "SELECT * FROM remote_peers WHERE account_id=? AND external_id=? AND (chat_id IS ? OR chat_id=?)",
            (account_id, external_id, chat_id, chat_id),
        )
        now = utcnow()
        if row:
            self.db.execute("UPDATE remote_peers SET display_name=COALESCE(?,display_name),kind=?,approved=MAX(approved,?),updated_at=? WHERE id=?",
                            (display_name, kind, int(approved), now, row["id"]))
            return self._row("remote_peers", row["id"]) or {}
        peer_id = uuid.uuid4().hex
        self.db.execute(
            "INSERT INTO remote_peers(id,account_id,external_id,chat_id,display_name,kind,approved,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (peer_id, account_id, external_id, chat_id, display_name, kind, int(approved), now, now),
        )
        return self._row("remote_peers", peer_id) or {}

    def approve_peer(self, peer_id: str, agent_id: str | None = None, agent_name: str | None = None) -> dict[str, Any]:
        peer = self._row("remote_peers", peer_id)
        if not peer:
            raise KeyError(peer_id)
        if not agent_id:
            agent = self.create_agent(agent_name or peer.get("display_name") or f"agent-{peer_id[:8]}")
            agent_id = agent["id"]
        self.db.execute("UPDATE remote_peers SET approved=1,updated_at=? WHERE id=?", (utcnow(), peer_id))
        binding_id = uuid.uuid4().hex
        self.db.execute("INSERT OR REPLACE INTO agent_bindings(id,agent_id,peer_id,active,created_at) VALUES(?,?,?,?,?)",
                        (binding_id, agent_id, peer_id, 1, utcnow()))
        self.db.audit("peer_approved", target=peer_id, detail={"agent_id": agent_id})
        return self._row("agents", agent_id) or {}

    def binding_for_peer(self, peer_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT a.*,p.account_id,p.external_id,p.chat_id,p.kind,p.approved FROM agents a JOIN agent_bindings b ON b.agent_id=a.id JOIN remote_peers p ON p.id=b.peer_id WHERE b.peer_id=? AND b.active=1", (peer_id,))
        return self._decode_row(dict(row)) if row else None

    # ---- history / queue ----------------------------------------------
    def conversation(self, agent_id: str, route_key: str) -> dict[str, Any]:
        row = self.db.fetchone("SELECT * FROM conversations WHERE agent_id=? AND route_key=?", (agent_id, route_key))
        if row:
            return dict(row)
        conversation_id = uuid.uuid4().hex
        self.db.execute("INSERT INTO conversations(id,agent_id,route_key,created_at,updated_at) VALUES(?,?,?,?,?)",
                        (conversation_id, agent_id, route_key, utcnow(), utcnow()))
        return dict(self.db.fetchone("SELECT * FROM conversations WHERE id=?", (conversation_id,)))

    def add_message(self, conversation_id: str, direction: str, content: str, *, sender: str = "system",
                    message_type: str = "text", metadata: dict[str, Any] | None = None) -> str:
        message_id = uuid.uuid4().hex
        self.db.execute("INSERT INTO messages(id,conversation_id,direction,sender,content,message_type,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                        (message_id, conversation_id, direction, sender, content, message_type, self.db.json(metadata or {}), utcnow()))
        return message_id

    def history(self, agent_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.fetchall("SELECT m.*,c.route_key FROM messages m JOIN conversations c ON c.id=m.conversation_id WHERE c.agent_id=? ORDER BY m.created_at DESC LIMIT ?", (agent_id, limit))
        return [self._decode_row(dict(row)) for row in reversed(rows)]

    def outbox(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.fetchall("SELECT * FROM outbox ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 1000)),))
        result: list[dict[str, Any]] = []
        for row in rows:
            item = self._decode_row(dict(row))
            item["payload"] = redact_mapping(item.get("payload") or {})
            result.append(item)
        return result

    # ---- durable delivery ---------------------------------------------
    def _wake_outbox(self) -> None:
        """Wake the delivery worker without requiring it to be running."""

        self._outbox_wakeup.set()

    def _recover_outbox_leases(self) -> None:
        """Release expired leases left by a terminated process.

        Requests interrupted after they reached a provider are intentionally
        recorded as ``unknown`` by :meth:`_deliver_outbox`, rather than being
        replayed here.  Only a lease which naturally expired before a result
        was persisted can safely return to the retry queue.
        """

        now = utcnow()
        self.db.execute(
            "UPDATE outbox SET state='pending',lease_until=NULL,last_error=COALESCE(last_error,'delivery lease expired') "
            "WHERE state='leased' AND lease_until IS NOT NULL AND lease_until<=?",
            (now,),
        )

    @staticmethod
    def _retry_at(delay_seconds: float) -> str:
        return (datetime.now(UTC) + timedelta(seconds=max(0.0, delay_seconds))).isoformat()

    @staticmethod
    def _retry_delay(attempts: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(900.0, max(1.0, float(retry_after)))
        return min(300.0, max(2.0, 2.0 ** max(0, attempts - 1)))

    def _claim_outbox(self, delivery_id: str) -> dict[str, Any] | None:
        """Atomically lease a ready row and return its persisted payload."""

        now = utcnow()
        lease = self._retry_at(OUTBOX_LEASE_SECONDS)
        with self.db.transaction() as connection:
            row = connection.execute("SELECT * FROM outbox WHERE id=?", (delivery_id,)).fetchone()
            if not row or row["state"] != "pending":
                return None
            try:
                available = datetime.fromisoformat(str(row["available_at"]))
                if available.tzinfo is None:
                    available = available.replace(tzinfo=UTC)
                ready = available <= datetime.now(UTC)
            except (TypeError, ValueError):
                # A malformed legacy value must not stall delivery forever.
                ready = True
            if not ready:
                return None
            changed = connection.execute(
                "UPDATE outbox SET state='leased',attempts=attempts+1,lease_until=?,last_error=NULL "
                "WHERE id=? AND state='pending' AND available_at<=?",
                (lease, delivery_id, now),
            ).rowcount
            if not changed:
                return None
            claimed = dict(row)
            claimed["attempts"] = int(claimed.get("attempts", 0) or 0) + 1
            claimed["lease_until"] = lease
            return claimed

    def _finish_outbox(self, delivery_id: str, *, state: str, error: str | None = None,
                       available_at: str | None = None, sent: bool = False) -> None:
        """Finish a leased item, retaining a redacted diagnostic only."""

        self.db.execute(
            "UPDATE outbox SET state=?,lease_until=NULL,last_error=?,available_at=COALESCE(?,available_at),sent_at=? "
            "WHERE id=? AND state='leased'",
            (state, redact_text(error) if error else None, available_at, utcnow() if sent else None, delivery_id),
        )

    async def _deliver_outbox(self, delivery_id: str) -> DeliveryResult:
        """Attempt exactly one persisted delivery.

        The row is leased before a provider call.  A timeout/transport
        exception becomes ``unknown`` because the provider may have accepted
        the request; explicit provider responses are retried only for 429 and
        5xx errors.  This favors no duplicate reply over an automatic replay.
        """

        row = self._claim_outbox(delivery_id)
        if not row:
            current = self.db.fetchone("SELECT state,last_error FROM outbox WHERE id=?", (delivery_id,))
            if current and current["state"] == "sent":
                return DeliveryResult.success()
            return DeliveryResult.failure("投递项当前不可发送")
        account_id = str(row["account_id"])
        adapter = self.channels.get(account_id)
        if adapter is None:
            # Waiting for an operator to start a channel is not a provider
            # attempt.  Restore the counter so a long offline period cannot
            # exhaust retry budget before one network request is made.
            self.db.execute(
                "UPDATE outbox SET state='pending',attempts=MAX(0,attempts-1),lease_until=NULL,last_error=?,available_at=? "
                "WHERE id=? AND state='leased'",
                ("channel adapter is offline", self._retry_at(OUTBOX_IDLE_SECONDS), delivery_id),
            )
            return DeliveryResult.failure("渠道未启动，将在启动后重试")
        payload = self.db.loads(row["payload_json"], {})
        if row["kind"] == "wechat_deferred":
            # A deferred row is only released by _flush_wechat_deferred,
            # keeping this guard in case an operator manually edits SQLite.
            self._finish_outbox(delivery_id, state="deferred", error="waiting for a fresh WeChat context")
            return DeliveryResult.failure("微信定时结果等待下一条消息")
        try:
            if row["kind"] == "text":
                result = await adapter.send_text(str(row["peer_id"] or ""), str(payload.get("text") or ""), context_token=payload.get("context_token"))
            elif row["kind"] == "file":
                path = Path(str(payload.get("path") or ""))
                # Re-check canonical containment at delivery time.  A file
                # can be replaced by a symlink after enqueueing; existence
                # alone is not sufficient for a workspace boundary.
                workspace = self._outbox_workspace(row)
                if workspace is None:
                    result = DeliveryResult.failure("待发送文件的 Agent 工作区不可用")
                else:
                    try:
                        safe_path = validate_send_file(workspace, path.relative_to(workspace))
                    except (FileSafetyError, ValueError):
                        result = DeliveryResult.failure("待发送文件不在 Agent 工作区内或已不存在")
                    else:
                        result = await adapter.send_file(str(row["peer_id"] or ""), str(safe_path), caption=payload.get("caption"), context_token=payload.get("context_token"))
            else:
                result = DeliveryResult.failure(f"不支持的 outbox 类型: {row['kind']}")
        except asyncio.CancelledError:
            self._finish_outbox(delivery_id, state="unknown", error="delivery interrupted during provider request")
            raise
        except Exception as error:
            self._finish_outbox(delivery_id, state="unknown", error=error)
            self.db.audit("outbox_delivery_unknown", target=delivery_id, detail={"error": redact_text(error), "account_id": account_id})
            return DeliveryResult.failure("投递结果未知，请在 WebUI 检查 outbox")

        if result.ok:
            self._finish_outbox(delivery_id, state="sent", sent=True)
            return result

        attempts = int(row.get("attempts", 1) or 1)
        status = result.status
        retryable = status == 429 or bool(status and status >= 500)
        if retryable and attempts < OUTBOX_MAX_ATTEMPTS:
            delay = self._retry_delay(attempts, result.retry_after)
            self._finish_outbox(delivery_id, state="pending", error=result.error, available_at=self._retry_at(delay))
            self._wake_outbox()
            return result
        if retryable:
            self._finish_outbox(delivery_id, state="dead", error=result.error)
            self.db.audit("outbox_delivery_dead", target=delivery_id, detail={"status": status, "attempts": attempts, "error": redact_text(result.error)})
        elif status is None and row["kind"] == "text":
            # No provider acknowledgement is not distinguishable from a
            # dropped response for text, so keep it visible for an operator.
            self._finish_outbox(delivery_id, state="unknown", error=result.error)
            self.db.audit("outbox_delivery_unknown", target=delivery_id, detail={"error": redact_text(result.error), "account_id": account_id})
        else:
            self._finish_outbox(delivery_id, state="failed", error=result.error)
        return result

    async def _run_outbox(self) -> None:
        """Drain persisted ready deliveries while the service is alive."""

        while True:
            try:
                self._recover_outbox_leases()
                rows = self.db.fetchall(
                    "SELECT id FROM outbox WHERE state='pending' AND available_at<=? ORDER BY created_at LIMIT 20",
                    (utcnow(),),
                )
                if rows:
                    for row in rows:
                        await self._deliver_outbox(str(row["id"]))
                    continue
                self._outbox_wakeup.clear()
                try:
                    await asyncio.wait_for(self._outbox_wakeup.wait(), timeout=OUTBOX_IDLE_SECONDS)
                except asyncio.TimeoutError:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.db.audit("outbox_worker_failed", detail={"error": redact_text(error)})
                await asyncio.sleep(OUTBOX_IDLE_SECONDS)

    async def _flush_wechat_deferred(self, account_id: str, peer_id: str, context_token: str | None) -> int:
        """Release deferred scheduled iLink replies when a fresh context arrives."""

        if not context_token:
            return 0
        now = utcnow()
        delivery_ids: list[str] = []
        with self.db.transaction() as connection:
            rows = connection.execute(
                "SELECT id,payload_json FROM outbox WHERE account_id=? AND peer_id=? AND state='deferred' AND kind='wechat_deferred' ORDER BY created_at",
                (account_id, peer_id),
            ).fetchall()
            for row in rows:
                payload = self.db.loads(row["payload_json"], {})
                payload["context_token"] = context_token
                changed = connection.execute(
                    "UPDATE outbox SET state='pending',kind='text',payload_json=?,available_at=?,lease_until=NULL,last_error=NULL "
                    "WHERE id=? AND state='deferred'",
                    (self.db.json(payload), now, row["id"]),
                ).rowcount
                if changed:
                    delivery_ids.append(str(row["id"]))
        for delivery_id in delivery_ids:
            await self._deliver_outbox(delivery_id)
        if delivery_ids:
            self.db.audit("wechat_deferred_flushed", target=account_id, detail={"peer_id": peer_id, "count": len(delivery_ids)})
        return len(delivery_ids)

    def enqueue_outbox(self, account_id: str, peer_id: str | None, payload: dict[str, Any], *, kind: str = "text",
                       conversation_id: str | None = None, state: str = "pending", available_at: str | None = None) -> str:
        if state not in {"pending", "deferred"}:
            raise ValueError("outbox 初始状态必须为 pending 或 deferred")
        delivery_id = uuid.uuid4().hex
        self.db.execute("INSERT INTO outbox(id,account_id,peer_id,conversation_id,kind,payload_json,available_at,created_at) VALUES(?,?,?,?,?,?,?,?)",
                        (delivery_id, account_id, peer_id, conversation_id, kind, self.db.json(payload), available_at or utcnow(), utcnow()))
        if state != "pending":
            self.db.execute("UPDATE outbox SET state=? WHERE id=?", (state, delivery_id))
        if state == "pending":
            self._wake_outbox()
        return delivery_id
