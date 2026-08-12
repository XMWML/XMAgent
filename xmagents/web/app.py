"""FastAPI WebUI and JSON management API."""

from __future__ import annotations

import json
import hmac
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from ..knowledge import KnowledgeError, KnowledgeService
from ..service import AppService
from ..channels.redaction import redact_mapping, redact_text
from ..files import MAX_FILE_BYTES


UPLOAD_CHUNK_BYTES = 1024 * 1024
CSRF_COOKIE = "xmagents_csrf"
# Keep the header name aligned with the browser client and documented API.
# HTTP header matching is case-insensitive, but the singular product name is
# intentional here: ``X-XMAgent-CSRF``.
CSRF_HEADER = "x-xmagent-csrf"


def _as_dict(value: Any, *, field: str = "config") -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise HTTPException(400, f"{field} 必须是 JSON 对象")
    return value


def _event_payload(event: Any) -> dict[str, Any]:
    """SSE-safe payload for one tool/text event."""

    value = event.to_dict() if hasattr(event, "to_dict") else dict(event)
    return redact_mapping(value)


def _new_csrf_token() -> str:
    """Create a browser-only double-submit token.

    The session cookie remains HttpOnly; this separate token is deliberately
    readable by the dashboard JavaScript so unsafe JSON requests can carry it
    in a custom header.  A client that creates a raw service session (for
    example a local maintenance script) does not receive this cookie and keeps
    the existing programmatic API behaviour.
    """

    return secrets.token_urlsafe(32)


def _csrf_token(request: Request) -> str:
    return request.cookies.get(CSRF_COOKIE) or _new_csrf_token()


def _set_csrf_cookie(response: Any, request: Request, token: str | None = None) -> None:
    if request.cookies.get(CSRF_COOKIE):
        return
    response.set_cookie(CSRF_COOKIE, token or _new_csrf_token(), httponly=False, samesite="lax", max_age=14 * 86400, path="/")


def _csrf_matches(request: Request, supplied: str | None = None) -> bool:
    """Validate the double-submit token when this is a browser session.

    Browser login and setup responses always establish this cookie.  A missing
    cookie is rejected instead of silently bypassing CSRF protection, so a
    partially cleared browser cookie jar cannot turn an authenticated session
    into a cross-site request target.
    """

    expected = request.cookies.get(CSRF_COOKIE)
    if not expected:
        return False
    provided = supplied or request.headers.get(CSRF_HEADER)
    return bool(provided and hmac.compare_digest(str(provided), expected))


def _require_form_csrf(request: Request, supplied: str | None) -> None:
    """Require a token for the unauthenticated login/setup forms too."""

    expected = request.cookies.get(CSRF_COOKIE)
    if not expected or not supplied or not hmac.compare_digest(str(supplied), expected):
        raise HTTPException(status_code=403, detail="CSRF 校验失败，请返回登录页后重试")


async def _stage_knowledge_upload(file: UploadFile, destination: Path) -> int:
    """Copy an uploaded SQLite file without materialising it in application RAM."""

    total = 0
    try:
        with destination.open("wb") as output:
            while True:
                # Request one byte beyond the remaining quota so an exactly
                # 100 MB file is accepted while an oversized file is rejected.
                chunk_size = min(UPLOAD_CHUNK_BYTES, MAX_FILE_BYTES - total + 1)
                chunk = await file.read(max(1, chunk_size))
                if not chunk:
                    return total
                total += len(chunk)
                if total > MAX_FILE_BYTES:
                    raise HTTPException(413, f"文件超过 {MAX_FILE_BYTES // 1024 // 1024}MB 限制")
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def create_app(service: AppService | None = None) -> FastAPI:
    service = service or AppService()
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await service.start()
        yield
        await service.stop()

    app = FastAPI(title="XMAgent", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    def is_authenticated(request: Request) -> bool:
        return service.valid_session(request.cookies.get("xmagents_session"))

    async def require_auth(request: Request) -> AppService:
        if not is_authenticated(request):
            raise HTTPException(status_code=401, detail="需要管理员登录")
        # All state-changing management calls use the browser's double-submit
        # token.  The local raw-session API remains compatible when no CSRF
        # cookie exists (see ``_csrf_matches`` above).
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} or request.url.path.endswith("/chat/stream"):
            supplied = request.headers.get(CSRF_HEADER) or request.query_params.get("csrf")
            if not _csrf_matches(request, supplied):
                raise HTTPException(status_code=403, detail="CSRF 校验失败，请刷新页面后重试")
        return service

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        if not service.configured:
            csrf_token = _csrf_token(request)
            response = templates.TemplateResponse(request=request, name="login.html", context={"setup": True, "error": None, "csrf_token": csrf_token})
            _set_csrf_cookie(response, request, csrf_token)
            return response
        if not is_authenticated(request):
            csrf_token = _csrf_token(request)
            response = templates.TemplateResponse(request=request, name="login.html", context={"setup": False, "error": None, "csrf_token": csrf_token})
            _set_csrf_cookie(response, request, csrf_token)
            return response
        csrf_token = _csrf_token(request)
        response = templates.TemplateResponse(request=request, name="dashboard.html", context={
            "configured": service.configured,
            "agents": service.list_agents(),
            "channels": service.list_channels(),
            "pending": [dict(row) for row in service.db.fetchall("SELECT * FROM remote_peers WHERE approved=0 ORDER BY created_at DESC")],
            "schedules": service.scheduler.list(),
            "apis": service.list_api_profiles(),
            "mcp_servers": service.list_mcp_servers(),
            "plugins": [item for item in service.plugin_loader.list()],
            "knowledge": [dict(row) for row in service.db.fetchall("SELECT * FROM knowledge_bases ORDER BY created_at DESC")],
            "outbox": service.outbox(limit=12),
            "csrf_token": csrf_token,
        })
        _set_csrf_cookie(response, request, csrf_token)
        return response

    @app.post("/login")
    async def login(request: Request, password: str = Form(...), csrf_token: str = Form(...)):
        _require_form_csrf(request, csrf_token)
        if service.authenticate(password):
            token = service.create_session()
            response = RedirectResponse("/", status_code=303)
            response.set_cookie("xmagents_session", token, httponly=True, samesite="lax", max_age=14 * 86400)
            _set_csrf_cookie(response, request)
            return response
        csrf_token = _csrf_token(request)
        response = templates.TemplateResponse(request=request, name="login.html", context={"setup": False, "error": "密码错误", "csrf_token": csrf_token}, status_code=401)
        _set_csrf_cookie(response, request, csrf_token)
        return response

    @app.post("/setup")
    async def setup(request: Request, password: str = Form(...), password_confirm: str = Form(...), csrf_token: str = Form(...)):
        _require_form_csrf(request, csrf_token)
        if service.configured:
            raise HTTPException(status_code=409, detail="管理员已经初始化")
        if password != password_confirm:
            raise HTTPException(status_code=400, detail="两次密码不一致")
        service.initialize_admin(password)
        # Establish the first administrator session immediately.  This avoids
        # a confusing second login step and makes first-run setup atomic from
        # the browser's perspective.
        response = RedirectResponse("/", status_code=303)
        response.set_cookie("xmagents_session", service.create_session(), httponly=True, samesite="lax", max_age=14 * 86400)
        _set_csrf_cookie(response, request)
        return response

    @app.post("/logout")
    async def logout(request: Request, csrf_token: str = Form(...)):
        try:
            _require_form_csrf(request, csrf_token)
        except HTTPException:
            raise HTTPException(status_code=403, detail="CSRF 校验失败，请刷新页面后重试")
        token = request.cookies.get("xmagents_session")
        if token:
            service.db.execute("DELETE FROM sessions WHERE token=?", (token,))
        response = RedirectResponse("/", status_code=303)
        response.delete_cookie("xmagents_session")
        response.delete_cookie(CSRF_COOKIE)
        return response

    @app.post("/api/auth/password")
    async def change_password(request: Request, payload: dict[str, Any], _: AppService = Depends(require_auth)):
        current = str(payload.get("current_password") or "")
        new_password = str(payload.get("new_password") or "")
        confirmation = str(payload.get("new_password_confirm") or "")
        if new_password != confirmation:
            raise HTTPException(400, "两次新密码不一致")
        try:
            service.change_admin_password(current, new_password)
        except ValueError as error:
            message = str(error)
            raise HTTPException(401 if "当前" in message else 400, message)
        token = service.create_session()
        response = JSONResponse({"ok": True})
        response.set_cookie("xmagents_session", token, httponly=True, samesite="lax", max_age=14 * 86400)
        _set_csrf_cookie(response, request)
        return response

    @app.get("/api/health")
    async def health():
        return {"ok": True, "configured": service.configured, "version": "0.1.0"}

    @app.get("/api/dashboard")
    async def dashboard(_: AppService = Depends(require_auth)):
        return {"agents": service.list_agents(), "channels": service.list_channels(), "schedules": service.scheduler.list(),
                "pending": [dict(row) for row in service.db.fetchall("SELECT * FROM remote_peers WHERE approved=0 ORDER BY created_at DESC")],
                "outbox_pending": service.db.fetchone("SELECT COUNT(*) AS count FROM outbox WHERE state='pending'")["count"]}

    @app.get("/api/agents")
    async def agents(_: AppService = Depends(require_auth)):
        return service.list_agents()

    @app.post("/api/agents")
    async def create_agent(payload: dict[str, Any], _: AppService = Depends(require_auth)):
        name = str(payload.get("name") or "Agent").strip()
        if not name:
            raise HTTPException(400, "Agent 名称不能为空")
        config = dict(_as_dict(payload.get("config"), field="config"))
        # The dashboard keeps common Agent settings as top-level form fields.
        # Fold them into the persisted config as well as the normalized Agent
        # columns so a successful create request cannot silently lose values.
        for key in ("permission_mode", "effort", "memory_enabled", "knowledge_base_id"):
            if key in payload:
                config[key] = payload[key]
        return service.create_agent(name, provider=str(payload.get("provider", "anthropic")), model=payload.get("model"),
                                    api_profile_id=payload.get("api_profile_id"), config=config)

    @app.patch("/api/agents/{agent_id}")
    async def update_agent(agent_id: str, payload: dict[str, Any], _: AppService = Depends(require_auth)):
        try:
            if "config" in payload:
                payload["config"] = _as_dict(payload["config"], field="config")
            return service.update_agent(agent_id, payload)
        except KeyError:
            raise HTTPException(404, "Agent 不存在")
        except ValueError as error:
            raise HTTPException(400, str(error))

    @app.delete("/api/agents/{agent_id}")
    async def delete_agent(agent_id: str, _: AppService = Depends(require_auth)):
        service.delete_agent(agent_id)
        return {"ok": True}

    @app.get("/api/agents/{agent_id}/history")
    async def history(agent_id: str, limit: int = 100, _: AppService = Depends(require_auth)):
        return service.history(agent_id, max(1, min(limit, 1000)))

    @app.get("/api/agents/{agent_id}/status")
    async def agent_status(agent_id: str, _: AppService = Depends(require_auth)):
        try:
            return service.agent_status(agent_id)
        except KeyError:
            raise HTTPException(404, "Agent 不存在")

    @app.post("/api/agents/{agent_id}/chat")
    async def agent_chat(agent_id: str, payload: dict[str, Any], _: AppService = Depends(require_auth)):
        text = str(payload.get("text") or "").strip()
        if not text:
            raise HTTPException(400, "消息不能为空")
        try:
            events = await service.chat_agent(agent_id, text, peer_id=str(payload.get("peer_id") or "webui"))
        except KeyError:
            raise HTTPException(404, "Agent 不存在")
        return [_event_payload(event) for event in events]

    @app.get("/api/agents/{agent_id}/chat/stream")
    async def agent_chat_stream(agent_id: str, text: str, peer_id: str = "webui", _: AppService = Depends(require_auth)):
        if not text.strip():
            raise HTTPException(400, "消息不能为空")

        async def stream():
            try:
                async for event in service.stream_agent_chat(agent_id, text, peer_id=peer_id):
                    yield f"data: {json.dumps(_event_payload(event), ensure_ascii=False)}\n\n"
            except KeyError:
                yield f"data: {json.dumps({'kind': 'error', 'content': 'Agent 不存在'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/channels")
    async def channels(_: AppService = Depends(require_auth)):
        return service.list_channels()

    @app.post("/api/channels")
    async def create_channel(payload: dict[str, Any], _: AppService = Depends(require_auth)):
        channel = str(payload.get("channel") or "").lower()
        if channel not in {"wechat", "telegram"}:
            raise HTTPException(400, "channel 必须是 wechat 或 telegram")
        return service.create_channel(channel, str(payload.get("name") or channel), token=payload.get("token"),
                                      base_url=payload.get("base_url"), proxy=payload.get("proxy"), config=_as_dict(payload.get("config"), field="config"))

    @app.patch("/api/channels/{account_id}")
    async def update_channel(account_id: str, payload: dict[str, Any], _: AppService = Depends(require_auth)):
        if not service.channel_for_admin(account_id):
            raise HTTPException(404, "渠道不存在")
        if "config" in payload:
            payload["config"] = _as_dict(payload["config"], field="config")
        return service.update_channel(account_id, payload)

    @app.post("/api/channels/{account_id}/start")
    async def start_channel(account_id: str, _: AppService = Depends(require_auth)):
        try:
            return await service.start_channel(account_id)
        except (KeyError, ValueError, RuntimeError) as error:
            raise HTTPException(400, str(error))

    @app.post("/api/channels/{account_id}/stop")
    async def stop_channel(account_id: str, _: AppService = Depends(require_auth)):
        await service.stop_channel(account_id)
        return {"ok": True}

    @app.delete("/api/channels/{account_id}")
    async def delete_channel(account_id: str, _: AppService = Depends(require_auth)):
        try:
            await service.delete_channel(account_id)
        except KeyError:
            raise HTTPException(404, "渠道不存在")
        return {"ok": True}

    @app.post("/api/channels/wechat/qr/start")
    async def start_wechat_qr(payload: dict[str, Any] | None = None, _: AppService = Depends(require_auth)):
        try:
            return await service.begin_wechat_qr((payload or {}).get("base_url"))
        except Exception as error:
            raise HTTPException(502, str(error))

    @app.get("/api/channels/wechat/qr/{login_id}")
    async def poll_wechat_qr(login_id: str, _: AppService = Depends(require_auth)):
        try:
            return await service.poll_wechat_qr(login_id)
        except KeyError:
            raise HTTPException(404, "二维码会话不存在或已结束")

    @app.post("/api/channels/wechat/qr/{login_id}/verify")
    async def verify_wechat_qr(login_id: str, payload: dict[str, Any], _: AppService = Depends(require_auth)):
        try:
            return await service.submit_wechat_verify(login_id, str(payload.get("code") or ""))
        except KeyError:
            raise HTTPException(404, "二维码会话不存在或已结束")

    @app.get("/api/pending")
    async def pending(_: AppService = Depends(require_auth)):
        return [dict(row) for row in service.db.fetchall("SELECT p.*,c.channel,c.name AS account_name FROM remote_peers p JOIN channel_accounts c ON c.id=p.account_id WHERE p.approved=0 ORDER BY p.created_at DESC")]

    @app.post("/api/pending/{peer_id}/approve")
    async def approve(peer_id: str, payload: dict[str, Any] | None = None, _: AppService = Depends(require_auth)):
        try:
            return service.approve_peer(peer_id, (payload or {}).get("agent_id"), (payload or {}).get("agent_name"))
        except KeyError:
            raise HTTPException(404, "待审批用户不存在")

    @app.get("/api/apis")
    async def apis(_: AppService = Depends(require_auth)):
        return service.list_api_profiles()

    @app.post("/api/apis")
    async def create_api(payload: dict[str, Any], _: AppService = Depends(require_auth)):
        models = payload.get("models") or []
        if not isinstance(models, list):
            raise HTTPException(400, "models 必须是模型名数组")
        return service.create_api_profile(name=str(payload.get("name") or ""), provider=str(payload.get("provider") or "anthropic"),
                                          base_url=payload.get("base_url"), models=[str(item) for item in models],
                                          secret=payload.get("api_key"), options=_as_dict(payload.get("options"), field="options"),
                                          enabled=bool(payload.get("enabled", True)))

    @app.patch("/api/apis/{profile_id}")
    async def update_api(profile_id: str, payload: dict[str, Any], _: AppService = Depends(require_auth)):
        try:
            if "options" in payload:
                payload["options"] = _as_dict(payload["options"], field="options")
            return service.update_api_profile(profile_id, payload)
        except KeyError:
            raise HTTPException(404, "API 配置不存在")

    @app.delete("/api/apis/{profile_id}")
    async def delete_api(profile_id: str, _: AppService = Depends(require_auth)):
        service.delete_api_profile(profile_id)
        return {"ok": True}

    @app.get("/api/mcp")
    async def mcp(_: AppService = Depends(require_auth)):
        return service.list_mcp_servers()

    @app.post("/api/mcp")
    async def create_mcp(payload: dict[str, Any], _: AppService = Depends(require_auth)):
        try:
            return service.create_mcp_server(str(payload.get("agent_id") or ""), str(payload.get("name") or ""),
                                             _as_dict(payload.get("config"), field="config"), enabled=bool(payload.get("enabled", True)))
        except KeyError:
            raise HTTPException(404, "Agent 不存在")

    @app.patch("/api/mcp/{server_id}")
    async def update_mcp(server_id: str, payload: dict[str, Any], _: AppService = Depends(require_auth)):
        try:
            if "config" in payload:
                payload["config"] = _as_dict(payload["config"], field="config")
            return service.update_mcp_server(server_id, payload)
        except KeyError:
            raise HTTPException(404, "MCP 不存在")

    @app.delete("/api/mcp/{server_id}")
    async def delete_mcp(server_id: str, _: AppService = Depends(require_auth)):
        service.delete_mcp_server(server_id)
        return {"ok": True}

    @app.get("/api/agents/{agent_id}/mcp/status")
    async def mcp_status(agent_id: str, _: AppService = Depends(require_auth)):
        return await service.mcp_status(agent_id)

    @app.post("/api/agents/{agent_id}/mcp/{name}/reconnect")
    async def reconnect_mcp(agent_id: str, name: str, _: AppService = Depends(require_auth)):
        try:
            return await service.control_mcp_server(agent_id, name, reconnect=True)
        except (KeyError, RuntimeError) as error:
            raise HTTPException(400, str(error))

    @app.post("/api/agents/{agent_id}/mcp/{name}/toggle")
    async def toggle_mcp(agent_id: str, name: str, payload: dict[str, Any], _: AppService = Depends(require_auth)):
        try:
            return await service.control_mcp_server(agent_id, name, enabled=bool(payload.get("enabled", True)))
        except (KeyError, RuntimeError) as error:
            raise HTTPException(400, str(error))

    @app.get("/api/plugins")
    async def plugins(agent_id: str | None = None, _: AppService = Depends(require_auth)):
        if agent_id and not service.public_agent(agent_id):
            raise HTTPException(404, "Agent 不存在")
        return service.list_plugins(agent_id)

    @app.patch("/api/plugins/{command}")
    async def toggle_plugin(command: str, payload: dict[str, Any], _: AppService = Depends(require_auth)):
        try:
            return service.set_plugin_enabled(command, bool(payload.get("enabled", True)), _as_dict(payload.get("config"), field="config"))
        except KeyError:
            raise HTTPException(404, "插件不存在")

    @app.patch("/api/agents/{agent_id}/plugins/{command}")
    async def toggle_agent_plugin(agent_id: str, command: str, payload: dict[str, Any], _: AppService = Depends(require_auth)):
        try:
            return service.set_agent_plugin_enabled(agent_id, command, bool(payload.get("enabled", True)))
        except KeyError:
            raise HTTPException(404, "Agent 或插件不存在")

    @app.get("/api/memory/{agent_id}")
    async def memory(agent_id: str, _: AppService = Depends(require_auth)):
        from ..memory import MemoryStore

        if not service.public_agent(agent_id):
            raise HTTPException(404, "Agent 不存在")
        return MemoryStore(service).list(agent_id)

    @app.delete("/api/memory/{agent_id}/{memory_id}")
    async def delete_memory(agent_id: str, memory_id: str, _: AppService = Depends(require_auth)):
        from ..memory import MemoryStore

        row = service.db.fetchone("SELECT agent_id FROM memory_entries WHERE id=?", (memory_id,))
        if not row or str(row["agent_id"]) != agent_id:
            raise HTTPException(404, "记忆不存在")
        MemoryStore(service).delete(memory_id)
        return {"ok": True}

    @app.patch("/api/memory/{agent_id}/{memory_id}")
    async def update_memory(agent_id: str, memory_id: str, payload: dict[str, Any], _: AppService = Depends(require_auth)):
        from ..memory import MemoryStore

        row = service.db.fetchone("SELECT agent_id FROM memory_entries WHERE id=?", (memory_id,))
        if not row or str(row["agent_id"]) != agent_id:
            raise HTTPException(404, "记忆不存在")
        try:
            return MemoryStore(service).update(memory_id, str(payload.get("content") or ""), enabled=payload.get("enabled"), kind=payload.get("kind"))
        except ValueError as error:
            raise HTTPException(400, str(error))

    @app.post("/api/memory/{agent_id}")
    async def add_memory(agent_id: str, payload: dict[str, Any], _: AppService = Depends(require_auth)):
        from ..memory import MemoryStore

        if not service.public_agent(agent_id):
            raise HTTPException(404, "Agent 不存在")
        return {"id": MemoryStore(service).add(agent_id, str(payload.get("content") or ""), kind=str(payload.get("kind") or "fact"))}

    @app.get("/api/knowledge")
    async def knowledge(_: AppService = Depends(require_auth)):
        return [dict(row) for row in service.db.fetchall("SELECT * FROM knowledge_bases ORDER BY created_at DESC")]

    @app.patch("/api/knowledge/{knowledge_id}")
    async def update_knowledge(knowledge_id: str, payload: dict[str, Any], _: AppService = Depends(require_auth)):
        if "enabled" in payload:
            service.db.execute("UPDATE knowledge_bases SET enabled=?,updated_at=datetime('now') WHERE id=?", (int(bool(payload["enabled"])), knowledge_id))
        row = service.db.fetchone("SELECT * FROM knowledge_bases WHERE id=?", (knowledge_id,))
        if not row:
            raise HTTPException(404, "知识库不存在")
        return dict(row)

    @app.delete("/api/knowledge/{knowledge_id}")
    async def delete_knowledge(knowledge_id: str, _: AppService = Depends(require_auth)):
        row = service.db.fetchone("SELECT source_path FROM knowledge_bases WHERE id=?", (knowledge_id,))
        if not row:
            raise HTTPException(404, "知识库不存在")
        source = Path(row["source_path"])
        if source.parent.resolve() == service.paths.knowledge.resolve():
            source.unlink(missing_ok=True)
            source.with_suffix(".fts.sqlite3").unlink(missing_ok=True)
        service.db.execute("UPDATE agents SET knowledge_base_id=NULL WHERE knowledge_base_id=?", (knowledge_id,))
        service.db.execute("DELETE FROM knowledge_bases WHERE id=?", (knowledge_id,))
        return {"ok": True}

    @app.post("/api/knowledge/import")
    async def knowledge_import(file: UploadFile = File(...), name: str = Form("knowledge"), _: AppService = Depends(require_auth)):
        temporary = service.paths.runtime / f"knowledge-upload-{secrets.token_hex(8)}.sqlite3"
        try:
            await _stage_knowledge_upload(file, temporary)
            return KnowledgeService(service).import_database(temporary, name)
        except KnowledgeError as error:
            raise HTTPException(400, str(error))
        finally:
            temporary.unlink(missing_ok=True)
            await file.close()

    @app.get("/api/schedules")
    async def schedules(_: AppService = Depends(require_auth)):
        return service.scheduler.list()

    @app.post("/api/schedules")
    async def create_schedule(payload: dict[str, Any], _: AppService = Depends(require_auth)):
        try:
            return service.scheduler.create(str(payload["agent_id"]), str(payload["prompt"]), str(payload["expression"]), str(payload.get("expression_type", "at")), peer_id=payload.get("peer_id"), timezone=str(payload.get("timezone", "Asia/Shanghai")))
        except (KeyError, ValueError) as error:
            raise HTTPException(400, str(error))

    @app.delete("/api/schedules/{schedule_id}")
    async def cancel_schedule(schedule_id: str, _: AppService = Depends(require_auth)):
        service.scheduler.cancel(schedule_id)
        return {"ok": True}

    @app.get("/api/logs")
    async def logs(limit: int = 100, _: AppService = Depends(require_auth)):
        return [redact_mapping(dict(row)) for row in service.db.fetchall("SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 500)),))]

    @app.get("/api/outbox")
    async def outbox(limit: int = 100, _: AppService = Depends(require_auth)):
        return service.outbox(limit=max(1, min(limit, 500)))

    @app.get("/api/events")
    async def events(_: AppService = Depends(require_auth)):
        async def stream():
            yield f"data: {json.dumps({'agents': len(service.list_agents()), 'channels': len(service.list_channels())}, ensure_ascii=False)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


__all__ = ["create_app"]
