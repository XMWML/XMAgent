"""XMAgent command line entry point and ASGI application factory."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

from .config import AppPaths
from .control import ControlClient, ControlError, controlled_environment
from .service import AppService


def build_service(data_dir: str | None = None) -> AppService:
    return AppService(AppPaths.from_root(Path.cwd(), data_dir))


def build_app(service: AppService | None = None):
    from .web.app import create_app

    return create_app(service or build_service())


def command_init(service: AppService) -> int:
    if service.configured:
        print("管理员密码已经设置。使用 WebUI 修改配置，或删除 data/xmagents.sqlite3 后重新初始化。")
        return 0
    first = input("设置管理员密码（至少 10 个字符）: ")
    second = input("再次输入管理员密码: ")
    if first != second:
        print("两次密码不一致", file=sys.stderr)
        return 2
    try:
        service.initialize_admin(first)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"初始化完成，数据目录：{service.paths.data}")
    return 0


def command_doctor(service: AppService) -> int:
    sidecars = [service.db.path.with_name(service.db.path.name + suffix) for suffix in ("-wal", "-shm")]
    checks = {
        "database": service.db.path.exists(),
        "data_permissions": (service.paths.data.stat().st_mode & 0o077) == 0,
        "database_sidecar_permissions": all(not path.exists() or (path.stat().st_mode & 0o077) == 0 for path in sidecars),
        "admin_configured": service.configured,
    }
    for name, passed in checks.items():
        print(f"{'OK' if passed else 'WARN'}  {name}")
    return 0 if checks["database"] else 1


def command_send_file(service: AppService, args: argparse.Namespace) -> int:
    """Send a workspace file through a bound peer from a controlled CLI call."""
    peer = service.db.fetchone(
        "SELECT p.*,a.id AS agent_id,a.workspace,c.channel FROM remote_peers p "
        "JOIN agent_bindings b ON b.peer_id=p.id "
        "JOIN agents a ON a.id=b.agent_id "
        "JOIN channel_accounts c ON c.id=p.account_id "
        "WHERE b.active=1 AND a.id=? LIMIT 1",
        (args.agent_id,),
    )
    if not peer:
        print("Agent 没有绑定渠道用户", file=sys.stderr)
        return 2
    try:
        result = asyncio.run(_send_file_once(service, dict(peer), args.path, args.caption))
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2
    print("文件发送成功" if getattr(result, "ok", False) else f"文件发送失败：{getattr(result, 'error', result)}")
    return 0 if getattr(result, "ok", False) else 1


def _agent_id_from_scope(args: argparse.Namespace) -> str:
    """Resolve an Agent id for CLI calls without trusting an arbitrary cwd."""

    agent_id = str(getattr(args, "agent_id", "") or os.getenv("XMAGENTS_AGENT_ID") or "").strip()
    if not agent_id:
        raise ValueError("请提供 agent_id，或由受控 Agent 环境设置 XMAGENTS_AGENT_ID")
    return agent_id


def command_agent_send_file(service: AppService, args: argparse.Namespace) -> int:
    """Worker-facing alias with the same workspace/path validation as CLI."""

    try:
        agent_id = _agent_id_from_scope(args)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    proxy = argparse.Namespace(agent_id=agent_id, path=args.path, caption=args.caption)
    return command_send_file(service, proxy)


async def _send_file_once(service: AppService, peer: dict[str, Any], relative_path: str, caption: str | None) -> Any:
    """Open only the adapter needed for a short-lived CLI delivery.

    ``AppService.start_channel`` also starts a forever polling task, which is
    appropriate for the server but not for a one-shot subprocess.  The CLI
    therefore starts the configured adapter directly, registers it long enough
    for the common service delivery path to use it, and always closes it.
    """

    account_id = str(peer["account_id"])
    adapter = service.channels.get(account_id)
    temporary_adapter = adapter is None
    if temporary_adapter:
        account = service._row("channel_accounts", account_id)
        if not account:
            raise ValueError("渠道账号不存在")
        adapter = service._build_adapter(account)
        try:
            await adapter.start()
        except Exception:
            await adapter.stop()
            raise
        service.channels[account_id] = adapter
    try:
        route_key = str(peer.get("chat_id") or peer["external_id"])
        conversation = service.db.fetchone(
            "SELECT context_token FROM conversations WHERE agent_id=? AND route_key=?",
            (str(peer["agent_id"]), route_key),
        )
        context_token = str(conversation["context_token"]) if conversation and conversation["context_token"] else None
        return await service.send_file(
            account_id,
            route_key,
            str(peer["workspace"]),
            relative_path,
            caption=caption,
            context_token=context_token,
        )
    finally:
        if temporary_adapter and adapter is not None:
            service.channels.pop(account_id, None)
            await adapter.stop()


def _schedule_agent_id(args: argparse.Namespace) -> str:
    positional = getattr(args, "agent_id", None)
    option = getattr(args, "agent_id_option", None)
    if positional and option and positional != option:
        raise ValueError("agent_id 位置参数与 --agent-id 不一致")
    agent_id = option or positional or os.getenv("XMAGENTS_AGENT_ID")
    if not agent_id:
        raise ValueError("请提供 agent_id（位置参数、--agent-id 或 XMAGENTS_AGENT_ID）")
    return str(agent_id)


def _schedule_prompt(args: argparse.Namespace) -> str:
    option = getattr(args, "prompt", None)
    positional = " ".join(getattr(args, "prompt_parts", [])).strip()
    if option and positional:
        raise ValueError("请使用 --prompt 或位置参数之一提供任务内容")
    prompt = str(option or positional).strip()
    if not prompt:
        raise ValueError("任务内容不能为空；使用 --prompt 提供")
    return prompt


def command_schedule_create(service: AppService, args: argparse.Namespace) -> int:
    try:
        agent_id = _schedule_agent_id(args)
        if not service._row("agents", agent_id):
            raise ValueError("Agent 不存在")
        selected = [(kind, str(value)) for kind, value in (("at", args.at), ("every", args.every), ("cron", args.cron)) if value is not None]
        if len(selected) != 1:
            raise ValueError("请且只能指定 --at、--every 或 --cron 之一")
        expression_type, expression = selected[0]
        peer_id = str(args.peer_id) if args.peer_id else None
        if peer_id:
            peer = service._row("remote_peers", peer_id)
            if not peer:
                raise ValueError("渠道用户不存在")
            binding = service.binding_for_peer(peer_id)
            if not binding or str(binding["id"]) != agent_id:
                raise ValueError("渠道用户未绑定到此 Agent")
        result = service.scheduler.create(
            agent_id,
            _schedule_prompt(args),
            expression,
            expression_type,
            peer_id=peer_id,
            timezone=str(args.timezone),
        )
    except (StopIteration, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def command_schedule_list(service: AppService, args: argparse.Namespace) -> int:
    try:
        agent_id = _schedule_agent_id(args) if (args.agent_id or args.agent_id_option or os.getenv("XMAGENTS_AGENT_ID")) else None
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(service.scheduler.list(agent_id), ensure_ascii=False, indent=2, default=str))
    return 0


def command_schedule_cancel(service: AppService, args: argparse.Namespace) -> int:
    if not service._row("schedules", args.schedule_id):
        print("定时任务不存在", file=sys.stderr)
        return 2
    service.scheduler.cancel(args.schedule_id)
    print("定时任务已取消")
    return 0


def _controlled_client() -> tuple[ControlClient, str, str] | None:
    """Return the Agent-scoped socket client when invoked by Claude Code."""

    scope = controlled_environment()
    if scope is None:
        return None
    socket_path, secret, agent_id = scope
    return ControlClient(socket_path), secret, agent_id


def _controlled_request(action: str, values: dict[str, Any]) -> dict[str, Any] | None:
    """Issue one scoped request, or return ``None`` for admin CLI mode."""

    scoped = _controlled_client()
    if scoped is None:
        return None
    client, secret, agent_id = scoped
    payload = {"action": action, "agent_id": agent_id, "secret": secret, **values}
    return client.request(payload)


def command_controlled_agent_send_file(args: argparse.Namespace) -> int:
    try:
        result = _controlled_request("send_file", {"relative_path": args.path, "caption": args.caption})
        if result is None:
            raise ControlError("不在受控 Agent 环境中")
    except (ControlError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    delivery = dict(result.get("delivery") or {})
    print("文件发送请求已提交" if delivery.get("ok") else f"文件发送失败：{delivery.get('error') or '投递未成功'}")
    return 0 if delivery.get("ok") else 1


def command_controlled_schedule(args: argparse.Namespace) -> int:
    """Route all schedule actions through a per-Agent local capability."""

    try:
        scoped = _controlled_client()
        if scoped is None:
            raise ControlError("不在受控 Agent 环境中")
        _, _, scoped_agent_id = scoped
        if args.schedule_command == "create":
            requested_agent = _schedule_agent_id(args)
            if requested_agent != scoped_agent_id:
                raise ControlError("受控 Agent 不能操作其他 Agent 的任务")
            selected = [(kind, str(value)) for kind, value in (("at", args.at), ("every", args.every), ("cron", args.cron)) if value is not None]
            if len(selected) != 1:
                raise ControlError("请且只能指定 --at、--every 或 --cron 之一")
            expression_type, expression = selected[0]
            result = _controlled_request("schedule_create", {
                "prompt": _schedule_prompt(args),
                "expression": expression,
                "expression_type": expression_type,
                "peer_id": args.peer_id,
                "timezone": args.timezone,
            })
            assert result is not None
            print(json.dumps(result["schedule"], ensure_ascii=False, indent=2, default=str))
            return 0
        if args.schedule_command == "list":
            requested_agent = _schedule_agent_id(args) if (args.agent_id or args.agent_id_option) else scoped_agent_id
            if requested_agent != scoped_agent_id:
                raise ControlError("受控 Agent 只能查看自己的任务")
            result = _controlled_request("schedule_list", {})
            assert result is not None
            print(json.dumps(result["schedules"], ensure_ascii=False, indent=2, default=str))
            return 0
        result = _controlled_request("schedule_cancel", {"schedule_id": args.schedule_id})
        assert result is not None
        print("定时任务已取消")
        return 0
    except (ControlError, StopIteration, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


def agent_send_file_main(argv: list[str] | None = None) -> int:
    """Dedicated console entry point used from the Claude Agent workspace."""

    return main(["agent-send-file", *(argv if argv is not None else sys.argv[1:])])


def command_serve(service: AppService, args: argparse.Namespace) -> int:
    if not service.configured:
        print("请先执行 xmagents init", file=sys.stderr)
        return 2
    import uvicorn

    app = build_app(service)
    # Uvicorn keeps its lifespan inside one Server.  Pass both pre-bound
    # sockets to that one Server, rather than starting two app instances.
    if args.host == "both":
        async def serve_both() -> None:
            sockets: list[socket.socket] = []
            for family, address in ((socket.AF_INET, ("0.0.0.0", args.port)), (socket.AF_INET6, ("::", args.port))):
                listener = socket.socket(family, socket.SOCK_STREAM)
                try:
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    if family == socket.AF_INET6:
                        listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                    listener.bind(address)
                    listener.listen(socket.SOMAXCONN)
                    listener.setblocking(False)
                    sockets.append(listener)
                except OSError:
                    listener.close()
                    if family == socket.AF_INET:
                        raise
            server = uvicorn.Server(uvicorn.Config(app, log_level=args.log_level))
            try:
                await server.serve(sockets=sockets)
            finally:
                for listener in sockets:
                    listener.close()

        asyncio.run(serve_both())
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xmagents", description="XMAgent 多渠道 Agent 服务")
    parser.add_argument("--data-dir", default=None, help="运行数据目录，默认 data/")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="首次设置管理员密码")
    subparsers.add_parser("doctor", help="检查数据库和本地运行环境")
    serve = subparsers.add_parser("serve", help="启动局域网 WebUI/服务")
    serve.add_argument("--host", default=os.getenv("XMAGENTS_HOST", "both"),
                       help="监听地址；默认 both 同时监听 IPv4/IPv6，可设为具体地址")
    serve.add_argument("--port", type=int, default=int(os.getenv("XMAGENTS_PORT", "8765")))
    serve.add_argument("--log-level", default="info")
    send_file = subparsers.add_parser("send-file", help="从 Agent 工作区发送文件")
    send_file.add_argument("agent_id")
    send_file.add_argument("path")
    send_file.add_argument("--caption", default=None)
    agent_send_file = subparsers.add_parser("agent-send-file", help="供 Agent 工作区受控环境发送文件")
    agent_send_file.add_argument("path")
    agent_send_file.add_argument("--agent-id", default=None)
    agent_send_file.add_argument("--caption", default=None)

    schedule = subparsers.add_parser("schedule", help="管理 Agent 定时任务")
    schedule_subparsers = schedule.add_subparsers(dest="schedule_command", required=True)
    schedule_create = schedule_subparsers.add_parser("create", help="创建定时任务")
    schedule_create.add_argument("agent_id", nargs="?", help="目标 Agent ID；也可使用 --agent-id 或 XMAGENTS_AGENT_ID")
    schedule_create.add_argument("prompt_parts", nargs="*", help="任务内容；也可使用 --prompt")
    schedule_create.add_argument("--agent-id", dest="agent_id_option")
    schedule_create.add_argument("--peer-id", help="可选的渠道用户 ID，用于将结果回传给该用户")
    schedule_create.add_argument("--prompt", help="任务内容")
    schedule_create.add_argument("--timezone", default="Asia/Shanghai")
    timing = schedule_create.add_mutually_exclusive_group(required=True)
    timing.add_argument("--at", help="ISO 8601 执行时间，例如 2026-08-13T09:00:00+08:00")
    timing.add_argument("--every", help="执行间隔秒数")
    timing.add_argument("--cron", help="五字段 Cron 表达式")

    schedule_list = schedule_subparsers.add_parser("list", help="列出定时任务")
    schedule_list.add_argument("agent_id", nargs="?", help="可选 Agent ID")
    schedule_list.add_argument("--agent-id", dest="agent_id_option")

    schedule_cancel = schedule_subparsers.add_parser("cancel", help="取消定时任务")
    schedule_cancel.add_argument("schedule_id")
    args = parser.parse_args(argv)

    # A complete control scope is deliberately handled before ``build_service``
    # so an Agent subprocess never opens the administrator SQLite database or
    # instantiates an adapter holding a channel credential.  A partial scope
    # fails closed in ``controlled_environment`` instead of falling through.
    try:
        scoped = _controlled_client()
    except ControlError as error:
        print(str(error), file=sys.stderr)
        return 2
    if scoped is not None:
        if args.command == "agent-send-file":
            return command_controlled_agent_send_file(args)
        if args.command == "schedule":
            return command_controlled_schedule(args)
        # Do not let an Agent child process reach administrator commands
        # (including ``send-file`` which opens the SQLite database directly).
        print("受控 Agent 环境只允许 xma-send-file 和 xma schedule", file=sys.stderr)
        return 2
    service = build_service(args.data_dir)
    if args.command == "init":
        return command_init(service)
    if args.command == "doctor":
        return command_doctor(service)
    if args.command == "send-file":
        return command_send_file(service, args)
    if args.command == "agent-send-file":
        return command_agent_send_file(service, args)
    if args.command == "schedule":
        if args.schedule_command == "create":
            return command_schedule_create(service, args)
        if args.schedule_command == "list":
            return command_schedule_list(service, args)
        return command_schedule_cancel(service, args)
    return command_serve(service, args)


if __name__ == "__main__":
    raise SystemExit(main())
