"""Workspace-scoped control socket for Agent subprocess helpers.

The Claude Code subprocess is intentionally not given direct database or
channel credentials.  It uses this small local protocol for the two actions
that need to cross the workspace boundary: delivering a workspace file and
managing its own scheduled tasks.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import hmac
import json
import os
import socket
import stat
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any


MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0


class ControlError(RuntimeError):
    """A safe error suitable for returning to a workspace subprocess."""


def derive_agent_secret(master_secret: bytes, agent_id: str) -> str:
    """Derive a process-local capability token scoped to one Agent id."""

    return hmac.new(master_secret, agent_id.encode("utf-8"), hashlib.sha256).hexdigest()


def _socket_is_stale(path: Path) -> bool:
    """Return whether an existing socket has no accepting server.

    Only a refused/missing endpoint is considered stale.  Timeouts and other
    failures are treated as an active or unknown owner and are never removed.
    """

    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.2)
        probe.connect(str(path))
    except OSError as error:
        return error.errno in {errno.ECONNREFUSED, errno.ENOENT}
    finally:
        probe.close()
    return False


def _prepare_socket_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    try:
        details = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISSOCK(details.st_mode):
        raise ControlError("控制 socket 路径不是可安全替换的 socket")
    if not _socket_is_stale(path):
        raise ControlError("已有 XMAgent 控制 socket 正在运行")
    path.unlink()


def _encode_response(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    except (TypeError, ValueError):
        encoded = '{"ok":false,"error":"控制服务无法序列化响应"}'.encode("utf-8")
    if len(encoded) > MAX_RESPONSE_BYTES:
        encoded = '{"ok":false,"error":"控制服务响应过大"}'.encode("utf-8")
    return encoded + b"\n"


class ControlServer:
    """One-line JSON Unix-domain socket server.

    ``handler`` owns authentication and business authorization.  Keeping the
    transport independent lets the service enforce database bindings without
    exposing persistence implementation details to the CLI module.
    """

    def __init__(self, path: str | Path, handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]):
        self.path = Path(path)
        self.handler = handler
        self._server: asyncio.AbstractServer | None = None
        self._owns_path = False

    @property
    def running(self) -> bool:
        return self._server is not None

    async def start(self) -> None:
        if self._server is not None:
            return
        _prepare_socket_path(self.path)
        try:
            self._server = await asyncio.start_unix_server(self._handle_connection, path=str(self.path), limit=MAX_REQUEST_BYTES + 1)
            self._owns_path = True
            os.chmod(self.path, 0o600)
        except Exception:
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()
                self._server = None
            # The stdlib can leave a socket node behind if binding succeeds
            # but a later setup step fails.  At this point it was created by
            # this server attempt, so mark it owned before cleanup.
            if not self._owns_path:
                try:
                    self._owns_path = stat.S_ISSOCK(self.path.lstat().st_mode)
                except FileNotFoundError:
                    pass
            self._remove_own_socket()
            raise

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._remove_own_socket()

    def _remove_own_socket(self) -> None:
        if not self._owns_path:
            return
        self._owns_path = False
        try:
            details = self.path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(details.st_mode) and not stat.S_ISLNK(details.st_mode):
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        response: dict[str, Any]
        try:
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=DEFAULT_TIMEOUT_SECONDS)
            except (asyncio.TimeoutError, ValueError, asyncio.LimitOverrunError):
                raise ControlError("控制请求无效或超时")
            if not line or len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
                raise ControlError("控制请求过大或格式无效")
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ControlError("控制请求不是有效 JSON") from error
            if not isinstance(payload, dict):
                raise ControlError("控制请求必须是对象")
            result = await asyncio.wait_for(self.handler(payload), timeout=DEFAULT_TIMEOUT_SECONDS)
            response = {"ok": True, "result": result}
        except ControlError as error:
            response = {"ok": False, "error": str(error)}
        except asyncio.TimeoutError:
            response = {"ok": False, "error": "控制请求处理超时"}
        except Exception:
            # Deliberately do not put a traceback, filesystem layout, or
            # capability values into a subprocess-visible response.
            response = {"ok": False, "error": "控制请求执行失败，请在 WebUI 查看运行日志"}
        try:
            writer.write(_encode_response(response))
            await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass


class ControlClient:
    """Synchronous client used by short-lived ``xma`` subprocesses."""

    def __init__(self, path: str | Path, *, timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self.path = str(path)
        self.timeout = max(0.1, float(timeout))

    def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            encoded = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ControlError("控制请求无法编码") from error
        if len(encoded) > MAX_REQUEST_BYTES:
            raise ControlError("控制请求过大")
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.settimeout(self.timeout)
            client.connect(self.path)
            client.sendall(encoded + b"\n")
            chunks: list[bytes] = []
            total = 0
            while True:
                block = client.recv(8192)
                if not block:
                    break
                chunks.append(block)
                total += len(block)
                if total > MAX_RESPONSE_BYTES + 1:
                    raise ControlError("控制服务响应过大")
                if b"\n" in block:
                    break
        except OSError as error:
            raise ControlError("无法连接 XMAgent 控制服务") from error
        finally:
            client.close()
        line = b"".join(chunks).split(b"\n", 1)[0]
        if not line:
            raise ControlError("控制服务未返回响应")
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ControlError("控制服务返回无效响应") from error
        if not isinstance(response, dict):
            raise ControlError("控制服务返回无效响应")
        if not response.get("ok"):
            raise ControlError(str(response.get("error") or "控制请求被拒绝"))
        result = response.get("result", {})
        if not isinstance(result, dict):
            raise ControlError("控制服务返回无效结果")
        return result


def controlled_environment() -> tuple[str, str, str] | None:
    """Return an Agent capability scope, failing closed for partial setup."""

    socket_path = os.getenv("XMAGENTS_CONTROL_SOCKET")
    secret = os.getenv("XMAGENTS_CONTROL_SECRET")
    agent_id = os.getenv("XMAGENTS_AGENT_ID")
    if not any((socket_path, secret, agent_id)):
        return None
    if not (socket_path and secret and agent_id):
        raise ControlError("受控 Agent 环境不完整，拒绝直接访问本地数据")
    return str(socket_path), str(secret), str(agent_id)


__all__ = [
    "ControlClient",
    "ControlError",
    "ControlServer",
    "controlled_environment",
    "derive_agent_secret",
]
