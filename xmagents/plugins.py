"""Filesystem plugin loader and plugin execution context.

Plugins are intentionally tiny Python modules in ``plugins/<name>/<name>.py``.
They may export ``PLUGIN`` metadata and either ``handle(ctx, args)`` or a
command-named callable.  Loading is isolated per module; a broken optional
plugin is reported to the caller rather than preventing the application from
starting.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import ipaddress
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class PluginContext:
    """Capabilities intentionally exposed to plugin code."""

    agent: Any
    settings: Any
    channel: str = ""
    user_id: str = ""
    peer_id: str = ""
    workspace: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    send_message: Any | None = None
    send_file: Any | None = None

    async def reply(self, text: str, **kwargs: Any) -> Any:
        if self.send_message is None:
            return None
        result = self.send_message(text, **kwargs)
        return await result if inspect.isawaitable(result) else result

    async def file(self, path: str | Path, **kwargs: Any) -> Any:
        if self.send_file is None:
            return None
        result = self.send_file(path, **kwargs)
        return await result if inspect.isawaitable(result) else result


@dataclass(slots=True)
class LoadedPlugin:
    name: str
    command: str
    description: str = ""
    usage: str = ""
    module: Any = None
    error: str | None = None

    async def handle(self, context: PluginContext, args: str) -> str:
        if self.error:
            return f"插件 {self.name} 加载失败：{self.error}"
        handler = getattr(self.module, "handle", None)
        if handler is None:
            handler = getattr(self.module, self.command, None)
        if handler is None:
            return f"插件 {self.name} 没有定义 handle(ctx, args)"
        try:
            value = handler(context, args)
            if inspect.isawaitable(value):
                value = await value
            return str(value or "")
        except Exception as error:  # plugin failures must not crash mailbox
            return f"插件 {self.name} 执行失败：{error}"


class PluginLoader:
    """Discover and load plugins under one or more directories."""

    def __init__(self, root: str | Path = "plugins", *, enabled: Mapping[str, bool] | None = None):
        self.root = Path(root)
        self.enabled = dict(enabled or {})
        self._plugins: dict[str, LoadedPlugin] = {}
        self._agent_enabled: dict[str, set[str]] = {}
        self.reload()

    def reload(self) -> dict[str, LoadedPlugin]:
        self._plugins = {}
        if not self.root.exists():
            return self._plugins
        for module_path in sorted(self.root.glob("*/[!_]*.py")):
            name = module_path.parent.name
            try:
                spec = importlib.util.spec_from_file_location(f"xmagents_user_plugin_{name}", module_path)
                if spec is None or spec.loader is None:
                    raise ImportError("cannot create module spec")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                metadata = dict(getattr(module, "PLUGIN", {}) or {})
                command = str(metadata.get("command") or name).lstrip("/").lower()
                self._plugins[command] = LoadedPlugin(
                    name=str(metadata.get("name") or name),
                    command=command,
                    description=str(metadata.get("description") or ""),
                    usage=str(metadata.get("usage") or ""),
                    module=module,
                )
            except Exception as error:
                self._plugins[name.lower()] = LoadedPlugin(name=name, command=name.lower(), error=str(error))
        return self._plugins

    def set_agent_enabled(self, agent_id: str, command: str, enabled: bool) -> None:
        command = command.lstrip("/").lower()
        current = self._agent_enabled.setdefault(str(agent_id), set(self._plugins))
        if enabled:
            current.add(command)
        else:
            current.discard(command)

    def set_agent_commands(self, agent_id: str, commands: Mapping[str, bool] | set[str] | None) -> None:
        """Replace the enabled plugin set for one agent.

        A missing database row means the plugin follows the global default;
        callers therefore pass a full effective set rather than only disabled
        entries.  Keeping this operation explicit makes a restarted service
        behave exactly like the WebUI did before it was stopped.
        """

        if commands is None:
            self._agent_enabled.pop(str(agent_id), None)
            return
        if isinstance(commands, Mapping):
            enabled = {str(name).lstrip("/").lower() for name, value in commands.items() if value}
        else:
            enabled = {str(name).lstrip("/").lower() for name in commands}
        self._agent_enabled[str(agent_id)] = enabled

    def get(self, command: str, *, agent_id: str | None = None) -> LoadedPlugin | None:
        command = command.lstrip("/").lower()
        plugin = self._plugins.get(command)
        if plugin is None or self.enabled.get(command, True) is False:
            return None
        if agent_id is not None and command not in self._agent_enabled.get(str(agent_id), set(self._plugins)):
            return None
        return plugin

    def list(self, *, agent_id: str | None = None) -> list[LoadedPlugin]:
        return [plugin for command, plugin in sorted(self._plugins.items()) if self.get(command, agent_id=agent_id)]

    def context(self, settings: Any, turn_context: Any, agent: Any) -> PluginContext:
        workspace = getattr(settings, "workspace", None)
        return PluginContext(
            agent=agent,
            settings=settings,
            channel=getattr(turn_context, "channel", ""),
            user_id=getattr(turn_context, "user_id", ""),
            peer_id=getattr(turn_context, "peer_id", ""),
            workspace=Path(workspace) if workspace else None,
            metadata=dict(getattr(turn_context, "metadata", {}) or {}),
            send_message=getattr(turn_context, "send_message", None),
            send_file=getattr(turn_context, "send_file", None),
        )


def local_ipv6_addresses() -> list[str]:
    """Return globally/link-local IPv6 addresses without network requests."""
    addresses: set[str] = set()
    try:
        for _, _, _, _, sockaddr in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET6):
            host = sockaddr[0].split("%", 1)[0]
            try:
                address = ipaddress.IPv6Address(host)
            except ValueError:
                continue
            if not address.is_loopback and not address.is_unspecified:
                addresses.add(host)
    except OSError:
        pass
    return sorted(addresses)


__all__ = ["LoadedPlugin", "PluginContext", "PluginLoader", "local_ipv6_addresses"]
