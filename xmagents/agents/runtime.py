"""Agent runtime abstractions.

The runtime deliberately does not know about FastAPI, channel adapters, or the
database schema.  ``AgentRuntime`` receives a small settings mapping and
optional persistence callbacks, which keeps it useful in tests and makes it
safe for the service layer to construct one runtime per bound peer.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import shutil
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_EFFORTS = ("low", "medium", "high", "xhigh", "max")
SUPPORTED_PERMISSION_MODES = (
    "bypassPermissions",
    "default",
    "acceptEdits",
    "plan",
    "dontAsk",
    "auto",
)


@dataclass(slots=True)
class RuntimeEvent:
    """A provider-neutral event emitted while an agent turn is running."""

    kind: str
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "content": self.content, "metadata": self.metadata}


@dataclass(slots=True)
class AgentSettings:
    """Normalised settings accepted by the two built-in providers.

    ``extra`` carries serialisable Claude SDK options not represented by the
    common fields.  It is intentionally a dict instead of a strict pydantic
    model so new SDK fields can be adopted without changing this module.
    """

    agent_id: str = ""
    provider: str = "anthropic"
    model: str | None = None
    api_key: str | None = None
    api_url: str | None = None
    permission_mode: str = "bypassPermissions"
    effort: str = "medium"
    system_prompt: str | None = None
    workspace: str | Path | None = None
    max_context_messages: int | None = None
    max_context_tokens: int | None = None
    memory_enabled: bool = True
    knowledge_enabled: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: "AgentSettings | Mapping[str, Any] | Any") -> "AgentSettings":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            source = dict(value)
        else:
            source = {}
            for name in cls.__dataclass_fields__:
                if hasattr(value, name):
                    source[name] = getattr(value, name)
            # AppService rows use config_json decoded to ``config`` in some
            # callers.  Merge it last while preserving explicit row columns.
            config = getattr(value, "config", None)
            if isinstance(config, Mapping):
                source.update(config)
        known = set(cls.__dataclass_fields__) - {"extra"}
        extra = dict(source.get("extra") or {})
        extra.update({k: v for k, v in source.items() if k not in known and k != "extra"})
        source["extra"] = extra
        source = {k: v for k, v in source.items() if k in cls.__dataclass_fields__}
        if source.get("workspace") is not None:
            source["workspace"] = str(source["workspace"])
        return cls(**source)

    def update(self, **values: Any) -> None:
        for key, value in values.items():
            if key in self.__dataclass_fields__:
                setattr(self, key, value)
            else:
                self.extra[key] = value

    def status(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "provider": self.provider,
            "model": self.model or "default",
            "effort": self.effort,
            "permission_mode": self.permission_mode,
            "workspace": str(self.workspace) if self.workspace else None,
            "memory_enabled": self.memory_enabled,
            "knowledge_enabled": self.knowledge_enabled,
            "max_context_messages": self.max_context_messages,
            "max_context_tokens": self.max_context_tokens,
        }


class Provider(ABC):
    """Provider contract implemented by Anthropic and OpenAI-compatible APIs."""

    def __init__(self, settings: AgentSettings):
        self.settings = settings

    @abstractmethod
    async def stream(self, prompt: str, history: list[dict[str, str]], *, session_id: str = "default") -> AsyncIterator[RuntimeEvent]:
        """Yield text/tool/result events for one turn."""
        if False:  # pragma: no cover - makes this a genuine async generator
            yield RuntimeEvent("result")

    async def reset(self) -> None:
        """Drop provider-side session state (single-shot providers are no-op)."""

    async def close(self) -> None:
        await self.reset()

    async def interrupt(self) -> None:
        """Interrupt a running turn when the provider supports it."""

    async def set_effort(self, effort: str) -> None:
        self.settings.effort = effort


class AnthropicProvider(Provider):
    """Thin lazy wrapper around ``claude-agent-sdk``'s interactive client."""

    def __init__(self, settings: AgentSettings, sdk_client: Any | None = None):
        super().__init__(settings)
        self.client = sdk_client
        self._connected = False
        self._sdk_module: Any | None = None
        self.last_options_warning: str | None = None

    def _load_sdk(self) -> Any:
        if self._sdk_module is None:
            try:
                import claude_agent_sdk as sdk  # type: ignore
            except ImportError as error:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "Anthropic runtime requires claude-agent-sdk; install project dependencies first"
                ) from error
            self._sdk_module = sdk
        return self._sdk_module

    def _options(self) -> Any:
        sdk = self._load_sdk()
        options: dict[str, Any] = dict(self.settings.extra.get("claude_options") or {})
        # Keep only values understood by the installed SDK.  This allows a
        # config saved for a newer SDK to continue running on an older one.
        try:
            fields = set(inspect.signature(sdk.ClaudeAgentOptions).parameters)
        except (TypeError, ValueError):
            fields = set()
        common = {
            "model": self.settings.model,
            "permission_mode": self.settings.permission_mode,
            "system_prompt": (self.settings.system_prompt or "") + self.workspace_instruction(),
            "cwd": str(self.settings.workspace) if self.settings.workspace else None,
            "include_partial_messages": True,
            "effort": self.settings.effort,
        }
        # Explicit extra options win over defaults, but never pass unknown
        # arguments to the SDK dataclass.
        for key, value in common.items():
            if value is not None and (not fields or key in fields):
                options.setdefault(key, value)
        for key in (
            "tools", "allowed_tools", "disallowed_tools", "mcp_servers", "strict_mcp_config",
            "max_turns", "max_budget_usd", "thinking", "setting_sources", "plugins", "env",
            "add_dirs", "sandbox", "hooks", "agents", "output_format", "extra_args",
            "continue_conversation", "resume", "session_id", "fork_session", "fallback_model",
            "include_hook_events", "user", "enable_file_checkpointing", "betas", "task_budget",
            "max_buffer_size", "cli_path", "settings", "permission_prompt_tool_name",
            "skills", "session_store", "session_store_flush", "load_timeout_ms", "max_thinking_tokens",
        ):
            if key in self.settings.extra and (not fields or key in fields):
                options[key] = self.settings.extra[key]
        # These simple toggles are deliberately exposed as settings rather
        # than requiring an administrator to memorise Claude Code tool names.
        # A user-supplied explicit tools list remains authoritative.
        enabled_web_tools: list[str] = []
        if self.settings.extra.get("enable_web_search"):
            enabled_web_tools.append("WebSearch")
        if self.settings.extra.get("enable_web_fetch"):
            enabled_web_tools.append("WebFetch")
        if enabled_web_tools:
            configured_tools = options.get("tools")
            if configured_tools is None:
                options["tools"] = enabled_web_tools
            elif isinstance(configured_tools, list):
                options["tools"] = list(dict.fromkeys([*configured_tools, *enabled_web_tools]))
        env = dict(options.get("env") or {})
        # ``use_local_claude_code_login`` was the original built-in profile
        # marker.  It now means "use the service process authentication as
        # supplied": that can be an existing Claude login, or inherited
        # ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL billing credentials.
        # Keep it as a compatibility marker, without stripping inherited env.
        use_local_claude_login = bool(
            self.settings.extra.get("use_local_claude_code")
            or self.settings.extra.get("use_local_claude_code_login")
        )
        if use_local_claude_login and (not fields or "cli_path" in fields):
            # The SDK bundles a CLI as a fallback, but this profile represents
            # the Claude Code installation owned by the service user. Prefer
            # that executable when it is on PATH so its version, login and API
            # billing behavior match `claude` used by the administrator.
            local_cli = shutil.which("claude")
            if local_cli:
                options.setdefault("cli_path", local_cli)
        if self.settings.workspace and not use_local_claude_login:
            config_dir = Path(self.settings.workspace) / ".claude-config"
            config_dir.mkdir(parents=True, exist_ok=True)
            try:
                config_dir.chmod(0o700)
            except OSError:
                pass
            env.setdefault("CLAUDE_CONFIG_DIR", str(config_dir))
        # These are not credentials. They scope ``xma agent-send-file`` and
        # ``xma schedule`` calls made by the Agent to the owning workspace.
        # The service still checks database bindings and canonical file paths.
        env.update({key: value for key, value in self._workspace_cli_environment().items() if key not in env})
        # SDK subprocesses merge options.env over their inherited process
        # environment. Never write an empty token here, because it would
        # silently disable a valid process-level credential.  The built-in
        # Claude Code profile intentionally leaves inherited values untouched:
        # a service user may authenticate with either `claude login` or API
        # billing environment variables.
        if not use_local_claude_login:
            auth_token = self.settings.api_key or os.getenv("ANTHROPIC_AUTH_TOKEN")
            if auth_token:
                env.setdefault("ANTHROPIC_AUTH_TOKEN", auth_token)
            if self.settings.api_url:
                env.setdefault("ANTHROPIC_BASE_URL", self.settings.api_url)
            elif os.getenv("ANTHROPIC_BASE_URL"):
                env.setdefault("ANTHROPIC_BASE_URL", os.environ["ANTHROPIC_BASE_URL"])
        options["env"] = {str(key): str(value) for key, value in env.items() if value is not None}
        if not use_local_claude_login and "setting_sources" not in options and (not fields or "setting_sources" in fields):
            # Regular profiles use their project-scoped default for per-Agent
            # isolation. The built-in local Claude Code profile leaves this
            # unset, so the CLI retains its normal user/project behavior.
            options["setting_sources"] = ["project"]
        if options.get("mcp_servers") and (not fields or "strict_mcp_config" in fields):
            options.setdefault("strict_mcp_config", True)

        # ``claude_options`` is user-editable JSON. Keep configurations made
        # against a newer SDK usable after a local dependency downgrade.
        if fields:
            options = {key: value for key, value in options.items() if key in fields}

        advanced = self.settings.extra.get("advanced_python")
        if advanced:
            fallback_options = dict(options)
            base_options = dict(options)
            try:
                namespace: dict[str, Any] = {
                    "ClaudeAgentOptions": sdk.ClaudeAgentOptions,
                    "sdk": sdk,
                }
                exec(compile(str(advanced), f"<agent:{self.settings.agent_id}:options>", "exec"), namespace, namespace)
                builder = namespace.get("build_options")
                if callable(builder):
                    # A factory may mutate the supplied copy, return a
                    # mapping, or construct the SDK options object itself.
                    built = builder(base_options)
                    if built is None:
                        options = base_options
                    elif isinstance(built, Mapping):
                        options = dict(built)
                    elif isinstance(built, sdk.ClaudeAgentOptions):
                        self.last_options_warning = None
                        return built
                    else:
                        raise TypeError("build_options 必须返回 dict、ClaudeAgentOptions 或 None")
                self.last_options_warning = None
            except Exception as error:
                # Advanced Python is administrator-controlled. A syntax or
                # runtime error must not prevent the base agent from working.
                options = fallback_options
                self.last_options_warning = f"高级配置已回退：{type(error).__name__}"
        if fields:
            options = {key: value for key, value in options.items() if key in fields}
        try:
            return sdk.ClaudeAgentOptions(**options)
        except TypeError:
            # A custom SDK build may not expose a signature; progressively
            # remove optional values until construction succeeds.
            optional = ("effort", "include_partial_messages", "env", "system_prompt", "cwd")
            for key in optional:
                options.pop(key, None)
                try:
                    return sdk.ClaudeAgentOptions(**options)
                except TypeError:
                    continue
            raise

    def _workspace_cli_environment(self) -> dict[str, str]:
        """Environment passed to Claude Code's workspace-scoped subprocess.

        The service command validates both binding and real paths, while this
        scope prevents an Agent in one workspace from accidentally targeting a
        different Agent by using the convenience command.
        """

        values = {
            "XMAGENTS_AGENT_ID": self.settings.agent_id,
            "XMAGENTS_WORKSPACE": str(self.settings.workspace or ""),
            "XMAGENTS_CONTROL_SOCKET": str(self.settings.extra.get("control_socket") or ""),
            "XMAGENTS_CONTROL_SECRET": str(self.settings.extra.get("control_secret") or ""),
        }
        return {key: value for key, value in values.items() if value}

    @staticmethod
    def workspace_instruction() -> str:
        """Operational contract for workspace files and deterministic tasks."""

        return (
            "\n\nXMAgent 工作区规则：当前 cwd 是此用户独立工作区。收到附件时，只读取提示中给出的"
            "工作区相对路径。发送文件使用 `xma agent-send-file <relative-path> [--caption ...]`；"
            "不得发送工作区外文件。创建定时任务时，先把自然语言时间转换为明确 ISO 时间、秒数或"
            "五字段 Cron，再使用 `xma schedule create --at/--every/--cron --prompt ...`。"
        )

    async def _ensure_client(self) -> Any:
        if self.client is None:
            sdk = self._load_sdk()
            self.client = sdk.ClaudeSDKClient(options=self._options())
        if not self._connected:
            connector = getattr(self.client, "connect", None)
            if connector is not None:
                result = connector()
                if inspect.isawaitable(result):
                    await result
            # Injectable test clients and custom transports can already be
            # connected and omit connect().  They are still one logical
            # provider session, so do not inject the restoration prompt on
            # every turn.
            self._connected = True
        return self.client

    async def stream(self, prompt: str, history: list[dict[str, str]], *, session_id: str = "default") -> AsyncIterator[RuntimeEvent]:
        # ClaudeSDKClient maintains its own session once connected.  The
        # application-level transcript is only needed when rebuilding it
        # after a restart, /new, or a context-window trim.
        restore_history = bool(history) and not self._connected
        client = await self._ensure_client()
        if restore_history:
            context = "\n".join(f"{item.get('role')}: {item.get('content')}" for item in history[-20:])
            prompt = f"以下是此 Agent 在服务重启前的近期对话，仅用于恢复上下文：\n<context>\n{context}\n</context>\n\n{prompt}"
        # Older SDK-compatible test/custom clients may still expose the
        # original ``query(prompt)`` signature.  Keep session propagation for
        # current clients while remaining compatible with that contract.
        try:
            await client.query(prompt, session_id=session_id)
        except TypeError as error:
            if "session_id" not in str(error):
                raise
            await client.query(prompt)
        receiver = getattr(client, "receive_response", None) or getattr(client, "receive_messages", None)
        if receiver is None:
            raise RuntimeError("Claude SDK client does not expose receive_response")
        async for message in receiver():
            event = _claude_event(message)
            if event is not None:
                yield event

    async def reset(self) -> None:
        if self.client is not None:
            try:
                await self.client.disconnect()
            except (AttributeError, RuntimeError):
                pass
        self.client = None
        self._connected = False

    async def interrupt(self) -> None:
        if self.client is not None and hasattr(self.client, "interrupt"):
            await self.client.interrupt()


def _redact_tool_input(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """Return a small, serialisable tool-input preview safe for status UIs."""

    if depth > 4:
        return "[truncated]"
    lowered = key.lower().replace("-", "_")
    if any(token in lowered for token in ("password", "secret", "token", "api_key", "apikey", "authorization", "cookie")):
        return "[redacted]"
    if isinstance(value, Mapping):
        return {str(item_key): _redact_tool_input(item_value, key=str(item_key), depth=depth + 1)
                for item_key, item_value in list(value.items())[:30]}
    if isinstance(value, (list, tuple)):
        return [_redact_tool_input(item, depth=depth + 1) for item in list(value)[:30]]
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:500] + "..."
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:500]


def _claude_event(message: Any) -> RuntimeEvent | None:
    """Convert SDK dataclasses (or test doubles) into stable events."""

    name = type(message).__name__
    if name == "StreamEvent":
        event = getattr(message, "event", {}) or {}
        delta = event.get("delta", {}) if isinstance(event, Mapping) else {}
        text = delta.get("text", "") if isinstance(delta, Mapping) else ""
        if text:
            return RuntimeEvent("text", str(text), {"stream": True})
        # Only surface actionable tool starts.  Forwarding every raw stream
        # frame would leak provider internals (and can expose tool arguments)
        # while flooding a WebUI event stream with pings and message deltas.
        block = event.get("content_block", {}) if isinstance(event, Mapping) else {}
        block_type = block.get("type") if isinstance(block, Mapping) else ""
        if event.get("type") == "content_block_start" and block_type in {"tool_use", "server_tool_use"}:
            tool_name = str(block.get("name") or "tool")
            return RuntimeEvent("tool", tool_name, {
                "tools": [{
                    "id": str(block.get("id") or ""),
                    "name": tool_name,
                    "input": _redact_tool_input(block.get("input") or {}),
                    "stream": True,
                }],
            })
        return None
    if name == "AssistantMessage":
        blocks = getattr(message, "content", message)
        if isinstance(blocks, str):
            return RuntimeEvent("text", blocks)
        text_parts: list[str] = []
        tools: list[dict[str, Any]] = []
        for block in blocks or []:
            block_name = type(block).__name__
            if block_name == "TextBlock" or hasattr(block, "text"):
                text_parts.append(str(getattr(block, "text", "")))
            elif block_name == "ToolUseBlock" or hasattr(block, "name"):
                tools.append({"id": getattr(block, "id", ""), "name": getattr(block, "name", ""),
                              "input": _redact_tool_input(getattr(block, "input", {}))})
        if text_parts and tools:
            # A final SDK message may contain both narration and a tool call.
            # Keep the text reply while retaining a redacted tool status for
            # callers that render real-time tool requests.
            return RuntimeEvent("text", "".join(text_parts), {"tools": tools})
        if tools:
            return RuntimeEvent("tool", metadata={"tools": tools}, content="; ".join(t["name"] for t in tools))
        if text_parts:
            return RuntimeEvent("text", "".join(text_parts))
    if name == "ResultMessage":
        return RuntimeEvent("result", str(getattr(message, "result", "") or ""), {
            "subtype": getattr(message, "subtype", ""),
            "is_error": bool(getattr(message, "is_error", False)),
            "session_id": getattr(message, "session_id", None),
            "usage": getattr(message, "usage", None),
            "total_cost_usd": getattr(message, "total_cost_usd", None),
            "errors": getattr(message, "errors", None),
        })
    if name == "SystemMessage":
        return RuntimeEvent("system", metadata={"subtype": getattr(message, "subtype", ""), "data": getattr(message, "data", {})})
    # UserMessage is an SDK transcript echo.  The gateway already persisted
    # and displayed the inbound text, so sending it back would duplicate the
    # user's message in the outbound reply.
    return None


class OpenAIProvider(Provider):
    """OpenAI-compatible Chat Completions streaming provider."""

    def __init__(self, settings: AgentSettings, client: Any | None = None):
        super().__init__(settings)
        self.client = client
        self._owns_client = client is None

    def _endpoint(self) -> str:
        base = (self.settings.api_url or "https://api.openai.com/v1").rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + ("" if base.endswith("/v1") else "/v1") + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    async def stream(self, prompt: str, history: list[dict[str, str]], *, session_id: str = "default") -> AsyncIterator[RuntimeEvent]:
        # Chat Completions has no server-side conversation/session identifier.
        # Keep the provider contract uniform so callers can always pass the
        # persisted conversation session without special-casing OpenAI.
        del session_id
        try:
            import httpx
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("OpenAI runtime requires httpx") from error
        messages = list(history)
        if self.settings.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self.settings.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.settings.model or "gpt-4o-mini",
            "messages": messages,
            "stream": True,
        }
        payload.update(dict(self.settings.extra.get("openai_options") or {}))
        client = self.client
        if client is None:
            client = httpx.AsyncClient(timeout=120, trust_env=False)
            self.client = client
        async with client.stream("POST", self._endpoint(), headers=self._headers(), json=payload) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise RuntimeError(f"OpenAI API HTTP {response.status_code}: {body[:500].decode(errors='replace')}")
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []) or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        # Mark deltas explicitly so channel adapters can edit
                        # one Telegram message while the SSE response is in
                        # flight instead of waiting for the final aggregate.
                        yield RuntimeEvent("text", str(text), {"provider": "openai", "stream": True})
                    tool_calls = delta.get("tool_calls")
                    if tool_calls:
                        yield RuntimeEvent("tool", metadata={"tool_calls": tool_calls})
                if chunk.get("usage"):
                    yield RuntimeEvent("result", metadata={"usage": chunk["usage"]})

    async def close(self) -> None:
        if self.client is not None and self._owns_client:
            await self.client.aclose()
        self.client = None


class MailboxOperationCancelled(RuntimeError):
    """A queued turn was deliberately discarded by a higher-priority reset."""


class SerialMailbox:
    """A small FIFO mailbox guaranteeing one active turn per agent."""

    def __init__(self):
        self._queue: asyncio.Queue[tuple[Callable[[], Awaitable[Any]], asyncio.Future[Any]]] = asyncio.Queue()
        self._worker: asyncio.Task[Any] | None = None
        self._closed = False

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._closed = False
            self._worker = asyncio.create_task(self._run(), name="xmagents-agent-mailbox")

    async def submit(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        if self._closed:
            raise RuntimeError("agent mailbox is closed")
        await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        await self._queue.put((operation, future))
        return await future

    async def cancel_pending(self) -> None:
        """Cancel queued operations while leaving the active turn alone."""
        while not self._queue.empty():
            _, future = self._queue.get_nowait()
            if not future.done():
                future.set_exception(MailboxOperationCancelled("该请求已被 /new 清除"))
            self._queue.task_done()

    async def _run(self) -> None:
        while not self._closed:
            try:
                operation, future = await self._queue.get()
            except asyncio.CancelledError:
                break
            if future.cancelled():
                self._queue.task_done()
                continue
            try:
                future.set_result(await operation())
            except asyncio.CancelledError as error:
                if not future.done():
                    future.set_exception(error)
                raise
            except Exception as error:  # keep mailbox alive after one failed turn
                if not future.done():
                    future.set_exception(error)
            finally:
                self._queue.task_done()

    async def close(self) -> None:
        self._closed = True
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        while not self._queue.empty():
            _, future = self._queue.get_nowait()
            if not future.done():
                future.set_exception(RuntimeError("agent mailbox is closed"))
            self._queue.task_done()


@dataclass(slots=True)
class TurnContext:
    channel: str = ""
    user_id: str = ""
    peer_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    attachments: list[Any] = field(default_factory=list)
    send_message: Any | None = None
    send_file: Any | None = None
    session_id: str = "default"


class AgentRuntime:
    """One conversational agent with commands, plugins and injectable stores."""

    def __init__(self, settings: AgentSettings | Mapping[str, Any] | Any,
                 *, provider: Provider | None = None, plugin_loader: Any | None = None,
                 memory: Any | None = None, knowledge: Any | None = None,
                 save_message: Callable[..., Any] | None = None,
                 save_settings: Callable[[AgentSettings], Any] | None = None,
                 reset_conversation: Callable[[TurnContext], Any] | None = None,
                 history_loader: Callable[[TurnContext], list[dict[str, str]]] | None = None):
        self.settings = AgentSettings.from_value(settings)
        self.provider = provider or (OpenAIProvider(self.settings) if self.settings.provider.lower() in {"openai", "openai-compatible", "openai_compatible"} else AnthropicProvider(self.settings))
        self.plugin_loader = plugin_loader
        self.memory = memory
        self.knowledge = knowledge
        self.save_message = save_message
        self.save_settings = save_settings
        self.reset_conversation = reset_conversation
        # ``history`` remains the most recently used route for backwards
        # compatibility with embedders. The authoritative in-memory state is
        # route-scoped: a deliberately shared Agent must never feed one peer's
        # OpenAI transcript into another peer's request.
        self.history: list[dict[str, str]] = []
        self._history_by_route: dict[str, list[dict[str, str]]] = {}
        self.history_loader = history_loader
        self.mailbox = SerialMailbox()
        self._active = False
        self._turn_started = 0.0
        self._generation = 0
        self._memory_tasks: set[asyncio.Task[Any]] = set()

    async def submit(self, text: str, context: TurnContext | Mapping[str, Any] | None = None,
                     on_event: Callable[[RuntimeEvent], Awaitable[None] | None] | None = None) -> list[RuntimeEvent]:
        ctx = context if isinstance(context, TurnContext) else TurnContext(**dict(context or {}))
        command, _ = parse_command(text)
        if command == "new":
            # Mark an active response obsolete before interrupting it. Some
            # providers can emit one final frame after interrupt(), and that
            # frame must not become a normal reply after /new.
            self._generation += 1
            try:
                await self.provider.interrupt()
            except Exception:
                pass
            await self.mailbox.cancel_pending()
        generation = self._generation
        try:
            return await self.mailbox.submit(lambda: self.handle(text, ctx, on_event=on_event, generation=generation))
        except MailboxOperationCancelled:
            # A channel worker should keep running after a queued inbound
            # message is discarded by /new, rather than propagate task
            # cancellation through its polling loop.
            return []

    async def handle(self, text: str, context: TurnContext | None = None,
                     *, on_event: Callable[[RuntimeEvent], Awaitable[None] | None] | None = None,
                     generation: int | None = None) -> list[RuntimeEvent]:
        context = context or TurnContext()
        command, args = parse_command(text)
        if command:
            result = await self._command(command, args, context)
            events = [RuntimeEvent("text", result)] if result else []
            # Commands are part of the conversation history too.  Persisting
            # both sides makes /new, /effort and plugin use visible in the
            # WebUI without duplicating ordinary chat persistence.
            if self.save_message:
                await _maybe_await(self.save_message("user", text, context=context))
                if result:
                    await _maybe_await(self.save_message("assistant", result, context=context))
            await self._emit(events, on_event)
            return events
        return await self._chat(text, context, on_event=on_event, generation=generation)

    async def _command(self, command: str, args: str, context: TurnContext) -> str:
        if command == "new":
            self._history_for(context).clear()
            await self.provider.reset()
            if self.reset_conversation:
                result = self.reset_conversation(context)
                if inspect.isawaitable(result):
                    await result
            return "已清除当前对话上下文。"
        if command == "effort":
            if not args:
                return f"当前思考强度：{self.settings.effort}"
            value = args.strip().lower()
            if value not in SUPPORTED_EFFORTS:
                return "用法：/effort low|medium|high|xhigh|max"
            self.settings.effort = value
            await self.provider.set_effort(value)
            # Effort is an option fixed at Claude session creation. Recreate
            # the provider so the next turn definitely uses the new setting.
            if isinstance(self.provider, AnthropicProvider):
                await self.provider.reset()
            if self.save_settings:
                result = self.save_settings(self.settings)
                if inspect.isawaitable(result):
                    await result
            return f"思考强度已设置为 {value}。"
        if command == "status":
            return self.status_text(context)
        if command == "help":
            return self.help_text()
        if command in {"sendfile", "send_file"}:
            if not args:
                return "用法：/sendfile relative/path [caption]"
            relative, _, caption = args.partition(" ")
            if context.send_file is None:
                return "当前渠道不支持发送文件。"
            try:
                result = context.send_file(relative, caption=caption.strip() or None)
                if inspect.isawaitable(result):
                    result = await result
                return "文件发送请求已提交。" if getattr(result, "ok", True) else f"文件发送失败：{getattr(result, 'error', result)}"
            except Exception as error:
                return f"文件发送失败：{error}"
        plugin = self._plugin(command)
        if plugin is not None:
            result = await self._run_plugin(plugin, args, context)
            return result
        return "未知命令。发送 /help 查看可用命令。"

    async def _chat(self, text: str, context: TurnContext, *, on_event: Callable[..., Any] | None = None,
                    generation: int | None = None) -> list[RuntimeEvent]:
        prompt = text
        if context.attachments:
            files = []
            for attachment in context.attachments:
                path = getattr(attachment, "path", None) or (attachment.get("path") if isinstance(attachment, Mapping) else None)
                if path:
                    files.append(str(path))
            if files:
                prompt += "\n\n用户发送了文件，请读取以下路径：\n" + "\n".join(f"- {path}" for path in files)
        if self.memory is not None and self.settings.memory_enabled:
            try:
                memory_text = await _maybe_await(self.memory.context(self.settings.agent_id, text))
                if memory_text:
                    prompt = f"<memory>\n{memory_text}\n</memory>\n\n{prompt}"
            except Exception:
                pass
        if self.knowledge is not None and self.settings.knowledge_enabled:
            try:
                knowledge_text = await _maybe_await(self.knowledge.context(self.settings.agent_id, text))
                if knowledge_text:
                    prompt = f"<knowledge>\n{knowledge_text}\n</knowledge>\n\n{prompt}"
            except Exception:
                pass
        # Include the current user message in the calculation.  Trimming it
        # only before appending would exceed a configured cap on every turn.
        history = self._history_for(context)
        history.append({"role": "user", "content": prompt})
        trimmed = self._trim_history(history)
        if trimmed and isinstance(self.provider, AnthropicProvider):
            # A ClaudeSDKClient keeps its full remote conversation even after
            # local history is pruned.  Rebuild it so the configured context
            # limit actually takes effect, then restore only retained turns.
            await self.provider.reset()
        if self.save_message:
            await _maybe_await(self.save_message("user", text, context=context))
        events: list[RuntimeEvent] = []
        self._active = True
        self._turn_started = time.monotonic()
        try:
            try:
                stream = self.provider.stream(prompt, history[:-1], session_id=context.session_id or "default")
            except TypeError:
                # Existing third-party providers implementing the original
                # two-argument contract remain usable.
                stream = self.provider.stream(prompt, self.history[:-1])
            async for event in stream:
                if generation is not None and generation != self._generation:
                    break
                events.append(event)
                await self._emit([event], on_event)
        except Exception as error:
            event = RuntimeEvent("error", str(error), {"provider": self.settings.provider})
            events.append(event)
            await self._emit([event], on_event)
        finally:
            self._active = False
        if generation is not None and generation != self._generation:
            # /new will run immediately after the active mailbox item and
            # clear local context. Do not persist or return stale partials.
            return []
        # Claude emits both partial StreamEvents and a final AssistantMessage.
        # Persist/send the final message when present; otherwise fall back to
        # the accumulated stream (for providers that only emit deltas).
        final_parts = [event.content for event in events if event.kind == "text" and not event.metadata.get("stream")]
        stream_parts = [event.content for event in events if event.kind == "text" and event.metadata.get("stream")]
        text_reply = "".join(final_parts or stream_parts)
        if text_reply:
            history.append({"role": "assistant", "content": text_reply})
            if self.save_message:
                await _maybe_await(self.save_message("assistant", text_reply, context=context))
            if self.memory is not None and self.settings.memory_enabled:
                self._observe_memory_in_background(text, text_reply)
        return events

    def _observe_memory_in_background(self, user_text: str, assistant_text: str) -> None:
        """Queue local memory maintenance after the response has completed."""

        async def observe() -> None:
            try:
                await _maybe_await(self.memory.observe(self.settings.agent_id, user_text, assistant_text))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # Memory is an enhancement.  It must not turn an already
                # delivered reply into a failed channel request.
                self._audit_memory_error(error)

        task = asyncio.create_task(observe(), name=f"xmagents-memory-{self.settings.agent_id}")
        self._memory_tasks.add(task)
        task.add_done_callback(self._memory_tasks.discard)

    def _audit_memory_error(self, error: Exception) -> None:
        service = getattr(self.memory, "service", None)
        database = getattr(service, "db", None)
        audit = getattr(database, "audit", None)
        if not callable(audit):
            return
        try:
            audit("memory_observe_failed", target=self.settings.agent_id, detail={"error": type(error).__name__})
        except Exception:
            pass

    async def _emit(self, events: list[RuntimeEvent], callback: Callable[..., Any] | None) -> None:
        if not callback:
            return
        for event in events:
            result = callback(event)
            if inspect.isawaitable(result):
                await result

    @staticmethod
    def _route_key(context: TurnContext) -> str:
        """Return the persistent conversation identity for a runtime turn."""

        return str(context.peer_id or "default")

    def _history_for(self, context: TurnContext) -> list[dict[str, str]]:
        """Get one peer's local transcript, loading it only on first use."""

        route_key = self._route_key(context)
        history = self._history_by_route.get(route_key)
        if history is None:
            try:
                loaded = self.history_loader(context) if self.history_loader else None
            except Exception:
                # Durable history is a convenience for a rebuilt runtime. A
                # database read failure must not prevent a new turn from
                # starting, and must never fall back to another peer's list.
                loaded = None
            history = [
                {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
                for item in (loaded or [])
                if isinstance(item, Mapping) and item.get("role") in {"user", "assistant", "system"}
            ]
            self._history_by_route[route_key] = history
        # Existing integrations inspect ``runtime.history``. Point it at the
        # active route rather than retaining a stale/global transcript.
        self.history = history
        return history

    def _trim_history(self, history: list[dict[str, str]] | None = None) -> bool:
        history = self.history if history is None else history
        before = list(history)
        max_messages = self.settings.max_context_messages
        if max_messages is not None and max_messages > 0 and len(history) > max_messages:
            history[:] = history[-max_messages:]
        max_tokens = self.settings.max_context_tokens
        if max_tokens is not None and max_tokens > 0:
            # Approximate token count without making the runtime depend on a
            # tokenizer.  Keeping recent turns is deterministic and safe.
            total = 0
            kept: list[dict[str, str]] = []
            for message in reversed(history):
                cost = max(1, len(message.get("content", "")) // 4)
                if kept and total + cost > max_tokens:
                    break
                kept.append(message)
                total += cost
            history[:] = list(reversed(kept))
        return history != before

    def _plugin(self, command: str) -> Any | None:
        if self.plugin_loader is None:
            return None
        try:
            return self.plugin_loader.get(command, agent_id=self.settings.agent_id)
        except TypeError:
            return self.plugin_loader.get(command)

    async def _run_plugin(self, plugin: Any, args: str, context: TurnContext) -> str:
        if hasattr(self.plugin_loader, "context"):
            plugin_context = self.plugin_loader.context(self.settings, context, self)
        else:
            plugin_context = context
        handler = getattr(plugin, "handle", plugin if callable(plugin) else None)
        if handler is None:
            return "插件没有可调用的 handle 函数。"
        value = handler(plugin_context, args)
        value = await _maybe_await(value)
        return str(value or "")

    def status_text(self, context: TurnContext | None = None) -> str:
        s = self.settings.status()
        context_count = len(self._history_for(context)) if context is not None else len(self.history)
        active = "运行中" if self._active else "空闲"
        mcp_enabled = bool(self.settings.extra.get("mcp_servers"))
        queue_size = self.mailbox._queue.qsize()
        return (f"Agent {s['agent_id'] or '(未命名)'}\n"
                f"provider: {s['provider']}\nmodel: {s['model']}\n"
                f"effort: {s['effort']}\npermission: {s['permission_mode']}\n"
                f"context messages: {context_count}\nstate: {active}\n"
                f"memory: {'on' if s['memory_enabled'] else 'off'}\n"
                f"knowledge: {'on' if s['knowledge_enabled'] else 'off'}\n"
                f"mcp: {'on' if mcp_enabled else 'off'}\nqueue: {queue_size}")

    def help_text(self) -> str:
        lines = [
            "/new - 清除当前对话上下文",
            "/effort [low|medium|high|xhigh|max] - 查看或设置思考强度",
            "/status - 查看 Agent 状态、模型和思考强度",
            "/help - 查看帮助",
            "/sendfile <relative/path> [caption] - 发送工作区文件",
        ]
        if self.plugin_loader is not None:
            try:
                plugins = self.plugin_loader.list(agent_id=self.settings.agent_id)
            except TypeError:
                plugins = self.plugin_loader.list()
            for plugin in plugins or []:
                command = getattr(plugin, "command", None) or (plugin.get("command") if isinstance(plugin, Mapping) else "")
                description = getattr(plugin, "description", "") or (plugin.get("description", "") if isinstance(plugin, Mapping) else "")
                if command:
                    lines.append(f"/{command} - {description}".rstrip())
        return "可用命令：\n" + "\n".join(lines)

    async def interrupt(self) -> None:
        await self.provider.interrupt()

    async def close(self) -> None:
        await self.mailbox.close()
        tasks = tuple(self._memory_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._memory_tasks.clear()
        await self.provider.close()


class AgentRuntimeManager:
    """Own runtimes by agent id and serialize their lifecycle."""

    def __init__(self, factory: Callable[[str], AgentRuntime]):
        self.factory = factory
        self.runtimes: dict[str, AgentRuntime] = {}

    def get(self, agent_id: str) -> AgentRuntime:
        if agent_id not in self.runtimes:
            self.runtimes[agent_id] = self.factory(agent_id)
        return self.runtimes[agent_id]

    def close_agent(self, agent_id: str) -> None:
        runtime = self.runtimes.pop(agent_id, None)
        if runtime:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(runtime.close())
            except RuntimeError:
                pass

    async def close(self) -> None:
        for runtime in list(self.runtimes.values()):
            await runtime.close()
        self.runtimes.clear()


def parse_command(text: str) -> tuple[str | None, str]:
    match = re.match(r"^\s*/([A-Za-z][\w-]*)(?:\s+(.*))?$", text or "", re.S)
    return (match.group(1).lower(), (match.group(2) or "").strip()) if match else (None, "")


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


__all__ = [
    "AgentRuntime", "AgentRuntimeManager", "AgentSettings", "AnthropicProvider", "OpenAIProvider", "Provider",
    "RuntimeEvent", "SerialMailbox", "TurnContext", "MailboxOperationCancelled", "SUPPORTED_EFFORTS",
    "SUPPORTED_PERMISSION_MODES", "parse_command",
]
