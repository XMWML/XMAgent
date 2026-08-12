"""Small, dependency-free helpers for keeping channel secrets out of logs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_SENSITIVE_KEYS = frozenset({
    "authorization", "authorizationtype", "token", "bot_token", "api_key",
    "apikey", "secret", "password", "proxy", "proxy_url", "mcp_headers",
    "context_token", "typing_ticket", "aes_key", "aeskey", "encrypt_key",
    "encryptkey",
})
_BEARER_RE = re.compile(r"(?i)\b(bearer|bot)\s+[A-Za-z0-9_:\-./=+]+")
_TELEGRAM_TOKEN_RE = re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{20,}\b")


def redact_secret(value: Any, *, visible: int = 4) -> str:
    """Return a stable, useful but non-reversible display representation."""

    text = str(value or "")
    if not text:
        return ""
    if len(text) <= visible * 2:
        return "***"
    return f"{text[:visible]}...{text[-visible:]}"


def redact_text(value: Any, *, secrets: tuple[str, ...] | list[str] = ()) -> str:
    """Mask common auth values embedded in an error message or URL."""

    text = str(value or "")
    text = _BEARER_RE.sub(lambda match: f"{match.group(1)} ***", text)
    text = _TELEGRAM_TOKEN_RE.sub("***", text)
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "***")
    return text


def redact_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recursively redact values whose field name is known to be secret."""

    result: dict[str, Any] = {}
    for key, item in (value or {}).items():
        normalized = str(key).lower().replace("-", "_")
        if normalized in _SENSITIVE_KEYS or any(part in normalized for part in ("token", "secret", "password", "api_key", "authorization", "aes_key", "context")):
            result[str(key)] = "***" if item not in (None, "") else ""
        elif isinstance(item, Mapping):
            result[str(key)] = redact_mapping(item)
        elif isinstance(item, list):
            result[str(key)] = [redact_mapping(entry) if isinstance(entry, Mapping) else redact_text(entry) for entry in item]
        else:
            result[str(key)] = redact_text(item)
    return result


__all__ = ["redact_mapping", "redact_secret", "redact_text"]
