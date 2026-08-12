"""Small SQLite repository used by all application layers.

The schema deliberately uses JSON columns for provider/channel-specific options;
the stable routing and delivery keys remain normalised and indexed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
  token TEXT PRIMARY KEY, expires_at TEXT NOT NULL, created_at TEXT NOT NULL, last_seen_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_profiles (
  id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, provider TEXT NOT NULL,
  base_url TEXT, models_json TEXT NOT NULL DEFAULT '[]', secret TEXT,
  options_json TEXT NOT NULL DEFAULT '{}', enabled INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS channel_accounts (
  id TEXT PRIMARY KEY, channel TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'stopped',
  token TEXT, base_url TEXT, proxy TEXT, config_json TEXT NOT NULL DEFAULT '{}',
  cursor TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS remote_peers (
  id TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES channel_accounts(id) ON DELETE CASCADE,
  external_id TEXT NOT NULL, chat_id TEXT, display_name TEXT, kind TEXT NOT NULL DEFAULT 'private',
  approved INTEGER NOT NULL DEFAULT 0, metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(account_id, external_id, chat_id)
);
CREATE TABLE IF NOT EXISTS agents (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, provider TEXT NOT NULL DEFAULT 'anthropic',
  api_profile_id TEXT REFERENCES api_profiles(id) ON DELETE SET NULL, model TEXT,
  permission_mode TEXT NOT NULL DEFAULT 'bypassPermissions', effort TEXT NOT NULL DEFAULT 'medium',
  workspace TEXT NOT NULL, config_json TEXT NOT NULL DEFAULT '{}',
  memory_enabled INTEGER NOT NULL DEFAULT 1, knowledge_base_id TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_bindings (
  id TEXT PRIMARY KEY, agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  peer_id TEXT NOT NULL REFERENCES remote_peers(id) ON DELETE CASCADE,
  active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL,
  UNIQUE(agent_id, peer_id), UNIQUE(peer_id)
);
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY, agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  route_key TEXT NOT NULL, session_id TEXT, context_token TEXT, effort TEXT,
  status TEXT NOT NULL DEFAULT 'ready', metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(agent_id, route_key)
);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  direction TEXT NOT NULL, sender TEXT, content TEXT NOT NULL, message_type TEXT NOT NULL DEFAULT 'text',
  metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_created ON messages(conversation_id, created_at);
CREATE TABLE IF NOT EXISTS inbox (
  id TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES channel_accounts(id) ON DELETE CASCADE,
  external_event_id TEXT NOT NULL, peer_id TEXT, payload_json TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'received', error TEXT, received_at TEXT NOT NULL, processed_at TEXT,
  UNIQUE(account_id, external_event_id)
);
CREATE TABLE IF NOT EXISTS outbox (
  id TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES channel_accounts(id) ON DELETE CASCADE,
  peer_id TEXT, conversation_id TEXT, kind TEXT NOT NULL DEFAULT 'text', payload_json TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
  available_at TEXT NOT NULL, lease_until TEXT, last_error TEXT, created_at TEXT NOT NULL, sent_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_outbox_pending ON outbox(state, available_at);
CREATE TABLE IF NOT EXISTS memory_entries (
  id TEXT PRIMARY KEY, agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  kind TEXT NOT NULL DEFAULT 'fact', content TEXT NOT NULL, source_message_id TEXT,
  enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge_bases (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, source_path TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mcp_servers (
  id TEXT PRIMARY KEY, agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  name TEXT NOT NULL, config_json TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(agent_id, name)
);
CREATE TABLE IF NOT EXISTS plugins (
  name TEXT PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT 1, config_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_plugins (
  agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  plugin_name TEXT NOT NULL REFERENCES plugins(name) ON DELETE CASCADE,
  enabled INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(agent_id, plugin_name)
);
CREATE TABLE IF NOT EXISTS schedules (
  id TEXT PRIMARY KEY, agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  peer_id TEXT REFERENCES remote_peers(id) ON DELETE SET NULL, expression TEXT NOT NULL,
  expression_type TEXT NOT NULL, timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai', prompt TEXT NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1, next_run_at TEXT, last_run_at TEXT, last_error TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
  id TEXT PRIMARY KEY, actor TEXT, action TEXT NOT NULL, target TEXT, detail_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._local = threading.local()
        self._lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        # WAL and shared-memory sidecars are created lazily by SQLite. Apply
        # the same owner-only mode after opening so credentials never end up
        # in a world-readable ``-wal`` file on a permissive filesystem.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _chmod_database_files(self) -> None:
        for path in (self.path, self.path.with_name(self.path.name + "-wal"), self.path.with_name(self.path.name + "-shm")):
            try:
                if path.exists():
                    os.chmod(path, 0o600)
            except OSError:
                pass

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)
        self._chmod_database_files()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self.connect()
            try:
                connection.execute("BEGIN")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
                self._chmod_database_files()

    def fetchone(self, query: str, args: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(query, args).fetchone()

    def fetchall(self, query: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute(query, args).fetchall())

    def execute(self, query: str, args: tuple[Any, ...] = ()) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(query, args)
            return cursor.rowcount

    @staticmethod
    def json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def loads(value: str | None, default: Any = None) -> Any:
        if not value:
            return {} if default is None else default
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {} if default is None else default

    def setting(self, key: str, default: Any = None) -> Any:
        row = self.fetchone("SELECT value FROM settings WHERE key=?", (key,))
        return self.loads(row["value"], default) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        now = utcnow()
        self.execute(
            "INSERT INTO settings(key,value,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
            (key, self.json(value), now),
        )

    def audit(self, action: str, actor: str = "system", target: str | None = None, detail: Any = None) -> None:
        self.execute(
            "INSERT INTO audit_log(id,actor,action,target,detail_json,created_at) VALUES(?,?,?,?,?,?)",
            (uuid.uuid4().hex, actor, action, target, self.json(detail or {}), utcnow()),
        )
