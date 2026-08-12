"""Per-agent durable memory and non-destructive ``CLAUDE.md`` synchronisation.

Memory is intentionally local and conservative.  It is not a second model
call: only an explicit request to remember something, or a small set of clear
self-identification statements, can become durable memory.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from pathlib import Path
from typing import Any

from .database import utcnow


MEMORY_START = "<!-- XMAGENT_MEMORY_START -->"
MEMORY_END = "<!-- XMAGENT_MEMORY_END -->"
_MEMORY_BLOCK = re.compile(
    rf"[ \t]*{re.escape(MEMORY_START)}.*?{re.escape(MEMORY_END)}[ \t]*(?:\n|$)", re.S
)


def _normalise_content(content: str) -> str:
    """Normalise just enough to identify repeated, literally equal facts."""

    value = unicodedata.normalize("NFKC", str(content)).strip()
    return re.sub(r"\s+", " ", value)


def _memory_key(content: str) -> str:
    """Return a bounded, punctuation-insensitive key for local deduplication."""

    value = _normalise_content(content).casefold()
    return value.rstrip(".。!！?？;；:：")


def _extract_facts(user_text: str) -> list[str]:
    """Extract only clearly durable user facts without interpreting chat text.

    The patterns deliberately favour false negatives.  A casual remark should
    remain in conversation history rather than unexpectedly become memory.
    """

    text = str(user_text or "")
    facts: list[str] = []

    # Explicit requests may appear in a natural sentence, but the fact itself
    # ends at a sentence boundary/newline.  This avoids storing trailing chat
    # commentary such as "谢谢" as part of the remembered value.
    explicit = re.compile(
        r"(?:请\s*)?记住\s*[:：]\s*([^\n。！？!?]{2,300})|"
        r"\bremember\s*:\s*([^\n.!?]{2,300})",
        re.I,
    )
    for match in explicit.finditer(text):
        fact = _normalise_content(match.group(1) or match.group(2) or "")
        if fact:
            facts.append(fact)

    # Stable declarations are allowed only as complete, short sentences.  Do
    # not infer preferences/facts from an assistant answer or general prose.
    stable = re.compile(
        r"(?:^|[\n。！？!?])\s*("
        r"我叫[^\n。！？!?]{1,120}|"
        r"我的(?:时区|偏好)是[^\n。！？!?]{1,160}|"
        r"我住在[^\n。！？!?]{1,160}|"
        r"my name is[^\n.!?]{1,120}|"
        r"my timezone is[^\n.!?]{1,120}|"
        r"i (?:prefer|live in)[^\n.!?]{1,160}"
        r")(?=$|[\n。！？!?])",
        re.I,
    )
    for match in stable.finditer(text):
        fact = _normalise_content(match.group(1))
        if fact:
            facts.append(fact)

    # Preserve input order while avoiding duplicate inserts from overlapping
    # explicit and stable patterns in the same inbound message.
    seen: set[str] = set()
    return [fact for fact in facts if (key := _memory_key(fact)) and not (key in seen or seen.add(key))]


class MemoryStore:
    def __init__(self, service: Any):
        self.service = service

    async def context(self, agent_id: str, query: str = "", limit: int = 8) -> str:
        rows = self.service.db.fetchall(
            "SELECT content FROM memory_entries WHERE agent_id=? AND enabled=1 ORDER BY updated_at DESC LIMIT ?",
            (agent_id, max(1, min(limit, 30))),
        )
        if not rows:
            return ""
        return "\n".join(f"- {row['content']}" for row in rows)

    async def observe(self, agent_id: str, user_text: str, assistant_text: str) -> None:
        # ``assistant_text`` is deliberately not examined.  The assistant may
        # speculate or quote user content; only the user's explicit input can
        # affect durable memory.
        del assistant_text
        for content in _extract_facts(user_text):
            self._upsert(agent_id, content, kind="explicit")
        self.compact(agent_id)
        self.sync_claude_md(agent_id)

    def add(self, agent_id: str, content: str, *, kind: str = "fact", source_message_id: str | None = None) -> str:
        memory_id = self._upsert(agent_id, content, kind=kind, source_message_id=source_message_id)
        self.compact(agent_id)
        self.sync_claude_md(agent_id)
        return memory_id

    def _upsert(self, agent_id: str, content: str, *, kind: str, source_message_id: str | None = None) -> str:
        """Insert a memory or refresh its exact canonical equivalent."""

        cleaned = _normalise_content(content)
        if not cleaned:
            raise ValueError("记忆内容不能为空")
        key = _memory_key(cleaned)
        rows = self.service.db.fetchall(
            "SELECT id,content FROM memory_entries WHERE agent_id=? ORDER BY updated_at DESC, created_at DESC",
            (agent_id,),
        )
        now = utcnow()
        for row in rows:
            if _memory_key(str(row["content"])) == key:
                self.service.db.execute(
                    "UPDATE memory_entries SET content=?,kind=?,source_message_id=COALESCE(?,source_message_id),updated_at=? WHERE id=?",
                    (cleaned, kind, source_message_id, now, row["id"]),
                )
                return str(row["id"])
        memory_id = uuid.uuid4().hex
        self.service.db.execute(
            "INSERT INTO memory_entries(id,agent_id,kind,content,source_message_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
            (memory_id, agent_id, kind, cleaned, source_message_id, now, now),
        )
        return memory_id

    def compact(self, agent_id: str) -> int:
        """Collapse legacy duplicate facts for one agent and retain the newest.

        This runs after each memory mutation.  The operation is intentionally
        per-agent, so equal facts in two users' workspaces never affect one
        another.  If any duplicate remains enabled, the surviving entry stays
        enabled instead of unexpectedly disappearing from active context.
        """

        rows = self.service.db.fetchall(
            "SELECT * FROM memory_entries WHERE agent_id=? ORDER BY updated_at DESC, created_at DESC, id DESC",
            (agent_id,),
        )
        groups: dict[str, list[Any]] = {}
        for row in rows:
            key = _memory_key(str(row["content"]))
            if key:
                groups.setdefault(key, []).append(row)
        removed = 0
        for duplicates in groups.values():
            if len(duplicates) < 2:
                continue
            keeper = duplicates[0]
            enabled = int(any(bool(item["enabled"]) for item in duplicates))
            if int(keeper["enabled"]) != enabled:
                self.service.db.execute(
                    "UPDATE memory_entries SET enabled=?,updated_at=? WHERE id=?",
                    (enabled, utcnow(), keeper["id"]),
                )
            duplicate_ids = tuple(str(item["id"]) for item in duplicates[1:])
            placeholders = ",".join("?" for _ in duplicate_ids)
            self.service.db.execute(
                f"DELETE FROM memory_entries WHERE id IN ({placeholders})", duplicate_ids,
            )
            removed += len(duplicate_ids)
        return removed

    def list(self, agent_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return [dict(row) for row in self.service.db.fetchall(
            "SELECT * FROM memory_entries WHERE agent_id=? ORDER BY updated_at DESC LIMIT ?", (agent_id, limit)
        )]

    def delete(self, memory_id: str) -> None:
        row = self.service.db.fetchone("SELECT agent_id FROM memory_entries WHERE id=?", (memory_id,))
        self.service.db.execute("DELETE FROM memory_entries WHERE id=?", (memory_id,))
        if row:
            self.sync_claude_md(str(row["agent_id"]))

    def update(self, memory_id: str, content: str, *, enabled: bool | None = None, kind: str | None = None) -> dict[str, Any]:
        row = self.service.db.fetchone("SELECT * FROM memory_entries WHERE id=?", (memory_id,))
        if not row:
            raise KeyError(memory_id)
        cleaned = _normalise_content(content)
        if not cleaned:
            raise ValueError("记忆内容不能为空")
        values: dict[str, Any] = {"content": cleaned, "updated_at": utcnow()}
        if enabled is not None:
            values["enabled"] = int(enabled)
        if kind is not None:
            values["kind"] = str(kind)
        assignments = ",".join(f"{key}=?" for key in values)
        self.service.db.execute(f"UPDATE memory_entries SET {assignments} WHERE id=?", (*values.values(), memory_id))
        self.compact(str(row["agent_id"]))
        self.sync_claude_md(str(row["agent_id"]))
        updated = self.service.db.fetchone("SELECT * FROM memory_entries WHERE id=?", (memory_id,))
        if updated is None:
            # A just-updated duplicate can only be removed when another row
            # has a newer timestamp. Return the retained equivalent instead.
            key = _memory_key(cleaned)
            for candidate in self.service.db.fetchall(
                "SELECT * FROM memory_entries WHERE agent_id=? ORDER BY updated_at DESC", (row["agent_id"],)
            ):
                if _memory_key(str(candidate["content"])) == key:
                    updated = candidate
                    break
        return dict(updated) if updated else {}

    def sync_claude_md(self, agent_id: str) -> None:
        row = self.service.db.fetchone("SELECT workspace FROM agents WHERE id=?", (agent_id,))
        if not row:
            return
        entries = self.service.db.fetchall(
            "SELECT content FROM memory_entries WHERE agent_id=? AND enabled=1 ORDER BY updated_at DESC", (agent_id,)
        )
        path = Path(row["workspace"]) / "CLAUDE.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
        except OSError:
            return
        block = ""
        if entries:
            block = (
                f"{MEMORY_START}\n"
                "## XMAgent persistent memory\n\n"
                + "\n".join(f"- {item['content']}" for item in entries)
                + f"\n{MEMORY_END}\n"
            )
        # Never regenerate the whole file.  The marked section is owned by
        # XMAgent; all unmarked content can be maintained by the Agent/user.
        if MEMORY_START in existing and MEMORY_END in existing:
            text = _MEMORY_BLOCK.sub(block, existing, count=1)
        elif block:
            separator = "" if not existing or existing.endswith("\n") else "\n"
            text = existing + separator + ("\n" if existing else "") + block
        else:
            text = existing
        if text != existing or not path.exists():
            path.write_text(text, encoding="utf-8")


class KnowledgeContext:
    def __init__(self, service: Any):
        self.service = service

    async def context(self, agent_id: str, query: str, limit: int = 5) -> str:
        row = self.service.db.fetchone("SELECT knowledge_base_id FROM agents WHERE id=?", (agent_id,))
        if not row or not row["knowledge_base_id"] or not query.strip():
            return ""
        from .knowledge import KnowledgeService

        results = KnowledgeService(self.service).search(row["knowledge_base_id"], query, limit)
        return "\n\n".join(f"[{item.get('title') or item.get('id')}]\n{item.get('content', '')}" for item in results)
