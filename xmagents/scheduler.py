"""SQLite-backed lightweight scheduler.

The scheduler is intentionally dependency-light. It supports one-shot ISO
timestamps, interval seconds, and five-field cron expressions. A task is
executed through the same agent callback used by inbound messages.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

try:
    from croniter import croniter
except ImportError:  # pragma: no cover
    croniter = None

from .database import utcnow


def next_run(expression: str, expression_type: str, timezone: str = "Asia/Shanghai", now: datetime | None = None) -> datetime:
    try:
        tz = ZoneInfo(timezone)
    except (KeyError, ValueError) as error:
        raise ValueError(f"无效的时区: {timezone}") from error
    current = (now or datetime.now(UTC)).astimezone(tz)
    if expression_type == "at":
        try:
            value = datetime.fromisoformat(expression.replace("Z", "+00:00"))
        except (TypeError, ValueError) as error:
            raise ValueError("无效的 at 时间，必须是 ISO 8601 时间") from error
        return (value if value.tzinfo else value.replace(tzinfo=tz)).astimezone(UTC)
    if expression_type == "every":
        try:
            seconds = float(expression)
        except (TypeError, ValueError) as error:
            raise ValueError("every 必须是正数秒数") from error
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("every 必须是正数秒数")
        return (current + timedelta(seconds=seconds)).astimezone(UTC)
    if expression_type == "cron" and croniter is not None:
        # croniter also accepts six/seven-field expressions for seconds and
        # years. XMAgent deliberately exposes the portable five-field form.
        if len(expression.split()) != 5:
            raise ValueError("cron 必须是五字段表达式")
        try:
            if not croniter.is_valid(expression):
                raise ValueError("无效的 Cron 表达式")
            return croniter(expression, current).get_next(datetime).astimezone(UTC)
        except ValueError:
            raise
        except Exception as error:
            raise ValueError("无效的 Cron 表达式") from error
    if expression_type == "cron" and croniter is None:
        raise ValueError("当前环境未安装 Cron 支持")
    raise ValueError("无效的定时任务表达式")


class Scheduler:
    def __init__(self, service: Any, execute: Callable[[dict[str, Any]], Awaitable[str]] | None = None):
        self.service = service
        self.execute = execute
        self._task: asyncio.Task[Any] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="xmagents-scheduler")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def create(self, agent_id: str, prompt: str, expression: str, expression_type: str,
               *, peer_id: str | None = None, timezone: str = "Asia/Shanghai") -> dict[str, Any]:
        import uuid

        # Resolve relationships before calculating/inserting the schedule so
        # callers receive a normal validation error instead of a SQLite
        # foreign-key 500. A supplied peer must be approved and actively
        # bound to this Agent; cross-Agent delivery is never implicit.
        agent = self.service.db.fetchone("SELECT id FROM agents WHERE id=?", (agent_id,))
        if not agent:
            raise ValueError("Agent 不存在")
        if peer_id is not None:
            requested_peer = str(peer_id).strip()
            if not requested_peer:
                raise ValueError("渠道用户 ID 不能为空")
            peer = self.service.db.fetchone(
                "SELECT p.id,p.approved FROM remote_peers p "
                "JOIN agent_bindings b ON b.peer_id=p.id AND b.agent_id=? AND b.active=1 "
                "WHERE p.id=?",
                (agent_id, requested_peer),
            )
            if not peer:
                raise ValueError("渠道用户未绑定到此 Agent")
            if not bool(peer["approved"]):
                raise ValueError("渠道用户尚未批准")
            peer_id = requested_peer

        first = next_run(expression, expression_type, timezone)
        now = utcnow()
        task_id = uuid.uuid4().hex
        self.service.db.execute(
            "INSERT INTO schedules(id,agent_id,peer_id,expression,expression_type,timezone,prompt,next_run_at,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (task_id, agent_id, peer_id, expression, expression_type, timezone, prompt, first.isoformat(), now, now),
        )
        return dict(self.service.db.fetchone("SELECT * FROM schedules WHERE id=?", (task_id,)))

    def cancel(self, task_id: str) -> None:
        self.service.db.execute("UPDATE schedules SET enabled=0,updated_at=? WHERE id=?", (utcnow(), task_id))

    def list(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        if agent_id:
            rows = self.service.db.fetchall("SELECT * FROM schedules WHERE agent_id=? ORDER BY next_run_at", (agent_id,))
        else:
            rows = self.service.db.fetchall("SELECT * FROM schedules ORDER BY next_run_at")
        return [dict(row) for row in rows]

    async def _run(self) -> None:
        while not self._stop.is_set():
            now = datetime.now(UTC)
            rows = self.service.db.fetchall(
                "SELECT * FROM schedules WHERE enabled=1 AND next_run_at IS NOT NULL AND next_run_at<=? ORDER BY next_run_at LIMIT 20",
                (now.isoformat(),),
            )
            for row in rows:
                item = dict(row)
                try:
                    if self.execute:
                        await self.execute(item)
                    if item["expression_type"] == "at":
                        self.service.db.execute("UPDATE schedules SET enabled=0,last_run_at=?,updated_at=? WHERE id=?", (utcnow(), utcnow(), item["id"]))
                    else:
                        following = next_run(item["expression"], item["expression_type"], item["timezone"], datetime.now(UTC))
                        self.service.db.execute("UPDATE schedules SET next_run_at=?,last_run_at=?,last_error=NULL,updated_at=? WHERE id=?", (following.isoformat(), utcnow(), utcnow(), item["id"]))
                except Exception as error:
                    # A failed run must leave the due window. Otherwise the
                    # worker observes the same row every second forever. One-
                    # shot jobs are terminal; recurring jobs retry at their
                    # next scheduled occurrence. If the stored expression is
                    # no longer valid, disable it rather than hot-looping.
                    failed_at = utcnow()
                    error_text = str(error)
                    if item["expression_type"] == "at":
                        self.service.db.execute(
                            "UPDATE schedules SET enabled=0,next_run_at=NULL,last_error=?,last_run_at=?,updated_at=? WHERE id=?",
                            (error_text, failed_at, failed_at, item["id"]),
                        )
                        continue
                    try:
                        following = next_run(item["expression"], item["expression_type"], item["timezone"], datetime.now(UTC))
                    except Exception as next_error:
                        error_text = f"{error_text}; 下次运行时间无效: {next_error}"
                        self.service.db.execute(
                            "UPDATE schedules SET enabled=0,next_run_at=NULL,last_error=?,last_run_at=?,updated_at=? WHERE id=?",
                            (error_text, failed_at, failed_at, item["id"]),
                        )
                    else:
                        self.service.db.execute(
                            "UPDATE schedules SET next_run_at=?,last_error=?,last_run_at=?,updated_at=? WHERE id=?",
                            (following.isoformat(), error_text, failed_at, failed_at, item["id"]),
                        )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
