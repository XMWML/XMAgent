from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from fastapi.testclient import TestClient

from xmagents.config import AppPaths
from xmagents.main import build_app
from xmagents.scheduler import Scheduler, next_run
from xmagents.service import AppService


def test_cron_is_strictly_five_fields_and_intervals_are_positive() -> None:
    now = datetime(2026, 8, 12, 0, 0, tzinfo=UTC)
    assert next_run("0 9 * * *", "cron", now=now) > now
    for expression in ("0 0 9 * * *", "*/5 * * * * *", "0 9 * *"):
        with pytest.raises(ValueError, match="五字段|无效"):
            next_run(expression, "cron", now=now)
    for expression in ("0", "-1", "nan", "inf", "not-a-number"):
        with pytest.raises(ValueError, match="正数"):
            next_run(expression, "every", now=now)


def test_create_validates_agent_and_peer_binding() -> None:
    with TemporaryDirectory() as directory:
        service = AppService(AppPaths.from_root(Path.cwd(), Path(directory) / "data"))
        with pytest.raises(ValueError, match="Agent 不存在"):
            service.scheduler.create("missing", "task", "60", "every")

        account = service.create_channel("telegram", "test", token="token")
        peer = service.upsert_peer(account["id"], "remote", approved=True)
        agent = service.create_agent("bound")
        with pytest.raises(ValueError, match="未绑定"):
            service.scheduler.create(agent["id"], "task", "60", "every", peer_id=peer["id"])

        service.approve_peer(peer["id"], agent_id=agent["id"])
        schedule = service.scheduler.create(agent["id"], "task", "60", "every", peer_id=peer["id"])
        assert schedule["agent_id"] == agent["id"]
        assert schedule["peer_id"] == peer["id"]


def test_web_schedule_rejects_unknown_agent_with_bad_request() -> None:
    with TemporaryDirectory() as directory:
        service = AppService(AppPaths.from_root(Path.cwd(), Path(directory) / "data"))
        service.initialize_admin("1234567890")
        app = build_app(service)
        with TestClient(app) as client:
            client.cookies.set("xmagents_session", service.create_session())
            client.cookies.set("xmagents_csrf", "test-csrf")
            response = client.post(
                "/api/schedules",
                json={"agent_id": "missing", "prompt": "run", "expression_type": "every", "expression": "60"},
                headers={"X-XMAgent-CSRF": "test-csrf"},
            )
            assert response.status_code == 400
            assert "Agent 不存在" in response.json()["detail"]


async def _run_failed_schedule(expression_type: str) -> dict:
    with TemporaryDirectory() as directory:
        service = AppService(AppPaths.from_root(Path.cwd(), Path(directory) / "data"))
        agent = service.create_agent("failing")
        schedule = service.scheduler.create(
            agent["id"], "run", "2020-01-01T00:00:00Z" if expression_type == "at" else "3600", expression_type,
        )
        # Make the row immediately due without changing its persisted shape.
        service.db.execute("UPDATE schedules SET next_run_at=? WHERE id=?", (datetime.now(UTC).isoformat(), schedule["id"]))
        attempted = asyncio.Event()

        async def fail(_: dict) -> str:
            attempted.set()
            raise RuntimeError("provider unavailable")

        runner = Scheduler(service, fail)
        await runner.start()
        await asyncio.wait_for(attempted.wait(), timeout=0.75)
        # Give the worker one event-loop turn to persist the failure state.
        await asyncio.sleep(0)
        result = dict(service.db.fetchone("SELECT * FROM schedules WHERE id=?", (schedule["id"],)))
        await runner.stop()
        return result


def test_failed_recurring_schedule_is_rescheduled() -> None:
    result = asyncio.run(_run_failed_schedule("every"))
    assert result["enabled"] == 1
    assert result["next_run_at"]
    assert result["last_error"] == "provider unavailable"


def test_failed_one_shot_schedule_is_disabled() -> None:
    result = asyncio.run(_run_failed_schedule("at"))
    assert result["enabled"] == 0
    assert result["next_run_at"] is None
    assert result["last_error"] == "provider unavailable"
