from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from xmagents.agents.runtime import AgentRuntime, AgentSettings, Provider, RuntimeEvent
from xmagents.config import AppPaths
from xmagents.memory import MEMORY_END, MEMORY_START, MemoryStore
from xmagents.service import AppService


def test_memory_observation_is_conservative_isolated_and_deduplicated() -> None:
    async def run() -> None:
        with TemporaryDirectory() as directory:
            service = AppService(AppPaths.from_root(Path.cwd(), Path(directory) / "data"))
            first = service.create_agent("first")
            second = service.create_agent("second")
            memory = MemoryStore(service)

            await memory.observe(first["id"], "请记住：我的项目是 XMAgent。", "好的")
            await memory.observe(first["id"], "记住：  我的项目是 XMAgent!", "好的")
            await memory.observe(second["id"], "我叫小明。", "好的")
            await memory.observe(first["id"], "你应该记住这个普通聊天", "记住：不应保存")

            first_entries = memory.list(first["id"])
            second_entries = memory.list(second["id"])
            assert [entry["content"] for entry in first_entries] == ["我的项目是 XMAgent"]
            assert [entry["content"] for entry in second_entries] == ["我叫小明"]

    asyncio.run(run())


def test_memory_compact_removes_legacy_duplicates_per_agent() -> None:
    with TemporaryDirectory() as directory:
        service = AppService(AppPaths.from_root(Path.cwd(), Path(directory) / "data"))
        first = service.create_agent("first")
        second = service.create_agent("second")
        now = "2026-08-12T00:00:00+00:00"
        for agent_id, content in (
            (first["id"], "Project is XMAgent."),
            (first["id"], " project   is xmagENT "),
            (first["id"], "PROJECT IS XMAGENT!"),
            (second["id"], "Project is XMAgent."),
        ):
            service.db.execute(
                "INSERT INTO memory_entries(id,agent_id,kind,content,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, agent_id, "fact", content, 1, now, now),
            )

        memory = MemoryStore(service)
        assert memory.compact(first["id"]) == 2
        assert len(memory.list(first["id"])) == 1
        assert len(memory.list(second["id"])) == 1


def test_memory_sync_preserves_agent_managed_claude_md_content() -> None:
    with TemporaryDirectory() as directory:
        service = AppService(AppPaths.from_root(Path.cwd(), Path(directory) / "data"))
        agent = service.create_agent("notes")
        path = Path(agent["workspace"]) / "CLAUDE.md"
        path.write_text("# Manual rules\n\n- Keep this line.\n", encoding="utf-8")
        memory = MemoryStore(service)

        first_id = memory.add(agent["id"], "首选语言是中文")
        text = path.read_text(encoding="utf-8")
        assert "# Manual rules" in text
        assert "- Keep this line." in text
        assert MEMORY_START in text and MEMORY_END in text
        assert "首选语言是中文" in text

        path.write_text(text + "\n## Agent note\nDo not replace this.\n", encoding="utf-8")
        memory.add(agent["id"], "时区是 Asia/Shanghai")
        text = path.read_text(encoding="utf-8")
        assert "## Agent note\nDo not replace this." in text
        assert "首选语言是中文" in text
        assert "时区是 Asia/Shanghai" in text

        memory.delete(first_id)
        text = path.read_text(encoding="utf-8")
        assert "# Manual rules" in text
        assert "## Agent note\nDo not replace this." in text


def test_memory_observation_runs_after_reply_and_failure_does_not_break_turn() -> None:
    class ImmediateProvider(Provider):
        async def stream(self, prompt, history, *, session_id="default"):
            del prompt, history, session_id
            yield RuntimeEvent("text", "reply")

    class SlowMemory:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def context(self, agent_id: str, query: str) -> str:
            del agent_id, query
            return ""

        async def observe(self, agent_id: str, user_text: str, assistant_text: str) -> None:
            del agent_id, user_text, assistant_text
            self.started.set()
            await self.release.wait()

    class FailingMemory:
        async def context(self, agent_id: str, query: str) -> str:
            del agent_id, query
            return ""

        async def observe(self, agent_id: str, user_text: str, assistant_text: str) -> None:
            del agent_id, user_text, assistant_text
            raise RuntimeError("memory unavailable")

    async def run() -> None:
        settings = AgentSettings(agent_id="background-memory")
        slow = SlowMemory()
        runtime = AgentRuntime(settings, provider=ImmediateProvider(settings), memory=slow)
        events = await asyncio.wait_for(runtime.submit("记住：不应等待"), timeout=0.1)
        assert events[0].content == "reply"
        await asyncio.wait_for(slow.started.wait(), timeout=0.1)
        assert runtime._memory_tasks
        slow.release.set()
        await asyncio.sleep(0)
        await runtime.close()

        failing = AgentRuntime(settings, provider=ImmediateProvider(settings), memory=FailingMemory())
        events = await asyncio.wait_for(failing.submit("记住：失败也不影响回复"), timeout=0.1)
        assert events[0].content == "reply"
        await asyncio.sleep(0)
        await failing.close()

    asyncio.run(run())
