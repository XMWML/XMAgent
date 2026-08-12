from __future__ import annotations

import asyncio

from xmagents.agents.runtime import AgentSettings, OpenAIProvider


class _Response:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def aread(self):
        return b""

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"hel"}}]}'
        yield 'data: {"choices":[{"delta":{"content":"lo"}}]}'
        yield "data: [DONE]"


class _Client:
    def stream(self, *_args, **_kwargs):
        return _Response()


def test_openai_sse_text_deltas_are_marked_streaming():
    async def run():
        provider = OpenAIProvider(
            AgentSettings(api_url="https://example.invalid/v1", api_key="secret", model="test"),
            client=_Client(),
        )
        events = [event async for event in provider.stream("hello", [], session_id="persisted-session")]
        assert [event.content for event in events] == ["hel", "lo"]
        assert all(event.metadata == {"provider": "openai", "stream": True} for event in events)

    asyncio.run(run())
