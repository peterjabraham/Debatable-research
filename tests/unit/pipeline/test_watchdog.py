import asyncio

import pytest

from src.pipeline.watchdog import with_watchdog
from src.utils.errors import AgentTimeoutError


async def fast_coro(cancel_event: asyncio.Event) -> str:
    await asyncio.sleep(0.01)
    return "done"


async def slow_coro(cancel_event: asyncio.Event) -> str:
    await asyncio.sleep(10)
    return "done"


async def near_miss_coro(cancel_event: asyncio.Event) -> str:
    """Completes just before the timeout."""
    await asyncio.sleep(0.05)
    return "near miss"


async def raising_coro(cancel_event: asyncio.Event) -> str:
    raise ValueError("inner error")


def test_resolves_correctly_within_timeout():
    on_timeout = lambda aid: None  # noqa: E731

    async def run():
        return await with_watchdog("A1", 5000, fast_coro, on_timeout)

    result = asyncio.run(run())
    assert result == "done"


def test_calls_on_timeout_when_exceeded():
    called = []

    def on_timeout(aid):
        called.append(aid)

    async def run():
        with pytest.raises(AgentTimeoutError):
            await with_watchdog("A1", 50, slow_coro, on_timeout)

    asyncio.run(run())
    assert called == ["A1"]


def test_raises_agent_timeout_error_when_exceeded():
    async def run():
        with pytest.raises(AgentTimeoutError) as exc_info:
            await with_watchdog("A2", 50, slow_coro, lambda _: None)
        assert exc_info.value.agent_id == "A2"

    asyncio.run(run())


def test_sets_cancel_event_when_timeout_fires():
    captured_event: list[asyncio.Event] = []

    async def capture_event_coro(cancel_event: asyncio.Event) -> str:
        captured_event.append(cancel_event)
        await asyncio.sleep(10)
        return "done"

    async def run():
        with pytest.raises(AgentTimeoutError):
            await with_watchdog("A1", 50, capture_event_coro, lambda _: None)
        assert len(captured_event) == 1
        assert captured_event[0].is_set()

    asyncio.run(run())


def test_does_not_call_on_timeout_when_completes_in_time():
    called = []

    async def run():
        await with_watchdog("A1", 5000, fast_coro, lambda aid: called.append(aid))

    asyncio.run(run())
    assert called == []


def test_propagates_coroutine_exceptions_unchanged():
    async def run():
        with pytest.raises(ValueError, match="inner error"):
            await with_watchdog("A1", 5000, raising_coro, lambda _: None)

    asyncio.run(run())


def test_completes_just_before_timeout():
    """No race condition when coro completes very close to the deadline."""
    async def run():
        result = await with_watchdog("A1", 200, near_miss_coro, lambda _: None)
        assert result == "near miss"

    asyncio.run(run())
