import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from src.utils.errors import AgentTimeoutError


async def with_watchdog(
    agent_id: str,
    timeout_ms: int,
    coro: Callable[[asyncio.Event], Coroutine[Any, Any, Any]],
    on_timeout: Callable[[str], None],
) -> Any:
    """
    Race the coroutine against a timeout.
    If timeout fires: calls on_timeout, then raises AgentTimeoutError.
    If coroutine raises: propagates the exception unchanged.
    Always cancels the losing side cleanly.
    """
    timeout_s = timeout_ms / 1000
    cancel_event = asyncio.Event()

    async def _run() -> Any:
        return await coro(cancel_event)

    try:
        result = await asyncio.wait_for(_run(), timeout=timeout_s)
        return result
    except asyncio.TimeoutError:
        cancel_event.set()
        on_timeout(agent_id)
        raise AgentTimeoutError(agent_id, timeout_ms)
