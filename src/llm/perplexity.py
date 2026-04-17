"""
Perplexity Sonar client — live web search with real citations.

Used by A1 (research collector) to eliminate hallucinated URLs. Sonar returns
a ``citations`` array of real URLs alongside model-generated summary text,
which we format into the numbered-entry shape A1 already validates.

The public ``call()`` signature matches ``src.llm.client.LLMClient.call`` so
A1 can accept either client interchangeably.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.pipeline.state import TokenUsage
from src.utils.errors import AgentTimeoutError, LLMRetryExhaustedError

logger = logging.getLogger(__name__)

PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_MODEL = "sonar-pro"


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout))


class PerplexityClient:
    """Async client for Perplexity Sonar Pro web search."""

    def __init__(self, api_key: str, timeout: float = 90.0):
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY is required")
        self._api_key = api_key
        self._timeout = timeout

    async def call(
        self,
        agent_id: str,
        prompt: str,
        signal: asyncio.Event | None = None,
    ) -> tuple[str, TokenUsage]:
        """
        Search the live web and return (text, token_usage).

        The returned text is the model's summary with a trailing ``Citations:``
        block listing real URLs returned by Perplexity's search — the caller
        (A1) is expected to have asked for numbered entries that reference
        these URLs by index.
        """
        if signal and signal.is_set():
            raise AgentTimeoutError(agent_id, 0)

        @retry(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=2, max=20),
            retry=retry_if_exception(_is_retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _post() -> dict[str, Any]:
            if signal and signal.is_set():
                raise AgentTimeoutError(agent_id, 0)

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": PERPLEXITY_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a research assistant. Use live web search to "
                            "find real, verifiable sources. Every URL must come "
                            "from your search results — never fabricate URLs."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 2000,
            }
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(PERPLEXITY_URL, headers=headers, json=payload)
                resp.raise_for_status()
                return resp.json()

        try:
            data = await _post()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in (429, 500, 502, 503, 504):
                raise
            raise LLMRetryExhaustedError(agent_id, 4) from exc

        choice = data["choices"][0]["message"]
        content = choice.get("content", "") or ""

        # Perplexity returns citations as a top-level field (array of URLs).
        citations: list[str] = data.get("citations") or []

        if citations:
            citation_block = "\n".join(
                f"[{i + 1}] {url}" for i, url in enumerate(citations)
            )
            content = f"{content}\n\nCitations:\n{citation_block}"

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        token_usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        return content, token_usage
