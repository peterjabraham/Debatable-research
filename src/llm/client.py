import asyncio
import logging

from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic
from anthropic.types import Message
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

AGENT_MODELS: dict[str, str] = {
    "A1": "claude-sonnet-4-6",
    "A2": "claude-sonnet-4-6",
    "A3": "claude-opus-4-5",
    "A35": "claude-sonnet-4-6",
    "A4": "claude-opus-4-5",
    "A5": "claude-opus-4-5",
    "A6": "claude-sonnet-4-6",
}

AGENT_TEMPERATURES: dict[str, float] = {
    "A1": 0.2, "A2": 0.1, "A3": 0.3,
    "A35": 0.5,
    "A4": 0.4, "A5": 0.2, "A6": 0.7,
}

MAX_TOKENS_PER_AGENT: dict[str, int] = {
    "A1": 2000, "A2": 3000, "A3": 2000,
    "A35": 2000,
    "A4": 3000, "A5": 1500, "A6": 4000,
}


def _is_retryable(exc: BaseException) -> bool:
    """Only retry on server errors and connection issues."""
    if isinstance(exc, APIStatusError):
        return exc.status_code in (429, 500, 529)
    return isinstance(exc, APIConnectionError)


class LLMClient:
    def __init__(self, api_key: str):
        self._client = AsyncAnthropic(api_key=api_key)

    async def call(
        self,
        agent_id: str,
        prompt: str,
        signal: asyncio.Event | None = None,
    ) -> tuple[str, TokenUsage]:
        """
        Call the LLM for a given agent. Returns (output_text, token_usage).
        Raises LLMRetryExhaustedError after 4 failed attempts.
        Respects cancellation via signal.
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
        async def _call_with_retry() -> Message:
            if signal and signal.is_set():
                raise AgentTimeoutError(agent_id, 0)

            response = await self._client.messages.create(
                model=AGENT_MODELS[agent_id],
                max_tokens=MAX_TOKENS_PER_AGENT[agent_id],
                temperature=AGENT_TEMPERATURES[agent_id],
                messages=[{"role": "user", "content": prompt}],
            )
            return response

        try:
            response = await _call_with_retry()
        except APIStatusError as e:
            if e.status_code not in (429, 500, 529):
                raise
            raise LLMRetryExhaustedError(agent_id, 4) from e

        # Extract text from the first content block
        first_block = response.content[0]
        text = first_block.text if hasattr(first_block, "text") else ""  # type: ignore[union-attr]

        usage = response.usage
        token_usage = TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )
        return text, token_usage
