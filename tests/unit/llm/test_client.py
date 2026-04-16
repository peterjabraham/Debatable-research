import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import APIStatusError

from src.llm.client import AGENT_MODELS, AGENT_TEMPERATURES, LLMClient
from src.pipeline.state import TokenUsage
from src.utils.errors import AgentTimeoutError, LLMRetryExhaustedError


def make_mock_response(text: str = "output", input_tokens: int = 10, output_tokens: int = 20):
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return response


def make_api_error(status_code: int) -> APIStatusError:
    response = MagicMock()
    response.status_code = status_code
    response.headers = {}
    return APIStatusError(
        message=f"HTTP {status_code}",
        response=response,
        body={"error": {"message": f"HTTP {status_code}"}},
    )


@pytest.fixture
def client():
    return LLMClient(api_key="test-key")


@pytest.mark.asyncio
async def test_returns_output_and_token_usage_on_success(client):
    mock_resp = make_mock_response("hello world", 5, 15)
    with patch.object(client._client.messages, "create", new=AsyncMock(return_value=mock_resp)):
        text, usage = await client.call("A1", "test prompt")
    assert text == "hello world"
    assert isinstance(usage, TokenUsage)
    assert usage.input_tokens == 5
    assert usage.output_tokens == 15
    assert usage.total_tokens == 20


@pytest.mark.asyncio
async def test_calls_with_correct_model_per_agent(client):
    mock_resp = make_mock_response()
    for agent_id in AGENT_MODELS:
        with patch.object(
            client._client.messages, "create", new=AsyncMock(return_value=mock_resp)
        ) as mock_create:
            await client.call(agent_id, "prompt")
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == AGENT_MODELS[agent_id]


@pytest.mark.asyncio
async def test_calls_with_correct_temperature_per_agent(client):
    mock_resp = make_mock_response()
    for agent_id in AGENT_TEMPERATURES:
        with patch.object(
            client._client.messages, "create", new=AsyncMock(return_value=mock_resp)
        ) as mock_create:
            await client.call(agent_id, "prompt")
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["temperature"] == AGENT_TEMPERATURES[agent_id]


@pytest.mark.asyncio
async def test_retries_on_529(client):
    error = make_api_error(529)
    success = make_mock_response("ok")
    calls = [error, error, success]

    async def side_effect(*args, **kwargs):
        val = calls.pop(0)
        if isinstance(val, Exception):
            raise val
        return val

    with patch.object(client._client.messages, "create", side_effect=side_effect):
        text, _ = await client.call("A1", "prompt")
    assert text == "ok"


@pytest.mark.asyncio
async def test_retries_on_500(client):
    error = make_api_error(500)
    success = make_mock_response("ok")
    calls = [error, success]

    async def side_effect(*args, **kwargs):
        val = calls.pop(0)
        if isinstance(val, Exception):
            raise val
        return val

    with patch.object(client._client.messages, "create", side_effect=side_effect):
        text, _ = await client.call("A1", "prompt")
    assert text == "ok"


@pytest.mark.asyncio
async def test_retries_on_429(client):
    error = make_api_error(429)
    success = make_mock_response("ok")
    calls = [error, success]

    async def side_effect(*args, **kwargs):
        val = calls.pop(0)
        if isinstance(val, Exception):
            raise val
        return val

    with patch.object(client._client.messages, "create", side_effect=side_effect):
        text, _ = await client.call("A1", "prompt")
    assert text == "ok"


@pytest.mark.asyncio
async def test_does_not_retry_on_400(client):
    error = make_api_error(400)
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise error

    with patch.object(client._client.messages, "create", side_effect=side_effect):
        with pytest.raises(APIStatusError) as exc_info:
            await client.call("A1", "prompt")
    assert exc_info.value.status_code == 400
    assert call_count == 1


@pytest.mark.asyncio
async def test_does_not_retry_on_401(client):
    error = make_api_error(401)
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise error

    with patch.object(client._client.messages, "create", side_effect=side_effect):
        with pytest.raises(APIStatusError) as exc_info:
            await client.call("A1", "prompt")
    assert exc_info.value.status_code == 401
    assert call_count == 1


@pytest.mark.asyncio
async def test_raises_retry_exhausted_after_4_529_attempts(client):
    error = make_api_error(529)
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise error

    with patch.object(client._client.messages, "create", side_effect=side_effect):
        with pytest.raises(LLMRetryExhaustedError):
            await client.call("A1", "prompt")
    assert call_count == 4


@pytest.mark.asyncio
async def test_respects_cancel_event(client):
    """If cancel_event is already set, raises AgentTimeoutError immediately."""
    cancel_event = asyncio.Event()
    cancel_event.set()

    with pytest.raises(AgentTimeoutError):
        await client.call("A1", "prompt", signal=cancel_event)
