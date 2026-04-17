"""Unit tests for PerplexityClient."""

import httpx
import pytest
import respx

from src.llm.perplexity import PERPLEXITY_URL, PerplexityClient


def test_init_requires_api_key():
    with pytest.raises(ValueError):
        PerplexityClient(api_key="")


@respx.mock
async def test_call_returns_text_with_citations_appended():
    respx.post(PERPLEXITY_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Here are the sources.\n1. Study X"}}
                ],
                "citations": [
                    "https://real-journal.org/study-x",
                    "https://example.com/report",
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 120},
            },
        )
    )
    client = PerplexityClient(api_key="pplx-test")
    text, usage = await client.call("A1", "Research remote work")

    assert "Study X" in text
    assert "https://real-journal.org/study-x" in text
    assert "https://example.com/report" in text
    assert "Citations:" in text
    assert usage.input_tokens == 50
    assert usage.output_tokens == 120
    assert usage.total_tokens == 170


@respx.mock
async def test_call_handles_missing_citations_field():
    respx.post(PERPLEXITY_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Body only"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        )
    )
    client = PerplexityClient(api_key="pplx-test")
    text, _ = await client.call("A1", "prompt")
    assert "Body only" in text
    assert "Citations:" not in text


@respx.mock
async def test_call_sends_bearer_auth_header():
    route = respx.post(PERPLEXITY_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "x"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    )
    client = PerplexityClient(api_key="pplx-secret")
    await client.call("A1", "prompt")
    assert route.calls.last.request.headers["authorization"] == "Bearer pplx-secret"
