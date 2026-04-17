"""
Topic refinement via GPT-4.1.

Given a rough user query (e.g. "remote work"), returns a structured pillar topic
and 5–7 specific cluster topics that are debatable and suitable for the pipeline.
"""
from __future__ import annotations

import json
import os

from openai import AsyncOpenAI
from pydantic import BaseModel


class ClusterTopic(BaseModel):
    topic: str
    description: str


class TopicOptions(BaseModel):
    pillar: ClusterTopic
    clusters: list[ClusterTopic]


_SYSTEM_PROMPT = """\
You are a content strategist who specialises in debatable research topics for long-form blog posts.

Given a user's rough query, return a JSON object with:
- "pillar": the broad umbrella topic (suitable as a content pillar / category page)
- "clusters": 5 to 7 specific, debatable subtopics (suitable as individual blog posts)

Each cluster must be:
- A clear yes/no or either/or question or proposition
- Genuinely contested — there must be credible evidence on both sides
- Specific enough that a research pipeline can find real sources

Return ONLY valid JSON matching this schema:
{
  "pillar": { "topic": "...", "description": "..." },
  "clusters": [
    { "topic": "...", "description": "..." },
    ...
  ]
}
"""


async def refine(query: str) -> TopicOptions:
    """
    Call GPT-4.1 to expand a rough user query into a pillar topic and cluster topics.
    Returns a TopicOptions instance.
    Raises ValueError if the API response cannot be parsed.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    client = AsyncOpenAI(api_key=api_key)

    response = await client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f'Query: "{query}"'},
        ],
        temperature=0.7,
        max_tokens=1000,
    )

    raw = response.choices[0].message.content or "{}"

    try:
        data = json.loads(raw)
        return TopicOptions.model_validate(data)
    except Exception as exc:
        raise ValueError(f"Failed to parse GPT-4.1 response: {exc}\nRaw: {raw[:300]}") from exc
