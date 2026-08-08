"""
LLM generation layer.

If OPENAI_API_KEY is set, we call OpenAI's chat completion API with a
citation-enforcing system prompt. If no key is configured (e.g. a fresh
local clone, or CI), we fall back to a deterministic *extractive* answer
built directly from the retrieved chunks. This keeps the whole app runnable
end-to-end with zero API keys, which matters for a demo/eval pipeline.
"""

import logging
from typing import List

from app.config import OPENAI_API_KEY, OPENAI_MODEL, LLM_TEMPERATURE, MODEL_PRICING, DEFAULT_PRICING
from app import metrics

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are "Ask My Docs", an assistant that helps patients understand \
their own medical documents (lab reports, discharge summaries, radiology reports, \
medication sheets).

Rules you MUST follow:
1. Answer ONLY using the numbered CONTEXT excerpts provided below. Do not use outside \
medical knowledge to fill gaps.
2. Every factual sentence in your answer must end with a citation marker like [1] or \
[2] that refers to the numbered context excerpt it came from. If a sentence combines \
information from two excerpts, cite both, like [1][2].
3. If the answer is not contained in the CONTEXT, say clearly that the documents \
provided don't contain that information, and suggest the patient ask their care team. \
Do not guess.
4. Never provide a diagnosis, a prescription change, or a definitive medical \
recommendation. Explain what the documents say in plain language, and encourage the \
patient to discuss results and next steps with their clinician.
5. Keep answers concise (roughly 3-6 sentences) and use plain, non-alarming language.
"""

USER_TEMPLATE = """CONTEXT EXCERPTS:
{context}

PATIENT QUESTION: {question}

Write a grounded, cited answer following all system rules."""


def _format_context(chunks) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] (from \"{chunk.doc_title}\" - {chunk.section})\n{chunk.text}")
    return "\n\n".join(lines)


def _extractive_fallback_answer(question: str, chunks: List) -> str:
    """
    Deterministic, fully-grounded answer used when no LLM API key is
    configured. It stitches together the most relevant retrieved sentences
    with citation markers, so the citation-enforcement contract still holds.
    """
    if not chunks:
        return (
            "I couldn't find anything in your documents that answers this question. "
            "Please check with your care team directly."
        )

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        snippet = chunk.text.strip().split("\n")[0]
        if len(snippet) > 260:
            snippet = snippet[:257].rstrip() + "..."
        parts.append(f"From your {chunk.doc_type.lower()} ({chunk.section}): {snippet} [{i}]")

    intro = (
        "Here's what your documents say related to your question "
        "(no AI language model is configured, so this is a direct extract "
        "of the most relevant passages):\n\n"
    )
    return intro + "\n".join(parts)


def generate_answer(question: str, chunks: List) -> str:
    if not chunks:
        return (
            "I couldn't find anything in your documents that answers this question. "
            "Please check with your care team directly."
        )

    if not OPENAI_API_KEY:
        logger.info("OPENAI_API_KEY not set -> using extractive fallback answer")
        return _extractive_fallback_answer(question, chunks)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        context = _format_context(chunks)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(context=context, question=question),
                },
            ],
        )
        _record_usage_metrics(response)
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never 500 the user
        logger.exception("LLM call failed, using extractive fallback: %s", exc)
        return _extractive_fallback_answer(question, chunks)


def _record_usage_metrics(response) -> None:
    """Pull token counts off the OpenAI response and record tokens + estimated cost."""
    usage = getattr(response, "usage", None)
    if not usage:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    pricing = MODEL_PRICING.get(OPENAI_MODEL, DEFAULT_PRICING)
    cost = prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]
    metrics.record_llm_usage(OPENAI_MODEL, prompt_tokens, completion_tokens, cost)
