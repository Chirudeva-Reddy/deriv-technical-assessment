"""Turn a question plus retrieved chunks into an Answer. generate() is the only place the generator is chosen."""

import logging
import os
import re
import time
from dataclasses import dataclass

import openai
from pydantic import BaseModel, ValidationError
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from prompts import REFUSAL_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from retrieval import RetrievedChunk

MODEL = "gpt-5-mini"
TIMEOUT_S = 30
MAX_RETRIES = 2

log = logging.getLogger("pipeline")


class GroundedAnswer(BaseModel):
    """Schema the LLM must return."""

    supported: bool
    answer: str
    cited_chunk_ids: list[str]


@dataclass
class Answer:
    answer: str
    citations: list[str]
    cited_chunk_ids: list[str]
    supported: bool
    generator: str
    reason: str | None = None


def refusal(generator: str, reason: str) -> Answer:
    return Answer(REFUSAL_MESSAGE, [], [], False, generator, reason)


def _from_grounded(result: GroundedAnswer, retrieved: list[RetrievedChunk], generator: str) -> Answer:
    if not result.supported:
        return refusal(generator, "model_unsupported")
    doc_by_chunk = {r.chunk.chunk_id: r.chunk.doc_id for r in retrieved}
    # Raw ids are kept as-is so the validator can catch any id that was not retrieved.
    citations = list(dict.fromkeys(doc_by_chunk[c] for c in result.cited_chunk_ids if c in doc_by_chunk))
    return Answer(result.answer, citations, result.cited_chunk_ids, True, generator)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ENGLISH_STOP_WORDS}


def _extractive(question: str, retrieved: list[RetrievedChunk]) -> GroundedAnswer:
    """Baseline: the 1-2 sentences of the top chunk that share the most words with the question.

    ponytail: cannot detect partial support; it answers whatever overlaps. The LLM path handles that.
    """
    top = retrieved[0].chunk
    body = top.text.split("\n", 1)[-1]
    sentences = re.split(r"(?<=[.!?])\s+", body)
    question_words = _words(question)
    overlap = [len(_words(s) & question_words) for s in sentences]
    top_two = sorted(range(len(sentences)), key=lambda i: -overlap[i])[:2]
    best = sorted(i for i in top_two if overlap[i] > 0)
    if not best:
        return GroundedAnswer(supported=False, answer=REFUSAL_MESSAGE, cited_chunk_ids=[])
    return GroundedAnswer(supported=True, answer=" ".join(sentences[i] for i in best), cited_chunk_ids=[top.chunk_id])


def _openai(question: str, retrieved: list[RetrievedChunk]) -> tuple[Answer, dict | None]:
    """Structured-output call. Any API error, model refusal or unparseable output fails closed to a refusal."""
    client = openai.OpenAI(timeout=TIMEOUT_S, max_retries=MAX_RETRIES)
    try:
        response = client.responses.parse(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=build_user_prompt(question, retrieved),
            text_format=GroundedAnswer,
        )
    except (openai.OpenAIError, ValidationError) as error:
        return refusal("openai", f"generation_error:{type(error).__name__}"), None
    usage = {"input": response.usage.input_tokens, "output": response.usage.output_tokens} if response.usage else None
    if response.output_parsed is None:
        return refusal("openai", "unparsed_output"), usage
    return _from_grounded(response.output_parsed, retrieved, "openai"), usage


def generate(question: str, retrieved: list[RetrievedChunk]) -> Answer:
    """Uses OpenAI when OPENAI_API_KEY is set, otherwise the extractive baseline."""
    start = time.perf_counter()
    tokens = None
    if os.getenv("OPENAI_API_KEY", "").strip():
        answer, tokens = _openai(question, retrieved)
    else:
        answer = _from_grounded(_extractive(question, retrieved), retrieved, "extractive")
    log.info("generation", extra={"fields": {
        "generator": answer.generator, "supported": answer.supported, "reason": answer.reason,
        "tokens": tokens, "latency_ms": round((time.perf_counter() - start) * 1000),
    }})
    return answer
