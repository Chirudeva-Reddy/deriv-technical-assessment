"""Turn a question plus retrieved chunks into an Answer. generate() is the only place the generator is chosen."""

import logging
import re
import time
from dataclasses import dataclass

from pydantic import BaseModel
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from prompts import REFUSAL_MESSAGE
from retrieval import RetrievedChunk

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


def generate(question: str, retrieved: list[RetrievedChunk]) -> Answer:
    start = time.perf_counter()
    answer = _from_grounded(_extractive(question, retrieved), retrieved, "extractive")
    log.info("generation", extra={"fields": {
        "generator": answer.generator, "supported": answer.supported, "reason": answer.reason,
        "latency_ms": round((time.perf_counter() - start) * 1000),
    }})
    return answer
