"""Deterministic checks on a generated answer. Pure: no I/O, no LLM."""

from generation import Answer
from prompts import REFUSAL_MESSAGE
from retrieval import RetrievedChunk


def validate(answer: Answer, retrieved: list[RetrievedChunk]) -> list[str]:
    """Return a list of failure codes; empty means the answer passed every check.

    ponytail: proves each cited source was retrieved, not that the claim is entailed by it.
    """
    retrieved_ids = {r.chunk.chunk_id for r in retrieved}
    retrieved_docs = {r.chunk.doc_id for r in retrieved}
    failures = []
    if not answer.answer.strip():
        failures.append("empty_answer")
    if answer.supported and not answer.citations:
        failures.append("supported_without_citation")
    failures += [f"cited_chunk_not_retrieved:{c}" for c in answer.cited_chunk_ids if c not in retrieved_ids]
    failures += [f"citation_doc_not_retrieved:{d}" for d in answer.citations if d not in retrieved_docs]
    if not answer.supported and (answer.answer != REFUSAL_MESSAGE or answer.citations or answer.cited_chunk_ids):
        failures.append("malformed_refusal")
    return failures
