"""Deterministic checks on a generated answer. Pure: no I/O, no LLM."""

import re

from generation import Answer
from prompts import REFUSAL_MESSAGE
from retrieval import RetrievedChunk


def validate(answer: Answer, retrieved: list[RetrievedChunk]) -> list[str]:
    """Return a list of failure codes; empty means the answer passed every check.

    ponytail: proves each cited source was retrieved and every number in the answer appears in it, not that
    the claim is entailed by it. An NLI/entailment check would close that gap.
    """
    retrieved_ids = {r.chunk.chunk_id for r in retrieved}
    cited_text = " ".join(r.chunk.text for r in retrieved if r.chunk.chunk_id in answer.cited_chunk_ids)
    retrieved_docs = {r.chunk.doc_id for r in retrieved}
    failures = []
    if not answer.answer.strip():
        failures.append("empty_answer")
    if answer.supported and not answer.citations:
        failures.append("supported_without_citation")
    failures += [f"cited_chunk_not_retrieved:{c}" for c in answer.cited_chunk_ids if c not in retrieved_ids]
    failures += [f"citation_doc_not_retrieved:{d}" for d in answer.citations if d not in retrieved_docs]
    if answer.supported:
        failures += [f"number_not_in_cited_chunks:{n}" for n in _numbers(answer.answer) - _numbers(cited_text)]
    if not answer.supported and (answer.answer != REFUSAL_MESSAGE or answer.citations or answer.cited_chunk_ids):
        failures.append("malformed_refusal")
    return failures


def _numbers(text: str) -> set[str]:
    """Numbers with thousands separators removed, so "5,000" and "5000" match."""
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", text)}
