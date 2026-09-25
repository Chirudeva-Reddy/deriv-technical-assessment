"""Run every question in questions.json through retrieve -> gate -> generate -> validate and write the artifacts to outputs/.

Usage: python run_pipeline.py
"""

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from generation import Answer, generate, refusal
from retrieval import MIN_SCORE, TOP_K, Index, RetrievedChunk, build_index, chunk_document, load_documents, retrieve
from validation import validate

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "docs"
QUESTIONS_PATH = ROOT / "questions.json"
LOG_PATH = ROOT / "logs" / "pipeline.jsonl"
OUTPUT_DIR = ROOT / "outputs"
BEHAVIORS = {"answerable", "unanswerable"}

log = logging.getLogger("pipeline")


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"ts": record.created, "event": record.getMessage(), **getattr(record, "fields", {})})


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(JsonLineFormatter())
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter("[%(message)s] %(fields)s", defaults={"fields": ""}))
    log.handlers = [file_handler, stderr_handler]
    log.setLevel(logging.INFO)


def build_pipeline_index(docs_dir: Path) -> Index:
    docs = load_documents(docs_dir)
    log.info("docs_loaded", extra={"fields": {"count": len(docs), "doc_ids": [d for d, _ in docs]}})
    chunks = [c for doc_id, text in docs for c in chunk_document(doc_id, text)]
    log.info("chunks_created", extra={"fields": {"count": len(chunks)}})
    index = build_index(chunks)
    log.info("index_built", extra={"fields": {"vocabulary": len(index.vectorizer.vocabulary_)}})
    return index


def load_questions(path: Path) -> list[dict]:
    """Load and validate the eval set; raises ValueError naming the first bad item."""
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError(f"{path}: expected a JSON list of questions")
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            raise ValueError(f"{path}[{i}]: expected an object")
        for key in ("id", "question"):
            if not isinstance(q.get(key), str) or not q[key].strip():
                raise ValueError(f"{path}[{i}]: '{key}' must be a non-empty string")
        if q.get("expected_behavior") not in BEHAVIORS:
            raise ValueError(f"{path}[{i}]: 'expected_behavior' must be one of {sorted(BEHAVIORS)}")
        if not isinstance(q.get("expected_docs", []), list):
            raise ValueError(f"{path}[{i}]: 'expected_docs' must be a list")
    return questions


def answer_question(question: str, index: Index) -> tuple[list[RetrievedChunk], Answer, list[str]]:
    """The orchestrator. Returns (retrieved, final answer, validation failures of the generated answer)."""
    retrieved = retrieve(index, question)
    log.info("retrieval", extra={"fields": {
        "question": question, "results": [(r.chunk.chunk_id, round(r.score, 3)) for r in retrieved],
    }})
    top_score = max((r.score for r in retrieved), default=0.0)
    if top_score < MIN_SCORE:
        log.info("gate_refused", extra={"fields": {"top_score": round(top_score, 3), "min_score": MIN_SCORE}})
        return retrieved, refusal("gate", "low_retrieval_confidence"), []

    answer = generate(question, retrieved)
    failures = validate(answer, retrieved)
    log.info("validation", extra={"fields": {"passed": not failures, "failures": failures}})
    if failures:
        answer = refusal(answer.generator, "validation_failed")
    return retrieved, answer, failures


def main() -> None:
    load_dotenv()
    setup_logging()
    index = build_pipeline_index(DOCS_DIR)
    questions = load_questions(QUESTIONS_PATH)

    retrieval_results, answers, report = [], [], []
    for q in questions:
        retrieved, answer, failures = answer_question(q["question"], index)
        retrieval_results.append({"question_id": q["id"], "retrieved_chunks": [
            {"doc_id": r.chunk.doc_id, "chunk_id": r.chunk.chunk_id, "score": round(r.score, 4), "text": r.chunk.text}
            for r in retrieved
        ]})
        answers.append({"question_id": q["id"], **asdict(answer)})
        expected_docs = q.get("expected_docs", [])
        answerable = q["expected_behavior"] == "answerable"
        report.append({
            "question_id": q["id"],
            "checks_passed": not failures,
            "failures": failures,
            "expected_behavior": q["expected_behavior"],
            "supported": answer.supported,
            "behavior_match": answer.supported == answerable,
            "retrieval_hit": bool({r.chunk.doc_id for r in retrieved} & set(expected_docs)) if answerable else None,
            "citation_hit": bool(set(answer.citations) & set(expected_docs)) if answerable else None,
        })

    hits = [r["retrieval_hit"] for r in report if r["retrieval_hit"] is not None]
    cited = [r["citation_hit"] for r in report if r["citation_hit"] is not None]
    summary = {
        "generators": sorted({a["generator"] for a in answers}),
        "questions": len(report),
        "checks_pass_rate": sum(r["checks_passed"] for r in report) / len(report),
        "behavior_accuracy": sum(r["behavior_match"] for r in report) / len(report),
        f"hit@{TOP_K}": f"{sum(hits)}/{len(hits)}",
        "citation_hit": f"{sum(cited)}/{len(cited)}",
        "min_score": MIN_SCORE,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, data in [
        ("retrieval_results.json", retrieval_results),
        ("answers.json", answers),
        ("validation_report.json", {"summary": summary, "questions": report}),
    ]:
        (OUTPUT_DIR / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    log.info("done", extra={"fields": summary})


if __name__ == "__main__":
    main()
