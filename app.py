"""Ask one question from the command line.

Usage: python app.py --question "How long do crypto withdrawals take?"
"""

import argparse
import json

from dotenv import load_dotenv

from run_pipeline import DOCS_DIR, answer_question, build_pipeline_index, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Grounded QA over the docs/ folder.")
    parser.add_argument("--question", required=True)
    question = parser.parse_args().question.strip()
    if not question:
        parser.error("--question must not be empty")

    load_dotenv()
    setup_logging()
    retrieved, answer, _ = answer_question(question, build_pipeline_index(DOCS_DIR))
    print(json.dumps({
        "question": question,
        "answer": answer.answer,
        "citations": answer.citations,
        "supported": answer.supported,
        "sources": [{"chunk_id": r.chunk.chunk_id, "doc_id": r.chunk.doc_id, "score": round(r.score, 4)} for r in retrieved],
        "generator": answer.generator,
        "reason": answer.reason,
    }, indent=2))


if __name__ == "__main__":
    main()
