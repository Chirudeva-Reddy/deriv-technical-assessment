import json
from types import SimpleNamespace

import httpx
import openai
import pytest

import generation
import retrieval
import run_pipeline
from generation import Answer, GroundedAnswer, generate, refusal
from prompts import REFUSAL_MESSAGE
from retrieval import Chunk, RetrievedChunk, build_index, chunk_document, load_documents, retrieve
from run_pipeline import answer_question, load_questions
from validation import validate


def test_chunks_keep_title_heading_and_ids():
    text = "# Refunds\n\n## Cards\nCard refunds take 5 days.\n\n## Bank\nBank refunds take 2 days.\n"
    chunks = chunk_document("refunds", text)
    assert [c.chunk_id for c in chunks] == ["refunds_0", "refunds_1"]
    assert all(c.doc_id == "refunds" for c in chunks)
    assert chunks[1].text == "Refunds > Bank\nBank refunds take 2 days."


def test_long_section_splits_on_paragraphs(monkeypatch):
    monkeypatch.setattr(retrieval, "MAX_CHUNK_CHARS", 30)
    chunks = chunk_document("d", "# T\n## S\nfirst paragraph here.\n\nsecond paragraph here.")
    assert [c.text for c in chunks] == ["T > S\nfirst paragraph here.", "T > S\nsecond paragraph here."]


def test_plain_text_falls_back_to_paragraphs():
    chunks = chunk_document("notes", "Just a note.\n\nAnother one.")
    assert len(chunks) == 1 and chunks[0].text.startswith("notes\n")


def test_retrieve_returns_expected_doc(tmp_path):
    (tmp_path / "pets.md").write_text("# Pets\n## Dogs\nDogs need a daily walk.")
    (tmp_path / "tax.md").write_text("# Tax\n## Filing\nTax returns are due in April.")
    chunks = [c for doc_id, text in load_documents(tmp_path) for c in chunk_document(doc_id, text)]
    results = retrieve(build_index(chunks), "When are tax returns due?")
    assert results[0].chunk.doc_id == "tax"
    assert results == sorted(results, key=lambda r: -r.score)


def test_load_documents_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")
    with pytest.raises(ValueError):
        load_documents(tmp_path)
    (tmp_path / "empty.md").write_text("  \n")
    with pytest.raises(ValueError):
        load_documents(tmp_path)


RETRIEVED = [RetrievedChunk(Chunk("withdrawals", "withdrawals_0", "W > Fees\nBank transfers cost 5 USD."), 0.5)]


def test_validator_accepts_grounded_answer_and_clean_refusal():
    good = Answer("Bank transfers cost 5 USD.", ["withdrawals"], ["withdrawals_0"], True, "test")
    assert validate(good, RETRIEVED) == []
    assert validate(refusal("test", "x"), RETRIEVED) == []


def test_validator_rules():
    assert validate(Answer(" ", ["withdrawals"], ["withdrawals_0"], True, "t"), RETRIEVED) == ["empty_answer"]
    assert "supported_without_citation" in validate(Answer("5 USD", [], [], True, "t"), RETRIEVED)
    assert validate(Answer("5 USD", ["withdrawals"], ["withdrawals_0", "kyc_1"], True, "t"), RETRIEVED) == [
        "cited_chunk_not_retrieved:kyc_1"
    ]
    assert validate(Answer("5 USD", ["kyc"], ["withdrawals_0"], True, "t"), RETRIEVED) == ["citation_doc_not_retrieved:kyc"]
    assert validate(Answer("Probably 5 USD", [], [], False, "t"), RETRIEVED) == ["malformed_refusal"]
    assert validate(Answer("It costs 7 USD.", ["withdrawals"], ["withdrawals_0"], True, "t"), RETRIEVED) == [
        "number_not_in_cited_chunks:7"
    ]
    thousands = [RetrievedChunk(Chunk("limits", "limits_0", "The limit is 10,000 USD."), 0.5)]
    assert validate(Answer("The limit is 10000 USD.", ["limits"], ["limits_0"], True, "t"), thousands) == []


def small_index():
    chunks = [Chunk("withdrawals", "withdrawals_0", "Withdrawals > Fees\nBank transfers cost a flat 5 USD.")]
    return build_index(chunks)


def test_gate_refuses_low_confidence_without_generating(monkeypatch):
    monkeypatch.setattr(run_pipeline, "generate", lambda *a: pytest.fail("generator must not be called"))
    _, answer, _ = answer_question("What is the capital of Australia?", small_index())
    assert (answer.supported, answer.reason, answer.answer) == (False, "low_retrieval_confidence", REFUSAL_MESSAGE)


def test_fails_closed_when_generator_cites_unretrieved_chunk(monkeypatch):
    fake = Answer("It costs 5 USD.", ["withdrawals"], ["kyc_verification_1"], True, "fake")
    monkeypatch.setattr(run_pipeline, "generate", lambda *a: fake)
    _, answer, failures = answer_question("How much do bank transfers cost?", small_index())
    assert "cited_chunk_not_retrieved:kyc_verification_1" in failures
    assert (answer.supported, answer.reason, answer.answer, answer.citations) == (
        False, "validation_failed", REFUSAL_MESSAGE, []
    )


@pytest.mark.parametrize("bad", [
    {"id": "q1"},
    [{"id": "q1", "question": "?", "expected_behavior": "maybe"}],
    [{"id": "", "question": "?", "expected_behavior": "answerable"}],
    [{"id": "q1", "question": "?", "expected_behavior": "answerable", "expected_docs": "withdrawals"}],
])
def test_load_questions_rejects_bad_input(tmp_path, bad):
    path = tmp_path / "q.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_questions(path)


def test_extractive_answers_from_top_chunk_or_refuses(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answer = generate("How much do bank transfers cost?", RETRIEVED)
    assert (answer.answer, answer.citations, answer.cited_chunk_ids) == (
        "Bank transfers cost 5 USD.", ["withdrawals"], ["withdrawals_0"]
    )
    assert generate("Who wrote Hamlet?", RETRIEVED).reason == "model_unsupported"


def fake_openai(monkeypatch, result):
    """Replace the OpenAI client: parse() raises `result` if it is an exception, else returns it as output_parsed."""
    def parse(**kwargs):
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(output_parsed=result, usage=None)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(generation.openai, "OpenAI", lambda **kw: SimpleNamespace(responses=SimpleNamespace(parse=parse)))


@pytest.mark.parametrize("result, reason", [
    (openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com")), "generation_error:APITimeoutError"),
    (None, "unparsed_output"),
    (GroundedAnswer(supported=False, answer="Maybe 7 USD", cited_chunk_ids=["withdrawals_0"]), "model_unsupported"),
])
def test_llm_failures_become_refusals(monkeypatch, result, reason):
    fake_openai(monkeypatch, result)
    answer = generate("How much do bank transfers cost?", RETRIEVED)
    assert (answer.supported, answer.answer, answer.citations, answer.reason) == (False, REFUSAL_MESSAGE, [], reason)


def test_llm_citations_map_to_docs_and_keep_raw_ids(monkeypatch):
    fake_openai(monkeypatch, GroundedAnswer(supported=True, answer="5 USD.", cited_chunk_ids=["withdrawals_0", "ghost_9"]))
    answer = generate("How much do bank transfers cost?", RETRIEVED)
    assert answer.citations == ["withdrawals"]
    assert answer.cited_chunk_ids == ["withdrawals_0", "ghost_9"]


def test_main_writes_artifacts_and_scores(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "fees.md").write_text("# Fees\n## Bank transfers\nBank transfers cost a flat 5 USD.")
    (docs / "pets.md").write_text("# Pets\n## Dogs\nDogs need a daily walk.")
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([
        {"id": "a", "question": "How much do bank transfers cost?", "expected_behavior": "answerable",
         "expected_docs": ["fees"]},
        {"id": "b", "question": "What is the capital of Peru?", "expected_behavior": "unanswerable"},
    ]))
    monkeypatch.setenv("OPENAI_API_KEY", "")  # load_dotenv does not override it, so the extractive path runs
    for name, value in [("DOCS_DIR", docs), ("QUESTIONS_PATH", questions),
                        ("OUTPUT_DIR", tmp_path / "out"), ("LOG_PATH", tmp_path / "logs" / "p.jsonl")]:
        monkeypatch.setattr(run_pipeline, name, value)

    run_pipeline.main()

    out = tmp_path / "out"
    assert {p.name for p in out.iterdir()} == {"retrieval_results.json", "answers.json", "validation_report.json"}
    report = json.loads((out / "validation_report.json").read_text())
    summary = report["summary"]
    summary.pop("min_score")
    assert summary == {
        "generators": ["extractive", "gate"], "questions": 2, "checks_pass_rate": 1.0, "behavior_accuracy": 1.0,
        "hit@3": "1/1", "citation_hit": "1/1",
    }
    assert [q["retrieval_hit"] for q in report["questions"]] == [True, None]
