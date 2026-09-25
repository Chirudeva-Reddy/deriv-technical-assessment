import retrieval
from retrieval import build_index, chunk_document, load_documents, retrieve


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
