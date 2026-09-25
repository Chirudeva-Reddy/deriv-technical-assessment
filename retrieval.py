"""Load docs, chunk them by markdown section, and retrieve chunks with TF-IDF."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

TOP_K = 3
MAX_CHUNK_CHARS = 800
# ponytail: set in the gap between the best unanswerable (0.165) and worst answerable (0.272) top score on
# the same eval questions, with no held-out set. Re-tune on a larger labelled set before trusting it.
MIN_SCORE = 0.2

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Index:
    vectorizer: TfidfVectorizer
    matrix: "scipy.sparse.csr_matrix"
    chunks: list[Chunk]


def load_documents(docs_dir: str | Path) -> list[tuple[str, str]]:
    """Return (doc_id, text) for every .md/.txt file, sorted by name. doc_id is the file stem."""
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"docs directory not found: {docs_dir}")
    paths = sorted(p for p in docs_dir.iterdir() if p.suffix in {".md", ".txt"})
    if not paths:
        raise ValueError(f"no .md or .txt documents in {docs_dir}")
    docs = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"document is empty: {path}")
        docs.append((path.stem, text))
    return docs


def _sections(doc_id: str, text: str) -> tuple[str, list[tuple[str | None, str]]]:
    """Split text on headings. Returns the doc title (first H1, else doc_id) and (heading, body) pairs."""
    title, heading, lines, sections = None, None, [], []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append((heading, body))
        lines.clear()

    for line in text.splitlines():
        match = HEADING.match(line)
        if not match:
            lines.append(line)
            continue
        flush()
        if len(match.group(1)) == 1 and title is None:
            title, heading = match.group(2).strip(), None
        else:
            heading = match.group(2).strip()
    flush()
    return title or doc_id, sections


def _pack_paragraphs(body: str) -> list[str]:
    # ponytail: a single paragraph longer than MAX_CHUNK_CHARS stays whole; split on sentences if docs get long paragraphs
    parts: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body):
        paragraph = paragraph.strip()
        if parts and len(parts[-1]) + len(paragraph) + 2 <= MAX_CHUNK_CHARS:
            parts[-1] += "\n\n" + paragraph
        elif paragraph:
            parts.append(paragraph)
    return parts


def chunk_document(doc_id: str, text: str) -> list[Chunk]:
    """One chunk per section (split on paragraphs above MAX_CHUNK_CHARS), prefixed with 'Title > Heading'.

    Files without headings (plain .txt) become one section, so they fall back to paragraph packing.
    """
    title, sections = _sections(doc_id, text)
    chunks = []
    for heading, body in sections:
        prefix = f"{title} > {heading}" if heading else title
        for part in _pack_paragraphs(body):
            chunks.append(Chunk(doc_id, f"{doc_id}_{len(chunks)}", f"{prefix}\n{part}"))
    return chunks


def build_index(chunks: list[Chunk]) -> Index:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform([c.text for c in chunks])
    return Index(vectorizer, matrix, chunks)


def retrieve(index: Index, question: str, k: int = TOP_K) -> list[RetrievedChunk]:
    """Top-k chunks by cosine similarity (rows are L2-normalised, so a dot product is cosine). Zero scores dropped."""
    scores = (index.matrix @ index.vectorizer.transform([question]).T).toarray().ravel()
    top = np.argsort(-scores, kind="stable")[:k]
    return [RetrievedChunk(index.chunks[i], float(scores[i])) for i in top if scores[i] > 0]
