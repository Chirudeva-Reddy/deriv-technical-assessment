"""Load docs, chunk them by markdown section, and retrieve chunks with TF-IDF + BM25 (hybrid)."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

TOP_K = 3
MAX_CHUNK_CHARS = 800
# ponytail: set in the gap between the best unanswerable (0.165) and worst answerable (0.272) top score on
# the same eval questions, with no held-out set. Re-tune on a larger labelled set before trusting it.
MIN_SCORE = 0.2
# Textbook BM25 defaults and reciprocal-rank-fusion constant; not tuned on the eval set.
BM25_K1, BM25_B = 1.5, 0.75
RRF_K = 60

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # TF-IDF cosine in [0, 1]; the MIN_SCORE gate reads this, not the fused rank


@dataclass(frozen=True)
class Index:
    vectorizer: TfidfVectorizer
    matrix: sparse.csr_matrix
    bm25_vectorizer: CountVectorizer
    bm25_matrix: sparse.csr_matrix  # per-(chunk, term) BM25 weight, so a query score is one sparse product
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


def stem_tokens(text: str) -> list[str]:
    """Lowercase word tokens, English stop words removed, trailing plural 's' stripped ("fees" -> "fee").

    ponytail: plural-only stemmer, misses "processing"/"processed"; swap in a Porter stemmer if word forms keep missing.
    """
    tokens = [t for t in re.findall(r"\w+", text.lower()) if t not in ENGLISH_STOP_WORDS]
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in tokens]


def _bm25_weights(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    """Turn a (chunk x term) count matrix into BM25 term weights: idf * tf*(k1+1) / (tf + k1*length_norm)."""
    counts = counts.tocoo()
    n_chunks = counts.shape[0]
    doc_freq = np.bincount(counts.col, minlength=counts.shape[1])
    idf = np.log(1 + (n_chunks - doc_freq + 0.5) / (doc_freq + 0.5))
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    length_norm = 1 - BM25_B + BM25_B * lengths / lengths.mean()
    tf = counts.data
    weights = idf[counts.col] * tf * (BM25_K1 + 1) / (tf + BM25_K1 * length_norm[counts.row])
    return sparse.csr_matrix((weights, (counts.row, counts.col)), shape=counts.shape)


def build_index(chunks: list[Chunk]) -> Index:
    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    bm25_vectorizer = CountVectorizer(analyzer=stem_tokens)
    return Index(vectorizer, vectorizer.fit_transform(texts),
                 bm25_vectorizer, _bm25_weights(bm25_vectorizer.fit_transform(texts)), chunks)


def _rrf(scores: np.ndarray) -> np.ndarray:
    """Reciprocal-rank-fusion contribution 1/(RRF_K + rank) per chunk; chunks with score 0 contribute nothing."""
    ranks = np.empty(len(scores))
    ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return np.where(scores > 0, 1 / (RRF_K + ranks), 0.0)


def retrieve(index: Index, question: str, k: int = TOP_K) -> list[RetrievedChunk]:
    """Top-k chunks by reciprocal-rank fusion of TF-IDF cosine and BM25. Chunks neither method matches are dropped.

    Fusing ranks rather than scores means the two scales never need calibrating against each other.
    """
    cosine = (index.matrix @ index.vectorizer.transform([question]).T).toarray().ravel()
    query_terms = index.bm25_vectorizer.transform([question]).sign()
    bm25 = (index.bm25_matrix @ query_terms.T).toarray().ravel()
    fused = _rrf(cosine) + _rrf(bm25)
    top = np.lexsort((-cosine, -fused))[:k]  # ties on fused rank go to the higher cosine
    return [RetrievedChunk(index.chunks[i], float(cosine[i])) for i in top if fused[i] > 0]
