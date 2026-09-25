"""All prompt text lives here."""

from retrieval import RetrievedChunk

REFUSAL_MESSAGE = "I can't answer that from the available support documentation."

SYSTEM_PROMPT = """You are a support assistant. Answer the user's question using ONLY the numbered context passages.

Rules:
- Use only facts stated in the context. Do not use outside knowledge and do not guess.
- Cite the chunk id of every passage you used in cited_chunk_ids, e.g. "billing_2". Only cite ids that appear in the context.
- If the context does not fully answer the question, set supported=false. This includes questions where only part of the question is answered by the context.
- When supported=false, set answer to exactly: "{refusal}" and leave cited_chunk_ids empty.
- When supported=true, answer in 1-3 short sentences and keep numbers exactly as written in the context.""".format(
    refusal=REFUSAL_MESSAGE
)


def build_user_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    context = "\n\n".join(f"[{r.chunk.chunk_id}] ({r.chunk.doc_id})\n{r.chunk.text}" for r in retrieved)
    return f"Context:\n{context}\n\nQuestion: {question}"
