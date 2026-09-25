# Decisions

Choices made before and during the build, and why.

| # | Decision | Why |
|---|----------|-----|
| D1 | Generator: OpenAI `gpt-5-mini`, with an extractive fallback when `OPENAI_API_KEY` is unset or empty | A real LLM for the graded run. The fallback keeps the pipeline runnable without a key and doubles as the baseline |
| D2 | Retrieval: TF-IDF (scikit-learn, 1–2-grams, English stop words, sublinear tf, cosine) | Deterministic, no model download, and scores fall in [0, 1], so a threshold means something |
| D3 | Interface: argparse CLI, `python app.py --question "..."` | Stdlib only, no server to start |
| D4 | Pinned `requirements.txt`; one main command, `python run_pipeline.py` | Runs with nothing but pip |
| D5 | Validator failure → fail closed: the answer is replaced by the refusal and `supported=false` | No unvalidated claim reaches the user |
| D6 | Chunking by markdown section; sections over 800 chars are split on paragraphs; no overlap; `chunk_id = <stem>_<n>` | One policy rule per chunk, with its heading kept in the text as context |
| D7 | One flat module per stage (`retrieval`, `prompts`, `generation`, `validation`, `run_pipeline`) | Each boundary a reviewer looks for is its own file |
| D8 | Stretch item: retrieval confidence gate. If the top score is below `MIN_SCORE` (0.2), refuse before generation | A deterministic guardrail that costs nothing and also gives the extractive fallback a way to refuse |
| D9 | Generation errors (timeout, API error after 2 retries, model refusal, unparsed output) → fail closed with a `reason` | Consistent with D5 |
| D10 | `DECISIONS.md` at the root; diagram in `design/` | `docs/` is the corpus and may be swapped by the evaluator |
| D11 | `python-dotenv` `load_dotenv()` in the entry points; `.env.example` committed | Keys stay in the gitignored `.env` |
| D12 | Corpus: 5 docs about a fictional platform, "Tradeport" (auth, rate limits, KYC, withdrawals, incident escalation), with concrete numbers | Answers can be checked against specific figures |
| D13 | Eval set: 5 answerable + 4 unanswerable (off-topic, plausible but absent, two partially answerable), each with `expected_docs`. q9 was added after review because it clears the gate, so the model's own refusal is scored, not only checked by hand | Allows hit@k, citation-hit and behaviour-accuracy scoring |
| D14 | The `CLAUDE.md` diagram path points to `design/` | Removes the clash with the corpus folder |
| D15 | Commit per slice on `main`, push once at the end | The remote never holds a half-built pipeline |
| D16 | pytest, plain functions, no conftest; the LLM is never called in tests | Tests run offline and for free |
| D17 | JSON-lines log in `logs/pipeline.jsonl` (gitignored) plus short stderr lines | "A simple structured log file" |
| D18 | Committed artifacts come from `gpt-5-mini`; the extractive run is the baseline; both are in the README | Every model has to beat a baseline |
| D19 | The validator also checks that every number in a supported answer appears in the cited chunks | Numbers are the cheapest fabrication to catch deterministically, and the policy docs are full of them |
| D20 | The report adds `citation_hit` (cited docs ∩ `expected_docs`) next to retrieval hit@k | Shows the answer cites the right source, not just that retrieval found it |
