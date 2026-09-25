# CLAUDE.md

Technical assessment repo (AI/ML engineering). Reviewers read the code, so clarity beats cleverness.

## Assessment technical constraints (hard requirements, from the brief)

These come straight from the assessment brief. They override every other section in this file. Before finishing any change, re-read this list and check the change against each item. If a change would break one, stop and raise it with the user; don't work around it quietly.

The system is a document QA pipeline: `docs/` is the knowledge corpus, and `questions.json` is the fixed eval set.

### 1. Use only local input documents for knowledge
- Every fact in an answer must come from a chunk retrieved from `docs/` in that run. No web search, no external APIs used as a knowledge source, and no fallback to the LLM's own world knowledge.
- The LLM (if one is configured) is only a writer over the retrieved chunks. The generation prompt must tell it to answer only from the supplied context and to refuse when the context does not contain the answer.
- The extractive fallback (no `OPENAI_API_KEY`) must also build its answer only from retrieved text.
- `docs/` is read at runtime through `retrieval.load_documents`. Don't copy doc contents into code, prompts, or constants.
- Check: q6 ("capital of Australia") and q7 (forex leverage) must be refused even though a general-purpose LLM "knows" or could guess an answer.

### 2. Do not hardcode answers to the sample questions
- No question text, question id (`q1`...`q8`), expected answer, or expected doc id from `questions.json` may appear in pipeline code, prompts, few-shot examples, keyword lists, or thresholds.
- `questions.json` is read only by the eval script and tests, never by retrieval, generation, or validation.
- Tuning (e.g. `MIN_SCORE`, `TOP_K`) must be a general rule justified in a comment or the README, not a value picked so that one specific question passes. Record what was tuned and on what.
- Check: the pipeline has to behave sensibly on a new question it has never seen. `grep -rn "q[1-8]\|capital of Australia\|SEV1" *.py` should find nothing outside the eval/tests.

### 3. Keep retrieval, generation, and validation as separate steps in code
- Three distinct functions with explicit inputs and outputs, called in sequence by a thin orchestrator:
  - **retrieve**: `question -> list[RetrievedChunk]` (lives in `retrieval.py`). Knows nothing about LLMs or answers.
  - **generate**: `question + retrieved chunks -> draft answer with citations`. Does no retrieval of its own.
  - **validate**: `draft answer + retrieved chunks -> accepted answer or refusal`. Does no retrieval or generation of its own, and doesn't trust the generator's claims about itself.
- Each step must be callable and testable on its own, with no hidden shared state and no step reaching into another's internals.
- Keep each step in its own module or clearly separated functions. Don't merge them into one "ask the LLM to retrieve, answer and check" call.

### 4. Do not treat citations as optional
- Every answer that is not a refusal must cite at least one source, as `doc_id` + `chunk_id` pointing to a chunk that was actually retrieved for this question.
- Validation must reject (turn into a refusal) any answer that has no citations, cites a chunk that was not retrieved, or cites a doc/chunk id that does not exist.
- Citations must be structured output (fields in a returned object), not just text the model may or may not include. Parse them and check them.
- Check: the eval reports whether the cited docs match `expected_docs` for each answerable question.

### 5. Unsupported questions must not receive confident fabricated answers
- Two refusal gates, both required:
  1. **Retrieval gate**: if no chunk scores at or above `MIN_SCORE`, refuse without calling the generator.
  2. **Validation gate**: if the draft answer is not supported by its cited chunks, refuse.
- A refusal is an explicit, structured outcome (e.g. `answered: False` with a reason), not an answer string that happens to sound unsure.
- **Partial support counts as unsupported.** If a question asks for two things and the docs cover only one, the pipeline must not invent the missing part. It either refuses or answers the supported part and clearly states that the rest is not in the documents. Example: q8 asks for Level 2 and Level 3 daily limits; `docs/withdrawals.md` only gives Level 2.
- When in doubt, refuse. A wrong confident answer is worse than a refusal.
- Check: q6, q7 and q8 must come out as refusals (or, for q8, a clearly flagged partial answer) in the eval.

### 6. The solution must be runnable and understandable within the time limit
- One command from a clean checkout (`pip install -r requirements.txt` then one script), documented in the README.
- Must run with no API key (extractive fallback) and with `OPENAI_API_KEY` set. A missing key is not an error.
- No services to start, no databases, no downloads at runtime, no GPU. Pinned deps only (`requirements.txt`).
- A reviewer should understand the whole flow by reading a few short files top to bottom.

### 7. Favor clarity and correct boundaries over heavy infrastructure
- No vector DB, no LangChain/LlamaIndex-style framework, no agents, no async, no caching layers, no plugin systems. TF-IDF + scikit-learn is the retrieval baseline; replace it only if the eval shows it failing, and write down why.
- "Correct boundaries" means: validate at I/O edges (doc loading, `questions.json`, LLM output parsing), and keep the three pipeline steps from item 3 cleanly separated.
- Handle LLM API errors, timeouts and malformed structured output explicitly, and fall back to a refusal or the extractive generator. Never let them crash into a fabricated answer.

## Prime directive: don't over-engineer

- Build only what the task asks for. No speculative features, plugin systems, or "for later" scaffolding.
- No abstraction with a single implementation (base classes, factories, registries, strategy patterns) until a second real case exists.
- Before writing, look for something that already does the job, in this order: code already in this repo, the stdlib, an already-installed library, and only then new code.
- Only add a dependency when it saves real code. `numpy`/`pandas`/`scikit-learn` before a framework, and a framework only when the task needs it.
- Flat and few files. A script plus a small module beats a deep package tree.
- Config: plain constants at the top of the file, or env vars for secrets. No config framework for a handful of values.
- Prefer deleting code to adding it. The shortest diff that is correct wins.
- When you simplify on purpose, leave a one-line comment that names the limit and when to upgrade.

## ML best practices (non-negotiable)

**Data**
- Split before any fitting or preprocessing. Fit scalers, encoders and imputers on train only, then transform val/test.
- Watch for leakage: target-derived features, future information in time series (split by time, never shuffle), and duplicates across splits.
- Look at the data first: shape, dtypes, nulls, class balance, target distribution. Write down what you assume.
- Validate inputs at the edges (file loads, API payloads) and fail loudly on bad schema.

**Modeling**
- Start with a baseline (majority class, mean, linear or logistic regression). Every complex model has to beat it.
- Choose metrics that fit the problem: PR-AUC or F1 for imbalanced data, and MAE vs RMSE chosen on purpose. Accuracy alone is rarely enough.
- Use cross-validation for small data and a held-out test set that is touched once, at the end.
- Tune with a small, justified search. No giant grids without a reason.

**Reproducibility**
- Fix random seeds (`random`, `numpy`, and the framework if one is used).
- Pin dependency versions (`requirements.txt` or `pyproject.toml`).
- Scripts run end to end from a clean checkout with one command, documented in the README.
- Never commit data dumps, model weights or `.env`. Keep secrets in env vars only.

**Evaluation and reporting**
- Report metrics with context: baseline vs model, and variance across folds or seeds when it matters.
- Include error analysis: where the model fails and why.
- State the limits and what you would do next with more time or data.

**LLM / GenAI work (if applicable)**
- Keep prompts in one visible place, not scattered through the code.
- Handle structured output, API errors, timeouts and retries explicitly.
- Evaluate on a small fixed test set with clear pass/fail criteria, not just a few manual checks.
- Log token and cost usage when it is relevant.

## Code standards

- Python: type hints on public functions, small pure functions, and descriptive names. No comments that just restate the code.
- Notebooks for exploration only. Logic that matters lives in `.py` files that notebooks import.
- Tests: one small runnable check per non-trivial piece of logic (data transforms, metrics, parsing). No test frameworks or fixtures beyond what's needed.
- Error handling at boundaries (I/O, network, user input). Don't wrap internal code in try/except.

## Tooling

- Architecture diagrams: use `/excalidraw-diagram` and save them under `design/` (`docs/` is the QA corpus).
- Commit small, focused changes with clear messages.
