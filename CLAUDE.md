# CLAUDE.md

Technical assessment repo (AI/ML engineering). Reviewers read the code, so clarity beats cleverness.

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

- Architecture diagrams: use `/excalidraw-diagram` and save them under `docs/`.
- Commit small, focused changes with clear messages.
