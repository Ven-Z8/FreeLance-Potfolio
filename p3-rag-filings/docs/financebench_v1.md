# FinanceBench — external benchmark run (v1, 2026-08-31)

**Result: 122/150 = 81.3%** in *reasoning-over-evidence* mode, all-free models
(`minimax/minimax-m3:free` for generation + judging), $0.00 cost. 17 of the 150
were answered as refusals.

## What FinanceBench is

FinanceBench (Patronus AI) is an open suite of 150 questions about publicly
traded companies, grounded in real 10-K / 10-Q / 8-K / earnings documents. It
is intended as a minimum performance standard for financial QA. Each record
carries the question, the official answer, and the evidence excerpt (+ page)
the answer comes from. License: CC-BY-NC-4.0 (the dataset is downloaded at
runtime, not committed).

## How we ran it (reasoning-over-evidence)

FinanceBench references filings *outside* this repo's 25-filing corpus, and
those sources are PDFs (our ingestion parses HTML 10-Ks). So this v1 run does
**reasoning-over-evidence**: each question is given its official evidence
excerpt as the retrieved context, and the pipeline runs grounded synthesis +
numeric-claim verification + financial math on top. The fact graph is
deliberately disabled for this run (it is built from our 25 filings, not
FinanceBench's, so it would inject irrelevant facts).

What this measures: grounded synthesis, numeric-claim verification, and
financial math over provided evidence. What it does **not** measure:
retrieval (the evidence is given, not retrieved). The full-retrieval variant
(download the 10-K PDFs, add PDF parsing, index, retrieve) is the planned
follow-up.

Scoring: our calibrated G-Eval judge scores our answer against the official
FinanceBench answer (factual equivalence, threshold 0.5).

## Reproduce

The adapter lives in the sibling `p1-eval-harness` project (it shares the
judge layer); it needs the P3 pipeline, so install both packages into one
venv first:

```bash
cd ../p1-eval-harness
uv run python scripts/benchmark_financebench.py          # all 150
uv run python scripts/benchmark_financebench.py --limit 20   # subset
```
The adapter downloads FinanceBench from HuggingFace
(`PatronusAI/financebench`, `financebench_merged.jsonl`) on first use into
`p1-eval-harness/data/financebench/` and writes per-question results to
`p1-eval-harness/reports/financebench/`.

## Caveats, stated plainly

- **Reasoning-over-evidence, not full retrieval.** The retrieval layer is not
  exercised here; this isolates synthesis + verification + math. The
  golden-set runs in this repo measure retrieval end-to-end.
- **Free model + judge.** Generation and judging both use the free
  `minimax/minimax-m3:free` model; run-to-run variance is a few questions.
- **Judge-scored.** Correctness is the G-Eval factual-equivalence verdict
  against the official answer, not an exact string match.
