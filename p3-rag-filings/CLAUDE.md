# P3 — Production RAG over SEC 10-K Filings

Read BUILD_BRIEF.md before writing any code. This file is the working contract.

## What this project is

RAG over 20–30 real, messy SEC 10-K filings, with an eval suite and an honest
published failure analysis. Positioning: *"RAG over genuinely messy documents —
with an eval suite and an honest list of where it breaks."*

## Non-negotiable constraints (scope discipline)

- NO chat UI with history/streaming polish. A minimal query interface (CLI or single input box) is fine.
- NO 10,000-document corpus. 20–30 hard documents, deeply handled.
- NO agentic-RAG complexity in v1 (multi-hop planning etc.). Note it as future work with a hypothesis of what it would fix.
- Every answer must cite specific chunks. Numerical claims get a verification pass against the cited source.
- The system must refuse when retrieval confidence is low — and we measure the refusal rate and whether refusals were correct.

## Tech conventions

- Python 3.11+, uv for deps, ruff for lint.
- Config in YAML/TOML files, not hardcoded. CLI entry points via `python -m` or a console script.
- Structured traces: every query run logs retrieval results, chunks used, citations, cost, latency as JSON.
- Tests for the ingestion pipeline (table preservation, chunk boundaries) — these are the fragile parts.

## Directory layout

```
p3-rag-filings/
  corpus/            # raw filings (gitignore large files; keep manifest)
  scripts/           # download_corpus.py etc.
  golden/            # golden question set (JSONL) + schema
  src/ragfilings/    # ingestion, retrieval, generation, verification
  evals/             # eval configs + reports (feeds P1 harness later)
  reports/           # generated scorecards
```

## Definition of done

- Scorecard published with the two-retrieval-strategy comparison.
- Failure analysis section written ("The N questions my system got wrong, and why").
- Golden set feeds P1's financial domain pack (one dataset, two portfolio pieces).

## Working style

- Small commits, each independently explainable.
- When a design tradeoff comes up, choose the option that is easier to measure.
- If something fails, capture it — failures are portfolio content here, not embarrassments.

## Current state (stage 3 complete, 2026-08-31)

Stages 0–3 of the rebuild are committed; TRACKER.md at the repo root holds
stage status and the published numbers.

- **Golden set v1**: `golden/golden_set_v1.jsonl` (80 cases, every expected
  answer proven against filing text; drafts `candidates_v1.jsonl` /
  `handcrafted_v1.jsonl`; audit evidence `audit_v1.json`, retired-set audit
  `audit_v0_retired.json`). Never edit expected answers without re-running
  `scripts/audit_golden.py`.
- **Eval**: `ragfilings regress` runs the suite into
  `reports/evals/<timestamp>-<sha8>-<strategy>/` (results + traces +
  `run_meta.json`) and diffs vs the latest baseline. Judge = DeepEval G-Eval
  via `[eval].judge_model` (see `src/ragfilings/eval/deepeval_judge.py`).
- **Judge calibration**: agreement 86.5% / kappa 0.669 on 52 hand-labeled
  pairs — `golden/judge_calibration_v1.jsonl` + `docs/judge_calibration_v1.md`.
- **Models**: OpenRouter credits are exhausted; generation/extraction/judge
  all run on `minimax/minimax-m3:free` (config.toml). Re-run the baseline on
  frontier models if credits are added — run snapshots make this comparable.

**Known issues to work next (stage 4):**
1. 16/80 incorrect refusals — the generation model claims a figure is
   missing from retrieved context when it is present (e.g. UNH 447,567;
   NVDA gross profit). The graph tool should rescue these.
2. 3 hallucinations on unanswerables (fin-8003, fin-8007, fin-8012).
3. Ambiguous questions: the system enumerates interpretations instead of
   asking; the judge scores that incorrectly (known calibration bias, 6/10).
4. Graph-layer data errors documented in `scripts/build_golden_v1.py`
   (BAD_FACTS/EXCLUDE lists) — ~60 mis-extracted facts the graph still
   serves via `query_graph`.
5. Free-judge JSON instability (~13/240 metric calls return malformed JSON;
   degrade to None by design).
