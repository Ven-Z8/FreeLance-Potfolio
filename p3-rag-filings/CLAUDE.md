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
