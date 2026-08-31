# Golden Question Set — Schema & Authoring Guide

One JSONL file, one test case per line. This format is shared with P1's harness
(see ../../p1-eval-harness/BUILD_BRIEF.md Phase 0) — change it there first if it
must change.

## Fields

| Field | Type | Notes |
|---|---|---|
| `id` | string | `fin-NNNN`, stable forever once assigned |
| `input` | string | The question, exactly as a user would ask it |
| `expected.answer` | string | Ground-truth answer (or `null` for unanswerable) |
| `expected.citations` | list | Filing + section that supports the answer, e.g. `"AAPL_2025_10K:Item7"` |
| `expected.type` | enum | `exact` \| `contains` \| `judge` — how to score |
| `variation_rules` | list | e.g. `numeric_tolerance:0.5%`, `unit_equivalence`, `fiscal_vs_calendar_year_ok` |
| `difficulty` | enum | `easy` \| `medium` \| `hard` |
| `failure_category` | enum | `lookup` \| `synthesis` \| `table` \| `unanswerable` \| `ambiguous` |
| `domain` | string | `financial` for this project |
| `notes` | string | Why this question is here / what it's designed to catch |

## Category targets (60–100 questions total)

| Category | Count | Purpose |
|---|---|---|
| Simple lookup | 15–25 | Baseline sanity — "What was NVDA's FY2025 revenue?" |
| Cross-section synthesis | 12–20 | Combine MD&A + risk factors + financials |
| Table-reading | 15–25 | Exact figures from real tables — exposes agents brutally |
| Not-in-corpus (unanswerable) | 10–15 | MUST refuse — hallucination test. `expected.answer: null` |
| Ambiguous | 8–15 | Should ask for clarification, not guess ("What was revenue?" — which company? which year?) |

## History

- **v0 (retired 2026-08-30):** 91 cases across `golden_set_v0_batch1..4`,
  `brutal_20`, `financebench_10k`, `benchmarks/`. An audit
  (`audit_v0_retired.json`) found the expected answers were never verified
  against the filings: fabricated figures (AMZN net sales $637,982 — actual
  FY2025: $716,924M), wrong-year figures (JNJ R&D $17,175 — actual FY2025:
  $14,665M), transcription errors (WMT $680,984 — actual: $680,985M). The
  entire set was retired rather than patched.
- **v1:** built from scratch with the stage-2 fact graph
  (`corpus/graph/financial_graph.json`) as the grounding source; every
  expected answer is checked against the source chunk text by
  `scripts/audit_golden.py` before entering the set.

## Authoring rules

1. Every answerable question's answer must be verified BY HAND against the
   actual filing before it enters the set. No LLM-generated ground truth.
2. Table questions: prefer figures that appear ONLY in a table (not repeated in prose).
3. Unanswerable questions should be plausible — about a company in the corpus,
   but facts not in a 10-K (e.g., intra-quarter stock price, competitor's private data).
4. Record the citation precisely enough that a human can check it in <1 minute.
5. Difficulty is about the retrieval+reasoning path, not obscurity.
