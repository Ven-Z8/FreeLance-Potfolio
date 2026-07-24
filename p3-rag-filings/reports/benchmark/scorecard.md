# Scorecard — RAG over SEC 10-K filings

*Generated 2026-07-23 16:42 · 91 golden questions · 25 filings · generation model `anthropic/claude-sonnet-4.5` · judge `anthropic/claude-sonnet-4.5`*

> ⚠️ Golden set verification pending: expected answers are PROVISIONAL (extracted from parsed filings, awaiting human verification against the original documents). Treat absolute numbers accordingly; the dense-vs-hybrid comparison is unaffected.

| Metric | Hybrid_rerank |
|---|---|
| Answer accuracy | 68% |
| Retrieval hit rate | 93% |
| Citation faithfulness | 61% |
| Hallucination rate (unanswerable) | 0% |
| Refusal correctness | 89% |
| Refusal rate | 21% |
| Cost / query | $0.0167 |
| Latency p50 | 4.5s |
| Latency p95 | 17.2s |
| Judge cost (eval overhead, total) | $0.040 |

## Accuracy by question category

| Category | Hybrid_rerank (n) |
|---|---|
| ambiguous | 60% (5) |
| lookup | 76% (25) |
| synthesis | 41% (17) |
| table | 63% (30) |
| unanswerable | 100% (14) |
