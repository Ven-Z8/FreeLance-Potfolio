# Scorecard — RAG over SEC 10-K filings

*Generated 2026-07-23 18:16 · 20 golden questions · 25 filings · generation model `anthropic/claude-sonnet-4.5` · judge `anthropic/claude-sonnet-4.5`*

> ⚠️ Golden set verification pending: expected answers are PROVISIONAL (extracted from parsed filings, awaiting human verification against the original documents). Treat absolute numbers accordingly; the dense-vs-hybrid comparison is unaffected.

| Metric | Dense | Hybrid | Hybrid_rerank | Agent_react |
|---|---|---|---|---|
| Answer accuracy | — | 45% | 55% | 40% |
| Retrieval hit rate | — | 80% | 80% | 80% |
| Citation faithfulness | — | 67% | 67% | 67% |
| Hallucination rate (unanswerable) | — | 0% | 0% | 0% |
| Refusal correctness | — | 62% | 100% | 100% |
| Refusal rate | — | 40% | 25% | 25% |
| Cost / query | — | $0.0178 | $0.0192 | $0.0150 |
| Latency p50 | — | 3.6s | 4.6s | 4.9s |
| Latency p95 | — | 13.1s | 14.4s | 13.5s |
| Judge cost (eval overhead, total) | — | — | — | — |

## Accuracy by question category

| Category | Dense (n) | Hybrid (n) | Hybrid_rerank (n) | Agent_react (n) |
|---|---|---|---|---|
| lookup | — | 50% (2) | 50% (2) | 50% (2) |
| synthesis | — | 29% (7) | 43% (7) | 14% (7) |
| table | — | 17% (6) | 33% (6) | 17% (6) |
| unanswerable | — | 100% (5) | 100% (5) | 100% (5) |
