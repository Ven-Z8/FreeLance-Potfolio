# Scorecard — RAG over SEC 10-K filings

*Generated 2026-07-23 15:00 · 61 golden questions · 25 filings · generation model `anthropic/claude-sonnet-4.5` · judge `anthropic/claude-sonnet-4.5`*

> ⚠️ Golden set verification pending: expected answers are PROVISIONAL (extracted from parsed filings, awaiting human verification against the original documents). Treat absolute numbers accordingly; the dense-vs-hybrid comparison is unaffected.

| Metric | Dense | Hybrid |
|---|---|---|
| Answer accuracy | 70% | 72% |
| Retrieval hit rate | 91% | 91% |
| Citation faithfulness | 53% | 63% |
| Hallucination rate (unanswerable) | 0% | 0% |
| Refusal correctness | 62% | 60% |
| Refusal rate | 34% | 33% |
| Cost / query | $0.0129 | $0.0131 |
| Latency p50 | 2.9s | 3.0s |
| Latency p95 | 7.0s | 5.2s |
| Judge cost (eval overhead, total) | $0.017 | $0.016 |

## Accuracy by question category

| Category | Dense (n) | Hybrid (n) |
|---|---|---|
| ambiguous | 60% (5) | 40% (5) |
| lookup | 76% (17) | 71% (17) |
| synthesis | 42% (12) | 33% (12) |
| table | 71% (17) | 94% (17) |
| unanswerable | 100% (10) | 100% (10) |
