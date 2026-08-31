# Master AI Engineering Portfolio Progress Tracker

**Current focus:** Rebuild P3 + P1 into an enterprise-grade Agentic Graph RAG
system with a DeepEval-based evaluation harness, proven on SEC 10-K filings.

> **Integrity note (2026-08-30):** A full code audit found that prior TRACKER
> claims materially overstated the state of this repo — fabricated agent
> metrics, headline scores contradicted by the repo's own result files, eval
> adapters that returned the golden answer on error, and projects marked
> complete that do not exist (P4). Every claim below is traceable to an
> artifact in the repo. The pre-rebuild state is preserved in commit
> `a6aba7e` for reference.

---

## 📌 Project Status Overview

| Project | Honest Status | Measured Reality |
|---|---|---|
| **P3** SEC 10-K RAG | 🔨 **REBUILD IN PROGRESS** → Agentic Graph RAG | Real corpus (25 actual 10-Ks, 8,419 chunks) + real ingestion/retrieval core; ~68–72% accuracy on hand-built golden questions; agentic layer was a facade and is being rebuilt |
| **P1** Eval Harness | 🔨 **REBUILD IN PROGRESS** → DeepEval-based | Financial-domain adapter genuinely ran against P3 (29 real traces); no LLM judge existed; other domain packs were ungrounded and are being rebuilt or dropped |
| **P2** Multi-Agent Workflow | ⚠️ Minimal prototype | ~450 LOC pipeline skeleton; claims of durable state, CRM execution, and evals were not implemented |
| **P4** Browser Agent | ❌ Does not exist | No code; removed from claims |
| **P5** Cost Optimization | ⚠️ Stub | ~110 LOC; the published 75%/63% numbers have no supporting experiment artifacts |

---

## 🎯 The Rebuild (P3 + P1)

**Goal:** Enterprise-production Agentic Graph RAG + multi-agent orchestration,
proven by a DeepEval evaluation harness — first on SEC 10-K filings, then
generalized to arbitrary complex domains via pluggable domain skill packs.

### What already exists and is kept (audit-verified real)
- EDGAR downloader + section parser + table-aware chunker (`p3: ingestion.py, chunking.py`) — genuinely sophisticated
- Dense (bge-small) + BM25/RRF hybrid retrieval + cross-encoder rerank over a persisted 8,419-chunk index
- Deterministic numeric verification + confidence-gated refusal (14/14 unanswerables correctly refused)
- Golden-set schema + financial eval adapter with real recorded traces

### What was removed (facade)
- Hardcoded ticker dictionaries and score boosts fitted to the golden set (replaced with caller-driven metadata filters)
- Fabricated per-agent token/cost constants; unused "ReAct" tools; the self-grading adapter fallback
- Unreproducible headline numbers (76.1% / 92.4% / 85.0% etc. — contradicted by the repo's own result files)

### Build stages
| Stage | Deliverable | Status |
|---|---|---|
| 0 | Hygiene & honesty: checkpoint commit, kill hacks/dead code, honest config + tracker | ✅ |
| 1 | Real agent core: instructor-validated structured outputs, genuinely invoked tools, real token/cost accounting | ⬜ |
| 2 | Graph RAG: typed fact graph (Company→Year→Metric→Value with provenance) + GraphRAG-style community summaries + graph query tool | ⬜ |
| 3 | DeepEval harness: DeepEval metric layer, cleaned golden set, calibrated judge with published human-agreement number, regression runner | ⬜ |
| 4 | Proof: dense vs hybrid vs +rerank vs +graph ablations, failure analysis, honest scorecards, docs + web | ⬜ |

### Verified baseline numbers (from repo artifacts)
- Financial golden subset (20 cases, `p1-eval-harness/reports/scorecard_hybrid_rerank.md`): **55.0%**
- Hybrid strategy, 61-case golden set (`p3-rag-filings/reports/results_hybrid.jsonl`): **72.1%**
- Refusal correctness on unanswerables (91-case run): **14/14**
- These are the starting line. Every future number must come from a reproducible run with config snapshot.
