# Stage 4 — Deterministic Fact-Graph Augmentation (`hybrid_rerank_graph`)

**Date:** 2026-08-31 · **Run:** `reports/evals/20260831-114754-70d5fc72-hybrid_rerank_graph`
· **Code:** commits `690ad57`…`70d5fc7` · **Models:** all-free
(`minimax/minimax-m3:free` for generation/extraction/judge), cost $0.00

## The problem being fixed

The stage-3 baseline (53.8%, `hybrid_rerank`) had one dominant failure mode:
**16 incorrect refusals**. The free generation model answered *"the figure is
not in the provided chunks"* when retrieval had simply missed the chunk that
contains it. The fact graph (built in stage 2) already held the exact figure,
deterministically parsed from the 10-K tables with chunk provenance — the
pipeline just never consulted it.

## The change

For a **clean-scope** question — a complete `(ticker, metric, fiscal-year)`
triple with no qualifier — the pipeline extracts that scope from the question
text alone (no LLM) and, if the fact graph holds the exact fact, injects the
figure(s) **and their provenance chunks** into the synthesis context *before*
the model answers. Retrieval stops being a single point of failure.

Two robustness details discovered while measuring:

- **Refusal-prose.** The free model does not reliably refuse via `answer=null`;
  it often returns the refusal as prose in the answer field, and sometimes
  answers with a wrong-but-related metric instead. Augmenting up front (rather
  than "rescue on refusal") sidesteps both, and a refusal-prose detector is
  kept as a fallback.
- **Derived figures.** Change/compare questions state a delta or percent that
  is in no chunk; those are computed deterministically from the injected facts
  and grounded for the numerical-verification pass.

### Guard rails (a wrong injection would turn a correct refusal into a hallucination)

- fire only on **complete** `(ticker, metric, year)` triples;
- metric scope from **unambiguous multi-word statement phrases** only (bare
  "revenue" is rejected — it would match "WhatsApp revenue");
- **abort on any residual qualifier** ("first quarter", "data center", a
  leftover noun) or sub-period phrasing;
- never surface **mis-extracted facts** — `corpus/graph/excluded_facts.json`
  (68 facts verified wrong by human review) is the single source of truth for
  both the golden build and the runtime;
- **ticker symbols matched case-sensitively** so the lowercased symbol `cost`
  cannot collide with the phrase "cost of revenue" (this silently disabled
  augmentation for fin-2002 until fixed).

Coverage check (deterministic, no LLM): all 16 former refusals resolve to a
graph fact; **all 12 unanswerables and all 10 ambiguous questions abstain**
(no clean scope and/or no graph fact), so the mechanism cannot force an answer
where none is warranted.

## Result

`ragfilings regress --strategy hybrid_rerank_graph --skip-judge-metrics`
(accuracy-focused; complementary DeepEval metrics skipped because the free
tier is rate-limited — deterministic scoring + G-Eval correctness still ran).

| Category | Baseline (`hybrid_rerank`) | `hybrid_rerank_graph` |
|---|---|---|
| lookup | 68% (15/22) | **100% (22/22)** |
| table | 56% (10/18) | **100% (18/18)** |
| synthesis | 44% (8/18) | **94% (17/18)** |
| unanswerable | 75% (9/12) | **75% (9/12)** — unchanged, safety held |
| ambiguous | 10% (1/10) | 20% (2/10) |
| **overall** | **53.8% (43/80)** | **85.0% (68/80)** |

**25 improved, 0 regressed** (80 common cases). All 16 former incorrect
refusals now answer with the correct, cited figure.

## The 12 questions still wrong, and why

None are regressions — every one was already wrong in the baseline.

**1 synthesis metric-disambiguation**
- `fin-3003` — *"How did Chevron's SG&A expense change FY2024→FY2025?"* The
  model read a combined "Operating, selling, general and administrative" line
  ($32,298M) instead of the standalone SG&A figure the golden expects
  ($5,126M / $4,834M). A row-label ambiguity in the MD&A, not a retrieval
  miss; augmentation does not target it.

**3 pre-existing unanswerable hallucinations** (stage-4 issue #2, untouched)
- `fin-8003` iPhones sold, `fin-8012` Tesla deliveries — the model answers
  from parametric knowledge (unit sales/deliveries are not in the financial
  statements). `fin-8003` is actually refusal-*prose* ("Apple does not
  disclose unit sales…") scored as a hallucination because it is a non-null
  answer.
- `fin-8007` Exxon realized crude price — model fabricates a per-barrel figure.
  These need a stronger refusal prior, not graph data.

**8 ambiguous questions the judge scores down** (stage-4 issue #3)
- `fin-9002…9008, fin-9010` — no fiscal year or a superlative ("Which company
  had the highest net income?"), so the correct behavior is to surface the
  ambiguity. The model instead commits to one interpretation; the judge —
  which has a documented one-directional bias against enumeration-style
  disambiguation (see `judge_calibration_v1.md`) — scores it wrong.
  Augmentation correctly abstains on all of these (no clean scope).

## Honest caveats

- **Accuracy-focused run.** Complementary DeepEval metrics
  (faithfulness / answer-relevancy / contextual-precision) were skipped
  (`--skip-judge-metrics`) because the free provider is rate-limited; they add
  ~3 judge calls per answered case. Accuracy (deterministic numeric matching +
  G-Eval correctness) is unchanged and is the comparable metric. Re-run the
  full metric set when credits allow.
- **Free-model non-determinism.** Refusal formatting varies run-to-run on the
  free model; the up-front augmentation removes the dependence on the model
  choosing to refuse, which is why the result is stable.

## Enterprise multi-hop set (`golden_set_enterprise_v1`)

The v1 set is lookup-heavy, so a 45-case enterprise set was added to prove
multi-hop reasoning: margin/intensity ratios, CAGR, cross-company comparisons,
year-over-year ratio change, and multi-year trends — each answer derived from
2+ fact-graph nodes — plus 8 enterprise unanswerables and 7 ambiguities.
`scripts/build_golden_enterprise_v1.py` derives every answer deterministically
and verifies each base figure against its source chunk (all 45 audit clean).

The augmentation was extended for multi-hop (commit `d6df0ad`):
- **Ratios** (net/operating/gross/FCF margin, R&D intensity): resolve the
  numerator + consolidated-revenue denominator, inject both facts, and ground
  the ratio so the numerical verifier accepts it. Definitional parentheticals
  are stripped first.
- **CAGR**: inject the two endpoint facts and ground the computed growth rate.
- **Comparisons / ratio changes** fall out of multi-ticker / multi-year
  handling; the ratio gap is grounded (full precision + 1-decimal rounding) so
  small deltas verify.
- **Trends** expand "from FYx to FYy" to the full inclusive year range.

| Config (enterprise set, 45 cases) | Accuracy |
|---|---|
| `hybrid_rerank` (no graph, control) | 37.8% (17/45) |
| `hybrid_rerank_graph` | **84.4% (38/45)** |

**+46.6pp, 21 improved / 0 regressed.** By category: synthesis (multi-hop
answerable) 90% (27/30) · unanswerable refusal 100% (8/8) · ambiguous
clarification 43% (3/7). The no-graph control's 37.8% is the point — these
answers are not in any single retrieved chunk, so retrieval-only cannot reach
them; the graph does the joining.

The 7 remaining failures: 3 synthesis (ent-1003/1019 R&D-intensity arithmetic,
ent-1016 PEP gross-margin refusal) and 4 ambiguities (ent-1040/1041/1042/1045)
where the model commits to one reading instead of clarifying (issue #3).

## Reproduce

```bash
uv sync --extra dev
uv run ragfilings regress --strategy hybrid_rerank_graph            # full (with metrics)
uv run ragfilings regress --strategy hybrid_rerank_graph --skip-judge-metrics  # accuracy-only
# enterprise set:
uv run ragfilings regress --golden-set golden/golden_set_enterprise_v1.jsonl --strategy hybrid_rerank_graph --skip-judge-metrics
# diff_report.md in the run dir compares against the latest baseline run.
```
