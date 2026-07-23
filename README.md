# Freelance Portfolio — Agentic AI Engineer

Working repo for the 5-project portfolio. Full spec: `agentic-ai-portfolio-build-spec.md.pdf`.
Progress: `TRACKER.md`.

## Parallel workflow

| Where | What |
|---|---|
| **Claude Code** (you) | All coding. Open a project folder; each has a `CLAUDE.md` (contract) + `BUILD_BRIEF.md` (plan + suggested first prompt). |
| **Cowork** (this session) | Specs, datasets, briefs, tracker, case studies, videos scripts, outreach content. |

## Current state (Week 1)

- `p3-rag-filings/` — START HERE. Brief ready; run `scripts/download_corpus.py` first (downloads ~25 10-Ks from SEC EDGAR; needs your local network).
- `p1-eval-harness/` — Phase 0 only for now: freeze the golden dataset schema so P3 can target it.
- `shared/` — cross-project assets (dataset format docs, diagrams) as they emerge.

## Build order

P3 (wk 1–2) → P1 core + financial pack (wk 3–5) → P2 (wk 6–8) → P5 (wk 9–10) → P4 (wk 11–13).
Outreach starts week 3, not at the end.
