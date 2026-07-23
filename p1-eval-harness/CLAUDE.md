# P1 — Domain-Adaptive Agent Eval Harness ("Proving Ground")

Read BUILD_BRIEF.md before writing any code. This is the portfolio centerpiece
and open-source play. Positioning: *"Plug in any agent, any domain — get a
reliability scorecard in an afternoon."*

## What this project is

A framework that takes (a) an agent system, (b) a domain-specific golden
dataset, (c) a metric configuration — and produces a full evaluation report
with regression tracking.

## Non-negotiable constraints (scope discipline)

- NO web UI beyond the generated HTML report. CLI + config files is more credible.
- NO support for every framework. Adapters for exactly two: raw API calls + one
  popular agent framework — with a documented adapter interface for others.
- NO hosted SaaS version. Resist completely.
- LLM-as-judge MUST be calibrated: hand-label 50 cases, measure judge agreement,
  publish the agreement number even if it's mediocre. Honesty sells.

## Tech conventions

- Python 3.11+, uv, ruff. Library-first design: `pip install`able, CLI on top.
- Golden dataset format: JSONL, one test case per line
  (input, expected outcome, acceptable-variation rules, difficulty tag, failure-category tag).
- Traces: full agent runs stored as structured JSON (every tool call, reasoning
  step, retry) — replayable and diffable.
- Config-driven metric selection; metrics are pluggable classes.

## Architecture (core engine, weeks 1–2 of P1 scope)

1. **Golden dataset toolkit** — schema + semi-automated dataset builder
   (raw domain docs in → drafted candidate test cases → human approves/edits).
2. **Trajectory capture** — record full runs as structured JSON traces.
3. **Metric layer, three tiers:**
   - Deterministic: exact match, schema validation, tool-call correctness, constraint satisfaction.
   - Statistical/retrieval: precision/recall vs golden citations, latency p50/p95, cost per task.
   - LLM-as-judge: faithfulness/helpfulness/tone, with calibration workflow built in.
4. **Regression runner** — one command re-runs the suite vs a new agent version,
   produces diff report per failure category, stores history for trend charts.
5. **Report generator** — HTML/markdown scorecard: pass rate, per-category
   breakdown, cost/latency, worst 10 failures with full trajectories, trend chart.

## Definition of done

- A stranger clones the repo, runs one command, reproduces the legal-domain eval report.
- README shows the scorecard image in the first screen-scroll.
- One deep-dive post published: "I evaluated the same agent architecture across
  3 domains. Here's where it silently failed."
- Calibrated judge agreement number is published.

## Dependency

P3 (../p3-rag-filings) produces the golden set for the financial domain pack.
Design the dataset format FIRST and share it with P3 so its golden set drops in
unchanged.
