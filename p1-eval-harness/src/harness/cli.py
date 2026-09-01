"""CLI for the Domain-Adaptive Agent Eval Harness.

    eval-harness run --strategy hybrid_rerank_graph --skip-judge-metrics
    eval-harness run --golden-set data/domain_a_financial --limit 5
    eval-harness diff reports/evals/<baseline-dir> reports/evals/<current-dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from harness import config as cfg_mod

P1_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES = ["dense", "hybrid", "hybrid_rerank", "agent_react",
              "dense_graph", "hybrid_graph", "hybrid_rerank_graph"]


def _adapter_for(domain: str, ragfilings_config: str | None):
    if domain == "financial":
        from harness.adapters.ragfilings_adapter import RAGFilingsAdapter
        return RAGFilingsAdapter(config_path=ragfilings_config)
    raise SystemExit(
        f"no adapter wired for domain {domain!r} yet — the v1 harness runs the "
        "financial domain against p3-rag-filings"
    )


def _cmd_run(args: argparse.Namespace) -> None:
    from harness.regress import run_regression
    from harness.report import generate_reports, summary_from_rows, write_scorecard

    cfg = cfg_mod.load(args.config)
    adapter = _adapter_for(args.domain, args.ragfilings_config)
    run_dir, diff, all_results = run_regression(
        cfg,
        adapter,
        args.golden_set,
        args.strategy,
        out_root=args.out_root,
        limit=args.limit,
        baseline=args.baseline,
        skip_judge_metrics=args.skip_judge_metrics,
    )

    md, png = write_scorecard(all_results, run_dir)
    rows = all_results[args.strategy]["rows"]
    html, _ = generate_reports(summary_from_rows(args.strategy, rows), run_dir)

    print(f"\nrun: {run_dir}")
    print(f"scorecard: {md}\n           {png}\n           {html}")
    if diff:
        print(f"diff vs {diff['baseline']}: "
              f"improved {len(diff['improved'])}, regressed {len(diff['regressed'])} "
              f"({diff['common_cases']} common cases)")
        if diff["regressed"]:
            print("regressed cases:", ", ".join(diff["regressed"]))
        print(f"report: {run_dir / 'diff_report.md'}")
    else:
        print("no baseline run found — this run becomes the baseline")


def _cmd_diff(args: argparse.Namespace) -> None:
    from harness.regress import diff_runs, write_diff_report

    base, new = Path(args.baseline), Path(args.current)
    if not base.exists() or not new.exists():
        raise SystemExit("both run dirs must exist")
    diff = diff_runs(base, new)
    out = Path(args.out) if args.out else new / "diff_report.md"
    write_diff_report(diff, out)
    print(f"improved {len(diff['improved'])}, regressed {len(diff['regressed'])} "
          f"({diff['common_cases']} common cases) -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(prog="eval-harness")
    p.add_argument("--config", default=None, help="path to harness config.toml")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the golden suite + scorecard + regression diff")
    run.add_argument("--domain", default="financial",
                     choices=["financial", "legal", "biomedical", "support"])
    run.add_argument("--strategy", default="hybrid_rerank", choices=STRATEGIES,
                     help="*_graph adds deterministic fact-graph augmentation")
    run.add_argument("--golden-set",
                     default=str(P1_ROOT / "data" / "domain_a_financial" / "golden_set_v1.jsonl"),
                     help="golden JSONL file, or a directory of golden_set_*.jsonl")
    run.add_argument("--ragfilings-config", default=None,
                     help="override the target system's config.toml")
    run.add_argument("--limit", type=int, default=None, help="only the first N cases (smoke runs)")
    run.add_argument("--baseline", default=None,
                     help="run dir name under --out-root (default: latest existing run)")
    run.add_argument("--out-root", default="reports/evals")
    run.add_argument("--skip-judge-metrics", action="store_true",
                     help="skip complementary DeepEval metrics (faithfulness/relevancy/"
                          "contextual-precision) for a faster accuracy-focused run")
    run.set_defaults(func=_cmd_run)

    diff = sub.add_parser("diff", help="compare two existing run dirs")
    diff.add_argument("baseline")
    diff.add_argument("current")
    diff.add_argument("--out", default=None)
    diff.set_defaults(func=_cmd_diff)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
