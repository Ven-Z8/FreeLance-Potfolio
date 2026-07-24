"""CLI Interface for Domain-Adaptive Agent Eval Harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from harness.adapters.ragfilings_adapter import RAGFilingsAdapter
from harness.report import generate_reports
from harness.runner import run_evaluation_suite


def main():
    parser = argparse.ArgumentParser(description="Domain-Adaptive Agent Eval Harness CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run evaluation suite on target domain")
    run_parser.add_argument("--domain", default="financial", choices=["financial", "biomedical", "legal", "healthcare"])
    run_parser.add_argument("--strategy", default="hybrid_rerank", choices=["dense", "hybrid", "hybrid_rerank", "agent_react"])
    run_parser.add_argument("--out", default="reports")
    run_parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.command == "run":
        if args.domain == "financial":
            golden_path = Path(__file__).resolve().parents[2] / "data" / "domain_a_financial" / "all_financial_golden.jsonl"
            adapter = RAGFilingsAdapter()
            summary = run_evaluation_suite(
                golden_path=golden_path,
                adapter=adapter,
                strategy=args.strategy,
                out_dir=args.out,
                limit=args.limit
            )
            html_p, md_p = generate_reports(summary, args.out)
            print(f"\n✨ Generated Scorecard Reports:\n   HTML: {html_p}\n   MD:   {md_p}")


if __name__ == "__main__":
    main()
