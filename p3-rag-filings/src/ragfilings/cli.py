"""CLI entry point.

    ragfilings parse corpus/AAPL_2025-10-31_10K.htm   # section tree of one filing
    ragfilings index                                   # parse + chunk + embed corpus
    ragfilings ask "What was Apple's FY2025 net sales?" [--strategy hybrid]
    ragfilings eval golden/ --strategy both [--limit 5]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import config as cfg_mod
from . import ingestion


def _cmd_parse(args: argparse.Namespace) -> None:
    cfg = cfg_mod.load(args.config)
    sections = ingestion.parse_file(args.filing,
                                    cfg["ingestion"]["min_section_chars"],
                                    cfg["ingestion"]["pointer_chars"])
    print(f"{Path(args.filing).name}: {len(sections)} sections\n")
    print(ingestion.render_tree(sections))


def _cmd_index(args: argparse.Namespace) -> None:
    """Parse every filing in the manifest, chunk, and build the embedding index."""
    from . import chunking, retrieval  # lazy: pulls in torch

    cfg = cfg_mod.load(args.config)
    root = cfg_mod.ROOT
    ing = cfg["ingestion"]
    chunks_dir = root / "corpus" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    with (root / cfg["corpus"]["manifest"]).open() as f:
        for m in csv.DictReader(f):
            path = root / "corpus" / m["local_file"]
            if not path.exists():
                print(f"skip {m['ticker']}: {path.name} missing "
                      "(run scripts/download_corpus.py)", file=sys.stderr)
                continue
            sections = ingestion.parse_file(path, ing["min_section_chars"],
                                            ing["pointer_chars"])
            chunks = chunking.chunk_sections(sections, m,
                                             cfg["chunking"]["max_chars"])
            with (chunks_dir / f"{m['ticker']}_chunks.jsonl").open(
                    "w", encoding="utf-8") as out:
                for c in chunks:
                    out.write(json.dumps(c, ensure_ascii=False) + "\n")
            all_chunks.extend(chunks)
            print(f"{m['ticker']:<6} {len(chunks):>5} chunks")
    print(f"\nembedding {len(all_chunks)} chunks with "
          f"{cfg['embedding']['model']} (first run downloads the model)...")
    retrieval.build_index(all_chunks, root / cfg["embedding"]["index_dir"],
                          cfg["embedding"]["model"])
    print(f"index written to {cfg['embedding']['index_dir']}")


def _cmd_ask(args: argparse.Namespace) -> None:
    from . import generation

    cfg = cfg_mod.load(args.config)
    result = generation.ask(args.question, cfg, strategy=args.strategy)
    print(f"strategy={result['strategy']} confidence={result['confidence']:.3f} "
          f"latency={result['latency_ms']/1000:.1f}s "
          f"cost=${result['usage']['cost_usd']:.4f}\n")
    if result["refused"]:
        print(f"REFUSED: {result['refusal_reason']}")
        return
    print(result["answer"])
    print("\ncitations:")
    for c in result["citations"]:
        print(f"  {c}")
    ver = result["verification"]
    if not ver["verified"]:
        failed = ", ".join(c["raw"] for c in ver["claims"] if not c["found"])
        print(f"\n⚠️ verification FAILED for: {failed}")


def _cmd_eval(args: argparse.Namespace) -> None:
    from . import evaluation, report

    cfg = cfg_mod.load(args.config)
    if args.strategy == "both":
        strategies = ["dense", "hybrid"]
    elif args.strategy == "all":
        strategies = ["dense", "hybrid", "hybrid_rerank", "agent_react"]

    else:
        strategies = [args.strategy]
    results = evaluation.run_eval(cfg, args.golden_dir, strategies,
                                  out_dir=args.out, limit=args.limit)
    md, png = report.write_scorecard(results, args.out)
    print(f"\nscorecard: {md}\n           {png}")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    from . import evaluation, report

    cfg = cfg_mod.load(args.config)
    print(f"Running Industry Benchmark Suite ({args.dataset})...")
    
    dataset_map = {
        "financebench": Path("golden/benchmarks/financebench.jsonl"),
        "ragas": Path("golden/benchmarks/ragas_suite.jsonl"),
        "rgb": Path("golden/benchmarks/rgb_benchmark.jsonl"),
        "finqa": Path("golden/benchmarks/finqa_tables.jsonl"),
    }

    if args.dataset == "all":
        target_dir = Path("golden")
    else:
        target_dir = dataset_map.get(args.dataset, Path("golden"))

    results = evaluation.run_eval(cfg, target_dir, ["hybrid_rerank"],
                                  out_dir="reports/benchmark", limit=args.limit)
    md, png = report.write_scorecard(results, "reports/benchmark")
    print(f"\nIndustry Benchmark Rating Scorecard: {md}\n                                   {png}")


def main() -> None:
    p = argparse.ArgumentParser(prog="ragfilings")
    p.add_argument("--config", default=None, help="path to config.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    parse = sub.add_parser("parse", help="parse one filing, print its section tree")
    parse.add_argument("filing", help="path to a 10-K .htm file")
    parse.set_defaults(func=_cmd_parse)

    index = sub.add_parser("index", help="parse + chunk + embed the whole corpus")
    index.set_defaults(func=_cmd_index)

    ask = sub.add_parser("ask", help="answer one question with citations")
    ask.add_argument("question")
    ask.add_argument("--strategy", choices=["dense", "hybrid", "hybrid_rerank", "agent_react"], default=None,
                     help="override config [retrieval] strategy")
    ask.set_defaults(func=_cmd_ask)

    ev = sub.add_parser("eval", help="run the golden-set eval + scorecard")
    ev.add_argument("golden_dir", nargs="?", default="golden")
    ev.add_argument("--strategy", choices=["dense", "hybrid", "hybrid_rerank", "agent_react", "both", "all"],
                    default="all")

    ev.add_argument("--limit", type=int, default=None,
                    help="only the first N cases (smoke runs)")
    ev.add_argument("--out", default="reports")
    ev.set_defaults(func=_cmd_eval)

    bm = sub.add_parser("benchmark", help="run industry standard RAG benchmarks")
    bm.add_argument("--dataset", choices=["financebench", "ragas", "rgb", "finqa", "all"], default="all")
    bm.add_argument("--limit", type=int, default=None)
    bm.set_defaults(func=_cmd_benchmark)

    args = p.parse_args()
    args.func(args)




if __name__ == "__main__":
    main()
