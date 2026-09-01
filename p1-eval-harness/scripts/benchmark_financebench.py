"""FinanceBench reasoning-over-evidence benchmark adapter.

Runs the P3 grounded-synthesis pipeline on FinanceBench's 150 open questions,
giving each question its official evidence text as the retrieved context
(reasoning-over-evidence: exercises grounding + verification + financial math,
not retrieval, since FinanceBench references filings outside our corpus).
Scores our answer against the official FinanceBench answer with the calibrated
G-Eval judge.

Usage: uv run python scripts/benchmark_financebench.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

P1 = Path(__file__).resolve().parent.parent
P3 = P1.parent / "p3-rag-filings"
sys.path.insert(0, str(P1 / "src"))
sys.path.insert(0, str(P3 / "src"))

from harness import config as harness_cfg                # noqa: E402
from harness.judge import DeepEvalScorer                 # noqa: E402
from ragfilings.config import load as load_cfg           # noqa: E402
from ragfilings.pipeline.engine import answer            # noqa: E402

FB = P1 / "data" / "financebench" / "financebench_merged.jsonl"
FB_URL = ("https://huggingface.co/datasets/PatronusAI/financebench/"
          "resolve/main/financebench_merged.jsonl")


def load_financebench() -> list[dict]:
    if not FB.exists():
        FB.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading FinanceBench -> {FB}")
        try:
            urllib.request.urlretrieve(FB_URL, FB)
        except Exception as e:
            raise SystemExit(
                f"could not download FinanceBench ({e}); save "
                f"financebench_merged.jsonl to {FB} manually "
                "(HuggingFace: PatronusAI/financebench, CC-BY-NC-4.0)"
            )
    return [json.loads(line) for line in FB.open() if line.strip()]


def evidence_hits(rec: dict) -> list[dict]:
    """Build pseudo retrieval hits from the official evidence text."""
    hits = []
    doc = rec.get("doc_name", "unknown")
    for i, ev in enumerate(rec.get("evidence", [])):
        text = ev.get("evidence_text") or ev.get("evidence_text_full_page") or ""
        if not text.strip():
            continue
        hits.append({
            "chunk": {
                "id": f"{doc}:evidence:{i}",
                "text": text,
                "section": "evidence",
                "ticker": None,
            },
            "score": 0.9,
            "dense_sim": 0.9,
        })
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_cfg(str(P3 / "config.toml"))
    # The synthesis pipeline reads p3's config; the judge reads the harness's.
    cfg["judge"] = harness_cfg.load().get("judge", {})
    recs = load_financebench()
    if args.limit:
        recs = recs[: args.limit]

    scorer = DeepEvalScorer(cfg)

    out_path = Path(args.out) if args.out else (
        P1 / "reports" / "financebench" /
        f"fb_{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_correct = 0
    with out_path.open("w", encoding="utf-8") as out:
        for i, rec in enumerate(recs):
            qid = rec["financebench_id"]
            question = rec["question"]
            official = rec["answer"]
            hits = evidence_hits(rec)
            if not hits:
                print(f"[{i+1}/{len(recs)}] {qid}: no evidence, skip")
                continue
            t0 = time.time()
            try:
                # graph_rescue=None: the graph is built from OUR 25 filings, not
                # FinanceBench's, so disable graph augmentation/clarification here.
                res = answer(question, hits, cfg, graph_rescue=None)
            except Exception as e:  # noqa: BLE001
                print(f"[{i+1}/{len(recs)}] {qid}: ERROR {e}")
                continue
            dt = time.time() - t0

            our_answer = res.get("answer")
            if res.get("refused") or not our_answer:
                our_answer = res.get("refusal_reason") or ""

            # Score our answer against the official answer with the judge.
            case = {
                "id": qid,
                "input": question,
                "expected": {"answer": official, "type": "judge"},
            }
            result = {"answer": our_answer, "refused": res.get("refused", False),
                      "citations": res.get("citations", [])}
            try:
                scored = scorer.correctness(case, result)
                correct = bool(scored.get("correct"))
            except Exception as e:  # noqa: BLE001
                print(f"[{i+1}/{len(recs)}] {qid}: judge ERROR {e}")
                correct = False

            n_correct += int(correct)
            print(f"[{i+1}/{len(recs)}] {qid}: "
                  f"{'CORRECT' if correct else 'WRONG'} ({dt:.1f}s)")
            out.write(json.dumps({
                "id": qid,
                "question": question,
                "official_answer": official,
                "our_answer": our_answer,
                "refused": res.get("refused", False),
                "correct": correct,
                "citations": res.get("citations", []),
                "latency_s": dt,
            }, ensure_ascii=False) + "\n")
            out.flush()

    total = len(recs)
    print(f"\nFinanceBench reasoning-over-evidence: {n_correct}/{total} "
          f"= {n_correct / total:.1%}")
    print(f"results -> {out_path}")


if __name__ == "__main__":
    main()
