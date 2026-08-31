"""Scorecard generation: reports/scorecard.md + reports/scorecard.png."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

CAVEAT = (
    "Golden set v1 (2026-08-30): 80 cases rebuilt from scratch on the stage-2 "
    "fact graph; every expected answer verified against filing text "
    "(see golden/audit_v1.json). Judge = DeepEval G-Eval over OpenRouter; "
    "judge cost below is the eval overhead, separate from per-query generation cost."
)

_PCT_METRICS = [
    ("accuracy", "Answer accuracy"),
    ("retrieval_hit_rate", "Retrieval hit rate"),
    ("citation_faithfulness", "Citation faithfulness"),
    ("hallucination_rate", "Hallucination rate (unanswerable)"),
    ("refusal_correctness", "Refusal correctness"),
    ("deepeval_faithfulness", "DeepEval faithfulness"),
    ("deepeval_answer_relevancy", "DeepEval answer relevancy"),
    ("deepeval_contextual_precision", "DeepEval contextual precision"),
]


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.0%}"


def _row(metrics: dict[str, Any], key: str) -> str:
    return _pct(metrics.get(key))


def write_scorecard(
    all_results: dict[str, dict[str, Any]],
    out_dir: str | Path,
    n_filings: int = 25,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    strategies = list(all_results)
    md_path = out_dir / "scorecard.md"
    png_path = out_dir / "scorecard.png"

    first = all_results[strategies[0]]["metrics"]
    lines = [
        "# Scorecard — RAG over SEC 10-K filings",
        "",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M')} · {first['n']} golden "
        f"questions · {n_filings} filings · generation model "
        f"`{first['model']}` · judge `{first['judge_model']}`*",
        "",
        f"> {CAVEAT}",
        "",
        "| Metric | " + " | ".join(s.capitalize() for s in strategies) + " |",
        "|---|" + "---|" * len(strategies),
    ]
    for key, label in _PCT_METRICS:
        cells = " | ".join(_row(all_results[s]["metrics"], key) for s in strategies)
        lines.append(f"| {label} | {cells} |")
    for key, label, fmt in [
        ("refusal_rate", "Refusal rate", _pct),
        ("cost_per_query_usd", "Cost / query", lambda v: "—" if v is None else f"${v:.4f}"),
        ("latency_p50_ms", "Latency p50", lambda v: "—" if v is None else f"{v/1000:.1f}s"),
        ("latency_p95_ms", "Latency p95", lambda v: "—" if v is None else f"{v/1000:.1f}s"),
        ("judge_cost_usd", "Judge cost (eval overhead, total)", lambda v: "—" if v is None else f"${v:.3f}"),
    ]:
        cells = " | ".join(fmt(all_results[s]["metrics"].get(key)) for s in strategies)
        lines.append(f"| {label} | {cells} |")

    lines += ["", "## Accuracy by question category", ""]
    cats = sorted({c for s in strategies for c in all_results[s]["metrics"]["by_category"]})
    lines.append("| Category | " + " | ".join(f"{s.capitalize()} (n)" for s in strategies) + " |")
    lines.append("|---|" + "---|" * len(strategies))
    for cat in cats:
        cells = []
        for s in strategies:
            c = all_results[s]["metrics"]["by_category"].get(cat)
            cells.append(f"{c['accuracy']:.0%} ({c['n']})" if c else "—")
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_png(all_results, png_path)
    return md_path, png_path


def _write_png(all_results: dict[str, dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    strategies = list(all_results)
    labels = [label for _, label in _PCT_METRICS]
    fig, (ax, side) = plt.subplots(1, 2, figsize=(12, 5.0), width_ratios=[2.5, 1.0], dpi=150)
    x = np.arange(len(labels))
    width = 0.8 / max(len(strategies), 1)
    colors = ["#3b82f6", "#06b6d4", "#10b981", "#8b5cf6"]
    for i, s in enumerate(strategies):
        m = all_results[s]["metrics"]
        vals = [m.get(k) or 0.0 for k, _ in _PCT_METRICS]
        bars = ax.bar(x + i * width, vals, width, label=s, color=colors[i % len(colors)])
        ax.bar_label(bars, fmt="{:.0%}", fontsize=7.5, padding=2)
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels([lb.replace(" (unanswerable)", "\n(unanswerable)") for lb in labels], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    first = all_results[strategies[0]]["metrics"]
    ax.set_title(
        f"Strategy Comparison — {first['n']} Questions",
        fontsize=11,
        loc="left",
        fontweight="bold",
    )

    side.axis("off")
    txt = []
    for s in strategies:
        m = all_results[s]["metrics"]
        txt.append(
            f"{s.upper()}\n"
            f"  cost/query  ${(m.get('cost_per_query_usd') or 0):.4f}\n"
            f"  p50 latency {(m.get('latency_p50_ms') or 0)/1000:.1f}s\n"
            f"  p95 latency {(m.get('latency_p95_ms') or 0)/1000:.1f}s"
        )
    txt.append(f"model: {first['model']}")
    side.text(0.0, 0.95, "\n\n".join(txt), va="top", ha="left", fontsize=8.5, family="monospace", transform=side.transAxes)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
