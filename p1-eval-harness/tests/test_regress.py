"""Regression runner: run-dir discovery, diff logic, report output (no API)."""

from __future__ import annotations

import json

from harness import regress


def _write_run(root, name, rows):
    d = root / name
    d.mkdir(parents=True)
    with (d / "results_hybrid_rerank.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return d


def _row(cid, correct, category="lookup", cost=0.01):
    return {"case_id": cid, "correct": correct, "category": category,
            "cost_usd": cost, "latency_ms": 100.0}


def test_diff_detects_improvements_and_regressions(tmp_path):
    base = _write_run(tmp_path, "run-a", [
        _row("fin-1", True), _row("fin-2", False), _row("fin-3", True),
    ])
    new = _write_run(tmp_path, "run-b", [
        _row("fin-1", False), _row("fin-2", True), _row("fin-3", True),
    ])
    diff = regress.diff_runs(base, new)
    assert diff["improved"] == ["fin-2"]
    assert diff["regressed"] == ["fin-1"]
    assert diff["common_cases"] == 3
    assert diff["baseline_view"]["accuracy"] == 2 / 3
    assert diff["current_view"]["accuracy"] == 2 / 3


def test_diff_only_compares_common_cases(tmp_path):
    base = _write_run(tmp_path, "run-a", [_row("fin-1", True), _row("fin-9", True)])
    new = _write_run(tmp_path, "run-b", [_row("fin-1", True), _row("fin-2", False)])
    diff = regress.diff_runs(base, new)
    assert diff["common_cases"] == 1
    assert diff["improved"] == [] and diff["regressed"] == []


def test_latest_run_picks_highest_sorted_dir(tmp_path):
    _write_run(tmp_path, "20260101-000000-abc-hybrid_rerank", [_row("x", True)])
    later = _write_run(tmp_path, "20260202-000000-def-hybrid_rerank", [_row("x", True)])
    assert regress.latest_run(tmp_path) == later
    assert regress.latest_run(tmp_path / "missing") is None


def test_write_diff_report_contains_both_views(tmp_path):
    base = _write_run(tmp_path, "run-a", [_row("fin-1", True, category="lookup")])
    new = _write_run(tmp_path, "run-b", [_row("fin-1", False, category="lookup")])
    diff = regress.diff_runs(base, new)
    out = tmp_path / "diff_report.md"
    regress.write_diff_report(diff, out)
    text = out.read_text()
    assert "run-a" in text and "run-b" in text
    assert "regressed: 1" in text
    assert "lookup" in text


def test_load_rows_reads_all_results_files(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "results_dense.jsonl").write_text(json.dumps(_row("a", True)) + "\n")
    (d / "results_hybrid.jsonl").write_text(json.dumps(_row("b", False)) + "\n")
    rows = regress.load_rows(d)
    assert {r["case_id"] for r in rows} == {"a", "b"}
