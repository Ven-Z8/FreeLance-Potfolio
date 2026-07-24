"""Scorecard Report Generator (HTML & Markdown).

Generates standalone, self-contained HTML & Markdown evaluation scorecards
with metrics, per-category breakdown, and full agent trajectory traces.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Eval Harness Reliability Scorecard — {{ summary.strategy.upper() }}</title>
  <style>
    body { font-family: 'Inter', system-ui, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }
    .container { max-width: 1100px; margin: 0 auto; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 24px; margin-bottom: 24px; }
    h1 { margin-top: 0; color: #38bdf8; }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 16px; }
    .stat-card { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 16px; text-align: center; }
    .stat-val { font-size: 28px; font-weight: 700; color: #10b981; }
    .stat-lbl { font-size: 12px; color: #94a3b8; margin-top: 4px; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 14px; }
    th, td { text-align: left; padding: 12px; border-bottom: 1px solid #334155; }
    th { background: #0f172a; color: #94a3b8; }
    .badge-pass { background: #064e3b; color: #34d399; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
    .badge-fail { background: #7f1d1d; color: #f87171; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>🛡️ Agent Eval Harness Scorecard</h1>
      <p>Target Strategy: <strong>{{ summary.strategy.upper() }}</strong> | Evaluated Test Cases: <strong>{{ summary.n }}</strong></p>
      
      <div class="metric-grid">
        <div class="stat-card">
          <div class="stat-val">{{ "%.1f"|format(summary.accuracy * 100) }}%</div>
          <div class="stat-lbl">Overall Accuracy</div>
        </div>
        <div class="stat-card">
          <div class="stat-val" style="color: #38bdf8;">{{ summary.correct_count }} / {{ summary.n }}</div>
          <div class="stat-lbl">Passed Cases</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>📋 Case Trajectory Results</h2>
      <table>
        <thead>
          <tr>
            <th>Case ID</th>
            <th>Outcome</th>
            <th>Query</th>
            <th>Latency</th>
          </tr>
        </thead>
        <tbody>
          {% for r in summary.results %}
          <tr>
            <td><code>{{ r.case_id }}</code></td>
            <td>
              {% if r.correct %}
                <span class="badge-pass">PASS</span>
              {% else %}
                <span class="badge-fail">FAIL</span>
              {% endif %}
            </td>
            <td>{{ r.trace.query }}</td>
            <td>{{ "%.1f"|format(r.trace.latency_ms / 1000) }}s</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def generate_reports(summary: Dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    """Generate HTML and Markdown scorecard reports."""
    from jinja2 import Template

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / f"scorecard_{summary['strategy']}.html"
    md_path = out_dir / f"scorecard_{summary['strategy']}.md"

    # Render HTML
    tmpl = Template(HTML_TEMPLATE)
    html_content = tmpl.render(summary=summary)
    html_path.write_text(html_content, encoding="utf-8")

    # Render Markdown
    md_lines = [
      f"# Eval Harness Scorecard — {summary['strategy'].upper()}",
      "",
      f"*Evaluated {summary['n']} Golden Cases | Accuracy: **{summary['accuracy']:.1%}** ({summary['correct_count']}/{summary['n']})*",
      "",
      "| Case ID | Status | Outcome | Latency |",
      "|---|---|---|---|",
    ]
    for r in summary["results"]:
        status = "✅ PASS" if r["correct"] else "❌ FAIL"
        md_lines.append(f"| `{r['case_id']}` | {status} | {r['outcome']} | {r['trace']['latency_ms']/1000:.1f}s |")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return html_path, md_path
