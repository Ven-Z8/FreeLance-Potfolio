"""Verification: every numerical claim in an answer must exist in its cited chunks.

The #1 failure class on financial tables is a *plausible* wrong number — right
row, wrong column (last year's figure), or wrong row entirely. The generator
cites chunk IDs, so before an answer ships we re-find each money/percent claim
inside the cited text. Unit scaling is normalized ($416.2 billion matches a
table row of 416,161 in millions); years and bare small counts are deliberately
NOT claims — verifying "fiscal 2025" against table text would fail everything.

ponytail: string/number matching, no LLM call. Catches column/row slips, which
are number substitutions; can't catch a wrong number that also appears
somewhere in the cited chunk (e.g. quoting the prior-year column verbatim
WITH its label). The eval's failure analysis measures what slips through.
"""

from __future__ import annotations

import re
from typing import Any

# "$416,161 million", "$416.2 billion", "46.9%", "$1,234.56", "416,161"
_CLAIM_RE = re.compile(
    r"(?:\$\s?[\d,]+(?:\.\d+)?(?:\s?(?:million|billion|thousand|trillion))?"
    r"|[\d,]*\d\.?\d*\s?%"
    r"|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b)"
)
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
# Financial tables state figures in an implicit unit ("in millions"); a claim
# in dollars must match table numbers under any of these relative scalings.
_UNIT_RATIOS = (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)
_REL_TOL = 5e-3  # "$416.2 billion" vs 416,161 (millions) — rounding, not error

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _to_value(raw: str) -> float:
    num = _NUM_RE.search(raw).group().replace(",", "")
    value = float(num)
    for word, mult in _SCALE.items():
        if word in raw.lower():
            value *= mult
    return value


def extract_claims(text: str) -> list[dict[str, Any]]:
    """Money/percent/thousands-separated figures worth verifying."""
    claims = []
    for m in _CLAIM_RE.finditer(text):
        raw = m.group().strip()
        claims.append({"raw": raw, "value": _to_value(raw), "is_pct": raw.endswith("%")})
    return claims


def _chunk_numbers(chunks: list[dict]) -> list[float]:
    out = []
    for c in chunks:
        for m in _NUM_RE.finditer(c["text"]):
            out.append(float(m.group().replace(",", "")))
    return out


def _matches(claim: dict[str, Any], numbers: list[float]) -> bool:
    ratios = (1.0,) if claim["is_pct"] else _UNIT_RATIOS
    for n in numbers:
        for r in ratios:
            target = claim["value"] / r
            if n and abs(n - target) / max(abs(n), abs(target)) <= _REL_TOL:
                return True
            if n == 0 and target == 0:
                return True
    return False


def verify(answer_text: str, cited_chunks: list[dict]) -> dict[str, Any]:
    """Check every claim against the cited chunks. Vacuously verified if none."""
    numbers = _chunk_numbers(cited_chunks)
    claims = [
        {**c, "found": _matches(c, numbers)} for c in extract_claims(answer_text)
    ]
    return {"verified": all(c["found"] for c in claims), "claims": claims}
