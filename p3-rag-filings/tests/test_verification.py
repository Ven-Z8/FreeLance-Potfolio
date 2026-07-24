"""Verification: numerical claims in an answer must exist in the cited chunks.

This is the wrong-column/wrong-year tripwire — the #1 failure class on
financial tables — so the matcher must handle unit scaling ($416.2 billion vs
a table row in millions) without matching everything (years, list counts).
"""

from ragfilings.verification import extract_claims, verify

CHUNK = {
    "id": "AAPL_2025_10K:Item8:c007",
    "text": "Total net sales | $416,161 | $391,035\nGross margin percentage | 46.9% | 46.2%",
}


def test_extracts_money_percent_and_scaled_figures():
    claims = extract_claims(
        "Net sales were $416,161 million ($416.2 billion), up from $391,035 million; "
        "gross margin was 46.9%. Apple reports three segments as of 2025."
    )
    raws = [c["raw"] for c in claims]
    assert "$416,161 million" in raws
    assert "$416.2 billion" in raws
    assert "46.9%" in raws
    # Years and small counts are not financial claims — verifying them would
    # fail every answer that mentions a fiscal year.
    assert not any("2025" == c["raw"] or "three" in c["raw"] for c in claims)


def test_verifies_exact_and_unit_scaled_matches():
    result = verify("Net sales were $416,161 million, or $416.2 billion.", [CHUNK])
    assert result["verified"]
    assert all(c["found"] for c in result["claims"])


def test_catches_planted_wrong_number():
    result = verify("Net sales were $999,999 million.", [CHUNK])
    assert not result["verified"]
    bad = [c for c in result["claims"] if not c["found"]]
    assert bad and bad[0]["raw"] == "$999,999 million"


def test_percent_must_match_and_no_claims_is_vacuously_verified():
    assert not verify("Gross margin was 99.9%.", [CHUNK])["verified"]
    assert verify("Apple designs consumer electronics.", [CHUNK])["verified"]
