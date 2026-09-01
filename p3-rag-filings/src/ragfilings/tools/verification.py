"""Back-compat shim: financial claim verification moved to
`ragfilings.domains.financial.verification`. Import from there in new code."""

from ..domains.financial.verification import (  # noqa: F401
    _CLAIM_RE,
    _SCALE,
    _matches,
    _to_value,
    extract_claims,
    verify,
)
