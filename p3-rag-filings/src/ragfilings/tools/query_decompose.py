"""Back-compat shim: financial query decomposition moved to
`ragfilings.domains.financial.query_decompose`. Import from there in new code."""

from ..domains.financial.query_decompose import (  # noqa: F401
    _MATH_KEYWORDS,
    decompose_query,
    needs_decomposition,
)
