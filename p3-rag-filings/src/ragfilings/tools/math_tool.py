"""Back-compat shim: the financial math tool moved to
`ragfilings.domains.financial.math_tool`. Import from there in new code."""

from ..domains.financial.math_tool import compute_financial_math, safe_eval  # noqa: F401
