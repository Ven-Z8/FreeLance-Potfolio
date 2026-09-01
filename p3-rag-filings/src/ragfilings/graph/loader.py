"""Back-compat shim: the financial graph loaders moved to
`ragfilings.domains.financial.loader`. Import from there in new code."""

from ..domains.financial.loader import (  # noqa: F401
    graph_path,
    load_graph_engine,
    load_rescue,
)
