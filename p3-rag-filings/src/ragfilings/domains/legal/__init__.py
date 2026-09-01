"""Legal domain pack — commercial contracts (CUAD corpus, CC-BY-4.0).

Corpus: 102 commercial agreements from the CUAD test split (The Atticus
Project), indexed with the same retrieval engine as the financial pack.
Fact layer: deterministic defined-term extraction. Claim semantics: quoted
language must exist verbatim in the cited excerpts, plus money/date claims.
"""

from .pack import LegalPack

PACK = LegalPack()

__all__ = ["PACK", "LegalPack"]
