"""Domain-Adaptive Agent Evaluation Harness Data Loader.

Ingests and normalizes evaluation datasets across 4 specialized enterprise domains:
  1. Finance (SEC 10-K & FinanceBench)
  2. Biomedical (PubMed QA & BioASQ)
  3. Legal (CUAD Commercial Contracts)
  4. Healthcare (ClinicalTrials.gov & OpenFDA API)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """Standardized multi-domain evaluation case schema."""
    id: str = Field(description="Unique evaluation case identifier.")
    domain: str = Field(description="Target domain: finance, biomedical, legal, healthcare.")
    question: str = Field(description="User or benchmark prompt question.")
    context: Optional[str] = Field(default=None, description="Reference context or document excerpt.")
    ground_truth: str = Field(description="Expected ground truth answer.")
    citations: List[str] = Field(default_factory=list, description="Ground truth document or section IDs.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Domain-specific metadata.")


def load_finance_dataset() -> List[EvalCase]:
    """Load sample Finance Domain evaluation cases (SEC 10-Ks & FinanceBench)."""
    return [
        EvalCase(
            id="fin-001",
            domain="finance",
            question="What was Apple's total net sales in fiscal year 2025?",
            context="Net sales were $416,161 million in 2025 compared to $391,035 million in 2024.",
            ground_truth="$416,161 million",
            citations=["AAPL_2025_10K:Item8"],
            metadata={"source": "FinanceBench", "metric": "headline_revenue"}
        ),
        EvalCase(
            id="fin-002",
            domain="finance",
            question="What is the 3-year gross margin trend for Nvidia from FY2023 to FY2025?",
            context="Gross margin was 56.9% in FY2023, 72.7% in FY2024, and 75.0% in FY2025.",
            ground_truth="Increased from 56.9% in FY2023 to 72.7% in FY2024 and 75.0% in FY2025.",
            citations=["NVDA_2025_10K:Item7"],
            metadata={"source": "FinanceBench", "metric": "margin_trend"}
        ),
    ]


def load_biomedical_dataset() -> List[EvalCase]:
    """Load sample Biomedical Domain evaluation cases (PubMed QA & BioASQ)."""
    return [
        EvalCase(
            id="bio-001",
            domain="biomedical",
            question="Does GLP-1 receptor agonist therapy reduce cardiovascular events in patients with type 2 diabetes?",
            context="Clinical trials demonstrate that GLP-1 receptor agonists significantly reduce major adverse cardiovascular events (MACE) by ~14% in type 2 diabetes patients.",
            ground_truth="Yes, GLP-1 receptor agonists reduce major adverse cardiovascular events by approximately 14%.",
            citations=["PMID_34567890"],
            metadata={"source": "PubMedQA", "mesh_terms": ["GLP-1", "Cardiovascular", "Diabetes"]}
        ),
        EvalCase(
            id="bio-002",
            domain="biomedical",
            question="What is the primary mechanism of action of Pembrolizumab?",
            context="Pembrolizumab is a humanized monoclonal antibody that blocks the interaction between PD-1 and its ligands, PD-L1 and PD-L2, reactivating T-cell anti-tumor immune response.",
            ground_truth="Blocks PD-1 receptor interaction with PD-L1/PD-L2 to reactivate anti-tumor T-cell immunity.",
            citations=["PMID_28910234"],
            metadata={"source": "BioASQ", "target": "PD-1"}
        ),
    ]


def load_legal_dataset() -> List[EvalCase]:
    """Load sample Legal Domain evaluation cases (CUAD Commercial Contracts)."""
    return [
        EvalCase(
            id="leg-001",
            domain="legal",
            question="What is the notice period required for termination without cause in this agreement?",
            context="Either party may terminate this Agreement without cause upon providing at least sixty (60) days prior written notice to the other party.",
            ground_truth="60 days prior written notice.",
            citations=["CUAD_Contract_Sec12.2"],
            metadata={"source": "CUAD", "clause_type": "Termination_Notice"}
        ),
        EvalCase(
            id="leg-002",
            domain="legal",
            question="Does the agreement contain an uncapped indemnification obligation for IP infringement?",
            context="Section 9.1: Supplier shall indemnify Customer against all third-party IP claims. Section 9.4: The liability caps in Section 8 shall not apply to indemnification under Section 9.1.",
            ground_truth="Yes, indemnification for IP infringement is explicitly excluded from the limitation of liability cap.",
            citations=["CUAD_Contract_Sec9.4"],
            metadata={"source": "CUAD", "clause_type": "Indemnification_Cap"}
        ),
    ]


def load_healthcare_dataset() -> List[EvalCase]:
    """Load sample Healthcare & Clinical Operations cases (ClinicalTrials.gov & OpenFDA API)."""
    return [
        EvalCase(
            id="hc-001",
            domain="healthcare",
            question="What is the primary endpoint for Clinical Trial NCT04567890?",
            context="Official Title: Phase III Study of Oncology Drug X. Primary Outcome Measure: Overall Survival (OS) evaluated at 24 months post-randomization.",
            ground_truth="Overall Survival (OS) at 24 months.",
            citations=["NCT04567890"],
            metadata={"source": "ClinicalTrials.gov", "phase": "Phase 3"}
        ),
        EvalCase(
            id="hc-002",
            domain="healthcare",
            question="What black box warning is listed on the FDA label for Drug Y?",
            context="WARNING: RISK OF SERIOUS HEPATOTOXICITY. Drug Y can cause severe, life-threatening liver injury. Monitor serum transaminases baseline and monthly.",
            ground_truth="Risk of serious hepatotoxicity / life-threatening liver injury requiring monthly liver enzyme monitoring.",
            citations=["OpenFDA_NDC_12345"],
            metadata={"source": "OpenFDA", "warning_type": "BlackBox"}
        ),
    ]


def fetch_all_domain_cases(out_dir: Optional[Path] = None) -> List[EvalCase]:
    """Aggregate cases from all 4 enterprise domains and save as JSON dataset."""
    all_cases = []
    all_cases.extend(load_finance_dataset())
    all_cases.extend(load_biomedical_dataset())
    all_cases.extend(load_legal_dataset())
    all_cases.extend(load_healthcare_dataset())

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "multi_domain_eval.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in all_cases], f, indent=2)

    return all_cases
