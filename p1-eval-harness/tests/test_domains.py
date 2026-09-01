"""Unit tests verifying all Project 1 domain datasets and adapters."""

from pathlib import Path
from harness.datasets.schema import load_jsonl
from harness.adapters.legal_adapter import LegalExtractionAdapter
from harness.adapters.biomedical_adapter import BiomedicalQAAdapter
from harness.adapters.support_adapter import CustomerSupportAdapter


def test_domain_b_legal_dataset():
    data_path = Path(__file__).resolve().parents[1] / "data" / "domain_b_legal" / "golden.jsonl"
    assert data_path.exists()
    cases = load_jsonl(str(data_path))
    assert len(cases) >= 20
    for c in cases:
        assert c.domain == "legal"
        assert c.id.startswith("leg-")


def test_domain_c_biomedical_dataset():
    data_path = Path(__file__).resolve().parents[1] / "data" / "domain_c_biomedical" / "golden.jsonl"
    assert data_path.exists()
    cases = load_jsonl(str(data_path))
    assert len(cases) >= 15
    for c in cases:
        assert c.domain == "biomedical"
        assert c.id.startswith("bio-")


def test_domain_d_support_dataset():
    data_path = Path(__file__).resolve().parents[1] / "data" / "domain_d_support" / "golden.jsonl"
    assert data_path.exists()
    cases = load_jsonl(str(data_path))
    assert len(cases) >= 15
    for c in cases:
        assert c.domain == "support"
        assert c.id.startswith("sup-")


def test_legal_adapter_run():
    adapter = LegalExtractionAdapter()
    data_path = Path(__file__).resolve().parents[1] / "data" / "domain_b_legal" / "golden.jsonl"
    cases = load_jsonl(str(data_path))
    trace = adapter.run_case(cases[0])
    assert trace.domain == "legal"
    assert trace.answer is not None
    assert len(trace.steps) == 2


def test_biomedical_adapter_run():
    adapter = BiomedicalQAAdapter()
    data_path = Path(__file__).resolve().parents[1] / "data" / "domain_c_biomedical" / "golden.jsonl"
    cases = load_jsonl(str(data_path))
    trace = adapter.run_case(cases[0])
    assert trace.domain == "biomedical"
    assert trace.answer is not None


def test_support_adapter_run():
    adapter = CustomerSupportAdapter()
    data_path = Path(__file__).resolve().parents[1] / "data" / "domain_d_support" / "golden.jsonl"
    cases = load_jsonl(str(data_path))
    trace = adapter.run_case(cases[0])
    assert trace.domain == "support"
