"""Unit tests for Multi-Domain Agent Eval Harness Data Loader."""

import pytest
from agenteval import loader


def test_load_all_domains(tmp_path):
    cases = loader.fetch_all_domain_cases(out_dir=tmp_path)
    assert len(cases) == 8

    domains = {c.domain for c in cases}
    assert domains == {"finance", "biomedical", "legal", "healthcare"}

    json_file = tmp_path / "multi_domain_eval.json"
    assert json_file.exists()


def test_domain_schemas():
    fin_cases = loader.load_finance_dataset()
    assert all(c.domain == "finance" for c in fin_cases)

    bio_cases = loader.load_biomedical_dataset()
    assert all(c.domain == "biomedical" for c in bio_cases)

    leg_cases = loader.load_legal_dataset()
    assert all(c.domain == "legal" for c in leg_cases)

    hc_cases = loader.load_healthcare_dataset()
    assert all(c.domain == "healthcare" for c in hc_cases)
