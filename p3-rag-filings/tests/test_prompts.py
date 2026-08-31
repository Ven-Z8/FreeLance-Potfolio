"""Tests for Centralized Prompt Registry."""

import pytest

from ragfilings.prompts import PromptRegistry, load_prompt


def test_prompt_registry_loads_synthesis_template():
    prompt = PromptRegistry.get_system_synthesis()
    assert "SEC 10-K" in prompt
    assert "[brackets]" in prompt


def test_prompt_registry_formats_verification_retry():
    retry = PromptRegistry.get_verification_retry(["$999M", "50%"])
    assert "$999M, 50%" in retry
    assert "Verification failed" in retry


def test_prompt_registry_loads_math_and_decompose():
    math_p = PromptRegistry.get_math_tool()
    assert "mathematical expression" in math_p

    dec_p = PromptRegistry.get_query_decompose()
    assert "sub_query" in dec_p


def test_prompt_registry_loads_all_specialized_templates():
    solo_p = PromptRegistry.get_solo_meta_orchestrator()
    assert "Solo Meta Orchestrator" in solo_p
    assert "search_sec_filings" in solo_p

    lead_p = PromptRegistry.get_lead_orchestrator()
    assert "Lead Orchestrator" in lead_p

    doc_p = PromptRegistry.get_document_analyst()
    assert "Document Understanding Analyst" in doc_p

    res_p = PromptRegistry.get_researcher()
    assert "Financial Research Agent" in res_p

    data_p = PromptRegistry.get_data_analyst()
    assert "Financial Data Analyst" in data_p

    synth_p = PromptRegistry.get_synthesis_expert()
    assert "Executive Synthesis" in synth_p

    audit_p = PromptRegistry.get_auditor_guardrail()
    assert "Auditor" in audit_p


def test_load_prompt_convenience_function():
    prompt = load_prompt("synthesis")
    assert "JSON object" in prompt


def test_missing_prompt_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        PromptRegistry.get_raw("non_existent_prompt_template_xyz")
