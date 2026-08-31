"""Unit tests for Project 2 Autonomous Multi-Agent Business Workflow."""

import pytest
from workflow.schema import RawLead
from workflow.pipeline import WorkflowPipeline
from workflow.state import DurableStateStore


def test_pipeline_lead_qualification_and_approval():
    store = DurableStateStore()
    pipeline = WorkflowPipeline(store=store)

    lead = RawLead(
        lead_id="lead-test-001",
        name="Sarah Jenkins",
        email="s.jenkins@apexcloud.io",
        company="Apex Cloud Systems",
        domain="apexcloud.io",
        message="Looking for help building an evaluation harness for multi-agent workflows."
    )

    state = pipeline.process_lead(lead)

    assert state.lead_id == "lead-test-001"
    assert state.current_stage == "approval_gate"
    assert state.qualification.qualified is True
    assert state.outreach_draft is not None
    assert state.outreach_draft.draft_status == "pending_approval"
    assert len(state.outreach_draft.grounded_citations) > 0

    # Human approval gate
    executed_state = pipeline.approve_and_execute("lead-test-001")
    assert executed_state.current_stage == "completed"
    assert executed_state.approval_status == "approved"
    assert executed_state.execution_result["status"] == "Outreach Sent"
    assert executed_state.cost_usd > 0.0


def test_pipeline_unqualified_lead():
    store = DurableStateStore()
    pipeline = WorkflowPipeline(store=store)

    lead = RawLead(
        lead_id="lead-test-002",
        name="David Miller",
        email="david@localbakery.com",
        company="Miller Bakery",
        domain="millerbakery.com",
        message="Website design inquiry."
    )

    state = pipeline.process_lead(lead)
    assert state.qualification.qualified is False
    assert state.current_stage == "completed"
    assert state.outreach_draft is None


def test_durable_state_crash_recovery(tmp_path):
    db_file = str(tmp_path / "test_state.db")
    store1 = DurableStateStore(db_file)
    pipeline1 = WorkflowPipeline(store=store1)

    lead = RawLead(
        lead_id="crash-001",
        name="Elena Rostova",
        email="elena@fintechscale.com",
        company="Fintech Scale",
        domain="fintechscale.com",
        message="Need agent evaluation audit."
    )

    state1 = pipeline1.process_lead(lead)
    assert state1.current_stage == "approval_gate"

    # Simulate process crash & restart using fresh store connecting to same SQLite DB
    store2 = DurableStateStore(db_file)
    restored_state = store2.load_state("crash-001")

    assert restored_state is not None
    assert restored_state.lead_id == "crash-001"
    assert restored_state.current_stage == "approval_gate"
    assert restored_state.raw_lead.name == "Elena Rostova"
