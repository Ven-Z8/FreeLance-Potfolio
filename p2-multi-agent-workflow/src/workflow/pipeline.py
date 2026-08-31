"""Pipeline orchestrator for Project 2 Autonomous Business Workflow."""

from __future__ import annotations

from typing import Optional, List
from workflow.schema import RawLead, LeadState
from workflow.state import DurableStateStore
from workflow.agents import (
    IntakeAgent, ResearchAgent, QualificationAgent, OutreachDraftingAgent, ExecutionAgent
)


class WorkflowPipeline:
    """Orchestrates end-to-end lead qualification, research, drafting, and approval execution."""

    def __init__(self, store: Optional[DurableStateStore] = None):
        self.store = store or DurableStateStore()
        self.intake_agent = IntakeAgent()
        self.research_agent = ResearchAgent()
        self.qual_agent = QualificationAgent()
        self.draft_agent = OutreachDraftingAgent()
        self.exec_agent = ExecutionAgent()

    def process_lead(self, raw: RawLead) -> LeadState:
        """Run lead through automatic stages up to approval gate."""
        state = self.intake_agent.run(raw)
        self.store.save_state(state)

        # Stage 2: Research
        state = self.research_agent.run(state)
        self.store.save_state(state)

        # Stage 3: Qualification
        state = self.qual_agent.run(state)
        self.store.save_state(state)

        # Stage 4: Outreach Drafting (if qualified)
        if state.qualification and state.qualification.qualified:
            state = self.draft_agent.run(state)
            self.store.save_state(state)

        return state

    def approve_and_execute(self, lead_id: str, approved_by: str = "human_operator") -> LeadState:
        """Human approval gate trigger: approves and executes outreach."""
        state = self.store.load_state(lead_id)
        if not state:
            raise ValueError(f"Lead ID {lead_id} not found in state store.")

        if state.current_stage != "approval_gate":
            raise ValueError(f"Lead {lead_id} is in stage '{state.current_stage}', not ready for approval.")

        state.approval_status = "approved"
        state.outreach_draft.draft_status = "approved"
        self.store.save_state(state)

        # Stage 6: Execution
        state = self.exec_agent.run(state)
        self.store.save_state(state)
        return state

    def reject_lead(self, lead_id: str, reason: str = "Operator rejected outreach") -> LeadState:
        """Human approval gate trigger: rejects outreach."""
        state = self.store.load_state(lead_id)
        if not state:
            raise ValueError(f"Lead ID {lead_id} not found in state store.")

        state.approval_status = "rejected"
        state.rejection_reason = reason
        if state.outreach_draft:
            state.outreach_draft.draft_status = "rejected"
        state.current_stage = "completed"
        self.store.save_state(state)
        return state
