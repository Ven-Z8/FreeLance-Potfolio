"""Data models and schemas for Project 2 Autonomous Multi-Agent Business Workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RawLead(BaseModel):
    """Raw inbound lead received via webhook or parse."""
    lead_id: str = Field(description="Unique lead identifier")
    name: str = Field(description="Prospect full name")
    email: str = Field(description="Prospect email address")
    company: str = Field(description="Prospect company name")
    domain: Optional[str] = Field(default=None, description="Company website domain")
    message: str = Field(description="Inbound inquiry message")


class ResearchCitation(BaseModel):
    """Cited claim from lead enrichment research."""
    claim: str = Field(description="Fact or claim inferred about company/prospect")
    source_url: str = Field(description="Source URL or document citation")
    confidence: float = Field(default=1.0, description="Source reliability score")


class EnrichedLead(BaseModel):
    """Lead populated with web research and firmographics."""
    raw: RawLead
    industry: str
    company_size: str
    tech_stack: List[str] = Field(default_factory=list)
    recent_news: str
    research_citations: List[ResearchCitation] = Field(default_factory=list)


class QualificationResult(BaseModel):
    """ICP (Ideal Customer Profile) evaluation outcome."""
    qualified: bool
    icp_score: float = Field(description="0.0 to 100.0 ICP fit score")
    reasoning: str
    confidence: float = Field(description="Evaluation confidence score 0.0 to 1.0")
    needs_human_review: bool = False


class OutreachDraft(BaseModel):
    """Personalized outreach email grounded in citations."""
    subject: str
    body: str
    grounded_citations: List[str] = Field(default_factory=list)
    draft_status: str = Field(default="pending_approval", description="pending_approval, approved, rejected, sent")


class LeadState(BaseModel):
    """Durable state tracker for an inbound lead across pipeline stages."""
    lead_id: str
    current_stage: str = Field(default="intake", description="intake, research, qualification, drafting, approval_gate, execution, completed, failed")
    raw_lead: RawLead
    enriched_lead: Optional[EnrichedLead] = None
    qualification: Optional[QualificationResult] = None
    outreach_draft: Optional[OutreachDraft] = None
    approval_status: str = Field(default="pending", description="pending, approved, rejected")
    rejection_reason: Optional[str] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    error_log: List[Dict[str, Any]] = Field(default_factory=list)
    execution_result: Optional[Dict[str, Any]] = None
