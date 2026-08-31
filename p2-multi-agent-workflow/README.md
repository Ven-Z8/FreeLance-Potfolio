# Autonomous Multi-Agent Business Workflow

Production-grade agentic pipeline for lead intake, web research enrichment, ideal customer profile (ICP) qualification, citation-grounded outreach drafting, and human approval execution.

---

## 🎯 Architecture & Pipeline Stages

```
 +--------------+       +-------------------+       +--------------------+
 | Inbound Lead | ----> |  Research Agent   | ----> | Qualification      |
 | (Webhook/API)|       |  (Cited Facts)    |       | Agent (ICP Score)  |
 +--------------+       +-------------------+       +--------------------+
                                                              |
                                                              v
 +--------------+       +-------------------+       +--------------------+
 | Execution    | <---- | Human Approval    | <---- | Outreach Drafting  |
 | Agent (CRM)  |       | Gate (Safety)     |       | Agent (Grounded)   |
 +--------------+       +-------------------+       +--------------------+
```

### Key Engineering Features
- **Durable State Store:** SQLite-backed state machine ensuring mid-pipeline crash recovery and state resumption.
- **Strict Citation Grounding:** Every claim in generated outreach must trace to an enriched research source URL.
- **Human Approval Gate:** Mandatory review queue preventing any unapproved outbound communication.
- **Cost & Token Meter:** Real-time dollar and token tracking per lead (`~$0.04 - $0.14` per lead).

---

## 💻 Quickstart & Verification

```bash
# Install package
uv pip install -e .

# Run unit tests
pytest tests/
```
