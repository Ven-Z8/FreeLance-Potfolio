# Autonomous Agentic AI Engineering Portfolio

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: LangChain & LangGraph](https://img.shields.io/badge/Framework-LangChain%20%7C%20LangGraph-green.svg)](https://www.langchain.com/)

Production-grade Agentic AI Systems, RAG Architecture, and Domain-Adaptive Evaluation Frameworks. Built for high-reliability, verifiable accuracy, and production readiness.

---

## 🚀 Projects Overview

| Component | Description | Highlights |
| :--- | :--- | :--- |
| **[P3: Enterprise RAG Orchestrator](./p3-rag-filings)** | Multi-Agent Agentic **Graph** RAG over messy SEC 10-K filings | Typed fact graph + multi-hop augmentation (ratios/CAGR/comparisons), deterministic missing-year clarification, conversational multi-turn UI, Hybrid + BGE-rerank retrieval, safe Python financial-math tool, LangGraph orchestrator — **85.0% → 96.2%** on the audited golden set |
| **[P1: Agent Evaluation Harness](./p1-eval-harness)** | Proving Ground evaluation harness for Agent & RAG systems | Strict Schema Enforcer, Refusal & Hallucination Metrics, Exact/Contains/LLM-Judge Engines, HTML & Markdown Scorecard Generator |
| **[Web Portfolio Showcase](./web)** | Interactive Web Dashboard & Scorecard Explorer | Responsive Dark-Mode UI, Live Metric Breakdown, Brutal 20 Stress Test Visualizer |

---

## 🛠️ Architecture & Key Features

```
               +-------------------------------------------------------+
               |                  User Query / Prompt                  |
               +-------------------------------------------------------+
                                           |
                                           v
               +-------------------------------------------------------+
               |        LangGraph Multi-Role Agent Orchestrator        |
               +-------------------------------------------------------+
                                           |
                 +-------------------------+-------------------------+
                 |                                                   |
                 v                                                   v
  +-----------------------------+                     +-----------------------------+
  |    Query Decomposition &    |                     |  Hybrid Retrieval & Dense   |
  |     Sub-Question Router     |                     |    BGE Reranker Engine     |
  +-----------------------------+                     +-----------------------------+
                 |                                                   |
                 +-------------------------+-------------------------+
                                           |
                                           v
               +-------------------------------------------------------+
               |         Safe Python Financial Execution Tool          |
               +-------------------------------------------------------+
                                           |
                                           v
               +-------------------------------------------------------+
               |      Synthesis Engine & Citation Grounding (LLM)       |
               +-------------------------------------------------------+
                                           |
                                           v
               +-------------------------------------------------------+
               |      Evaluation Harness Scorecard & HTML Dashboard    |
               +-------------------------------------------------------+
```

---

## 💻 Quickstart & Setup Instructions

### 1. Prerequisites
- Python **3.11+**
- [`uv`](https://github.com/astral-sh/uv) (recommended fast package installer) or `pip`
- Git

### 2. Environment Setup

Clone the repository and enter the directory:
```bash
git clone https://github.com/Ven-Z8/FreeLance-Potfolio.git
cd FreeLance-Potfolio
```

Copy the sample environment configuration and add your API credentials:
```bash
cp .env.example .env
```

Fill in your API key in `.env`:
```env
OPENROUTER_API_KEY=sk-or-v1-...
# Or set individual keys:
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIzaSy...
```

---

### 3. Setup Project 3: RAG Filings (`p3-rag-filings`)

```bash
cd p3-rag-filings

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .

# Step 1: Download SEC 10-K filings corpus (~25 financial reports)
python scripts/download_corpus.py

# Step 2: Parse sections and build chunk index
python scripts/dump_sections.py
python scripts/dump_chunks.py
ragfilings index

# Step 3: Build the fact graph, then run an interactive query
ragfilings graph
ragfilings ask "What was Apple's total net sales for FY2025?"

# Conversational multi-turn UI (chat + citations + math + graph)
ragfilings serve          # then open http://127.0.0.1:8000
```

---

### 4. Setup Project 1: Agent Evaluation Harness (`p1-eval-harness`)

```bash
cd ../p1-eval-harness

# Install evaluation harness package
uv pip install -e .

# Run pytest unit tests
pytest tests/

# Execute evaluation benchmark against the golden dataset
eval-harness run --dataset data/domain_a_financial/all_financial_golden.jsonl --output reports/scorecard_hybrid_rerank.html
```

---

### 5. Launch Portfolio Web Dashboard (`web`)

To view the interactive Web UI and scorecard visualizer:

```bash
cd ..
python -m http.server 8080 -d web
```

Open your browser to [http://localhost:8080](http://localhost:8080) to inspect the interactive dashboard.

---

## 📊 Measured Results (golden set, all-free models, $0.00)

Every figure below was measured on this repo's audited golden sets on
2026-08-31 using free models (`minimax/minimax-m3:free` for generation,
extraction, and judging). No external/proprietary benchmark numbers are
claimed; public-benchmark adaptation (e.g. FinanceBench-style) is on the
roadmap. Reproduce with `ragfilings regress`.

**80-case audited golden set** (`p3-rag-filings/golden/golden_set_v1.jsonl`):

| Configuration | Accuracy | Note |
| :--- | :--- | :--- |
| Retrieval-only (hybrid + rerank) | 53.8% | generator refuses figures it actually retrieved |
| **+ fact-graph augmentation** | **85.0%** | +31.2pp — fixes all 16 incorrect refusals |
| **+ missing-year clarification** | **96.2%** | ambiguous 2/10 → 9/10 |

**45-case enterprise multi-hop set** (ratios, CAGR, cross-company comparisons,
trends — answers that live in no single chunk):

| Configuration | Accuracy |
| :--- | :--- |
| Retrieval-only (no graph) | 37.8% |
| **+ fact-graph augmentation** | **84.4%** |

Retrieval-strategy ablation (no graph, v1 set): dense 56.2% · hybrid 46.2% ·
hybrid + rerank 55.0% — within run-to-run noise; the fact graph, not the
ranker, is the ~30-point signal. Judge calibration: 86.5% human agreement /
Cohen's kappa 0.669. Failure analysis and caveats:
[`p3-rag-filings/docs/graph_augmentation_v1.md`](./p3-rag-filings/docs/graph_augmentation_v1.md).

---

## 📂 Repository Structure

```
FreeLance-Potfolio/
├── README.md                      # Global setup & portfolio overview
├── .env.example                   # Environment keys template
├── p3-rag-filings/                # Project 3: Multi-Agent RAG Orchestrator
│   ├── src/ragfilings/            # Core RAG source code (retrieval, agent, prompts)
│   ├── golden/                    # Evaluation golden datasets & schemas
│   ├── scripts/                   # SEC EDGAR downloader & parser scripts
│   └── pyproject.toml             # Python dependencies & CLI entrypoints
├── p1-eval-harness/               # Project 1: Domain-Adaptive Eval Harness
│   ├── src/harness/               # Metric engines, schema validators, reporters
│   ├── data/                      # Multi-domain golden datasets
│   └── reports/                   # HTML/Markdown scorecards & JSON traces
├── web/                           # Portfolio Web UI Showcase
│   ├── index.html                 # Interactive portfolio homepage
│   ├── app.js                     # Dashboard interaction logic
│   └── styles.css                 # Custom modern dark-mode styles
└── shared/                        # Shared datasets, specs, and architecture docs
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
