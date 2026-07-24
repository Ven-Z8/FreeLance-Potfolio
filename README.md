# Autonomous Agentic AI Engineering Portfolio

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: LangChain & LangGraph](https://img.shields.io/badge/Framework-LangChain%20%7C%20LangGraph-green.svg)](https://www.langchain.com/)

Production-grade Agentic AI Systems, RAG Architecture, and Domain-Adaptive Evaluation Frameworks. Built for high-reliability, verifiable accuracy, and production readiness.

---

## 🚀 Projects Overview

| Component | Description | Highlights |
| :--- | :--- | :--- |
| **[P3: Enterprise RAG Orchestrator](./p3-rag-filings)** | Multi-Agent RAG system over messy SEC 10-K financial filings | Hybrid (BM25 + BGE Dense) Retrieval, BGE Reranker, Query Decomposition, Ticker Masking, Python Financial Math Execution Tool, LangGraph Orchestrator |
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

# Step 3: Run interactive query or benchmark test
ragfilings query "What was Apple's total net sales for FY2025?"
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

## 📊 Benchmark Performance Results

Evaluating on **FinanceBench**, **RAGAS**, **RGB (Robustness)**, and **FinQA** enterprise benchmarks:

| Benchmark | Strategy | Accuracy / Score | Key Metric |
| :--- | :--- | :--- | :--- |
| **FinanceBench (Hardest)** | Hybrid + BGE Rerank | **71.4%** | Fact Verification & Exact Numerical Math |
| **RGB (Noise & Refusal)** | Ticker Filter + Guard | **100%** | Unanswerable Abstention & Noise Resistance |
| **RAGAS Synthesis** | Synthesis Prompt v2 | **80.0%** | Context Precision & Citation Grounding |
| **FinQA** | Python Math Exec Tool | **75.0%** | Multi-Step Financial Computation |

---

## 📂 Repository Structure

```
FreeLance-Potfolio/
├── README.md                      # Global setup & portfolio overview
├── TRACKER.md                     # Development roadmap & phase tracker
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
