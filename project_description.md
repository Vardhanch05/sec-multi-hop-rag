# SEC Filing Multi-Hop RAG System (ScribeGraph): Detailed Project Description

## 1. Project Overview
The **SEC Filing Multi-Hop RAG (Retrieval-Augmented Generation) System** (codenamed **ScribeGraph**) is an enterprise-grade, full-stack AI research platform built for financial analysts, regulatory auditors, and quantitative researchers. The system enables users to ask complex, temporally-aware questions across historical SEC EDGAR filings—specifically quarterly (10-Q) and annual (10-K) reports for public companies.

Unlike conventional single-shot RAG applications that retrieve simple semantic matches, ScribeGraph intelligently plans multi-hop retrieval specs across standardized fiscal periods (e.g., comparing FY2024 vs. FY2025 or tracking Q1–Q4 trend lines). Crucially, the system incorporates a fine-tuned **Natural Language Inference (NLI) contradiction detection layer** using a cross-encoder model to identify when management disclosures or risk factors in recent filings conflict with prior statements.

---

## 2. Key Features & Capabilities

- **Multi-Hop Temporal Retrieval:** Translates complex user inquiries (e.g., *"How did NVIDIA's R&D spend and risk disclosures evolve between FY2024 and FY2025?"*) into parallelized timeframe retrieval plans.
- **Automated Contradiction Detection:** Evaluates cross-period claim pairs using `cross-encoder/nli-deberta-v3-base` to automatically flag disclosure inconsistencies (entailment vs. contradiction scores).
- **Async Background Ingestion Engine:** FastAPI `BackgroundTasks` + SQLite job tracking (`ingestion_tasks`) coupled with a Next.js `IngestModal` component, enabling non-blocking background SEC EDGAR fetching, parsing, and vector embedding with live progress bars.
- **Modern Full-Stack Architecture:**
  - **Frontend:** Next.js 14/16 (React 19, TypeScript, Tailwind CSS, Framer Motion) providing a dynamic dark-mode financial research dashboard with real-time SSE streaming.
  - **Backend:** FastAPI (Python 3.11) exposing structured REST and streaming endpoints (`/api/chat`, `/api/stats`, `/api/ingest`, `/api/ragas`).
- **Containerization & Cloud Readiness:** Fully containerized with Docker & Docker Compose (`sec-rag-backend` and `sec-rag-frontend`), ensuring 100% reproducible environments.
- **Automated SEC EDGAR Ingestion:** Automated fetching, parsing, and section chunking (isolating MD&A, Risk Factors, Financial Tables) via `edgartools`.
- **Hybrid Storage & Caching:** Dual SQLite/PostgreSQL relational data store alongside ChromaDB/Qdrant vector store, with an in-memory semantic query cache to eliminate redundant LLM inference.
- **High-Rigor Evaluation & Testing:** 50+ Pytest unit/integration tests with property-based testing (`hypothesis`) and automated 5-dimensional benchmark evaluations via RAGAS.

---

## 3. Architecture & System Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Next.js 16 Dashboard (React 19)                     │
│               [Chat Interface, Citations, Contradiction Cards, Ingest Modal]│
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ REST / Server-Sent Events (SSE)
┌─────────────────────────────────────v───────────────────────────────────────┐
│                          FastAPI Backend (ui/api.py)                        │
└─────┬───────────────────────────────┬───────────────────────────────┬───────┘
      │                               │                               │
┌─────v──────────────┐      ┌─────────v──────────────┐      ┌─────────v──────────────┐
│  Query Classifier  │      │  Multi-Hop Retrieval   │      │ Async Ingestion Worker │
│ (Temporal Specs)   │      │ (Vector + Cache + DB)  │      │(FastAPI BackgroundTask)│
└────────────────────┘      └────────────────────────┘      └────────────────────────┘
                                      │
                            ┌─────────v──────────────┐
                            │   Answer Synthesizer   │
                            │ (Groq Llama 3.3 70B)   │
                            └────────────────────────┘
```

1. **Query Planning:** The input query is analyzed by `QueryClassifier` to determine target tickers, fiscal years, and document sections (e.g., MD&A, Risk Factors).
2. **Multi-Hop Search:** `HopPlanner` constructs independent search queries per period, retrieving vector chunks from ChromaDB/Qdrant without cross-period text starvation.
3. **Claim & Contradiction Scoring:** `ClaimExtractor` isolates key assertions, and `NLIScorer` evaluates statement pairs to detect contradictions ($Score \ge 0.70$).
4. **Synthesis & Streaming:** `AnswerSynthesizer` streams structured markdown answers with cited accession numbers and contradiction callout cards via FastAPI SSE streams.
5. **Non-blocking Ingestion:** Users queue new filings via `POST /api/ingest`, which executes asynchronous EDGAR downloading and ChromaDB vector indexing while reporting real-time progress percentages via `GET /api/ingest/status/{task_id}`.

---

## 4. Technology Stack

| Category | Technology | Description / Justification |
| :--- | :--- | :--- |
| **Primary LLM** | Llama 3.3 70B (Groq API) | Ultra-fast inference via Groq LPUs with exceptional financial reasoning. |
| **Fallback LLM** | Llama 3.1 8B (Groq API) | High-speed fallback when primary model hits rate limits. |
| **NLI Model** | `cross-encoder/nli-deberta-v3-base` | Precise sentence-pair contradiction classification without LLM hallucinations. |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight 384-dim semantic embeddings optimized for fast vector retrieval. |
| **Vector DB** | ChromaDB (Dev) / Qdrant (Prod) | Local zero-config vector store with seamless upgrade path to Qdrant Cloud. |
| **Relational DB** | SQLite (Dev) / PostgreSQL (Prod) | Structured metadata, semantic query cache, background job states, and filing metrics storage. |
| **Backend API** | FastAPI + Uvicorn (Python 3.11) | Asynchronous, type-safe Python backend with Pydantic contracts & `BackgroundTasks`. |
| **Frontend UI** | Next.js 16 + React 19 + Tailwind CSS | Slate-stone financial dashboard with real-time SSE streaming & Framer Motion. |
| **Containerization**| Docker & Docker Compose | Containerized multi-service deployment (`sec-rag-backend`, `sec-rag-frontend`). |
| **Testing** | Pytest + Hypothesis | Unit, integration, and property-based edge-case test suite. |
| **Evaluation** | RAGAS Framework | Automated Faithfulness, Answer Relevancy, and Context Precision metrics. |

---

## 5. Technology Choices & Justification ("Why")

- **Why FastAPI BackgroundTasks for Ingestion?**  
  SEC EDGAR downloads, parsing, and vector embedding take 15 to 60+ seconds. Running ingestion synchronously in an HTTP request would cause browser timeouts and freeze the UI. `BackgroundTasks` returns HTTP 202 immediately and processes heavy workload asynchronously in the background.

- **Why DeBERTa-v3 NLI Cross-Encoder instead of LLM Prompting?**  
  LLMs often hallucinate or struggle with strict boolean entailment logic when asked to detect contradictions in long documents. A fine-tuned cross-encoder directly computes explicit entailment, neutral, and contradiction probability distributions, offering 10x higher reliability at zero LLM token cost.

- **Why FastAPI + Next.js (React 19) split?**  
  Separating the core Python ML/RAG engine into FastAPI while deploying a modern React 19 / Next.js frontend delivers a high-performance web experience with instant UI rendering, server-side streaming, and standard REST API separation.

- **Why Multi-Hop Spec Planning?**  
  Standard RAG queries fetch a single top-K set of document chunks, which causes single-period dominance (e.g., Q1 chunks starving Q3 chunks). Multi-hop planning executes independent per-period queries, guaranteeing equal context budget across timeframes.

- **Why Full Containerization with Docker Compose?**  
  Complex AI/ML applications require specific system binaries, C++ compilers, Node runtimes, and Python dependencies. Docker guarantees 100% environment reproducibility across Windows, Linux, and Cloud deployments with a single `docker compose up` command.

---

## 6. Challenges Encountered & Resolutions

### 1. **HTTP Timeouts on Heavy Filing Ingestion**
- **Issue:** Fetching and indexing large SEC 10-K filings exceeded HTTP request timeout limits.
- **Resolution:** Built a background job task queue in FastAPI using `BackgroundTasks` and SQLite job status tracking, paired with real-time polling in the Next.js UI (`IngestModal`).

### 2. **ChromaDB Prose Dominance & Vector Starvation**
- **Issue:** Multi-period queries frequently returned chunks from only one filing because narrative sections scored slightly higher than comparative table sections.
- **Resolution:** Implemented independent vector collection retrieval per timeframe and chunk-type in `retriever.py`, enforcing character context budgeting per hop before passing to the synthesizer.

### 3. **API Rate Limiting & Latency Bottlenecks**
- **Issue:** Consecutive multi-hop LLM calls hit Groq rate limits, causing latency spikes and failed responses.
- **Resolution:** Introduced instant dual-model fallbacks (switching from Llama 3.3 70B to Llama 3.1 8B on `RateLimitError`) and SQLite-backed semantic query caching (`retrieval/semantic_cache.py`).

---

## 7. How to Run the Project

Refer to `run_instructions.txt` for detailed launch commands:

```powershell
# Run entire application with Docker Compose
docker compose up -d

# Frontend: http://localhost:3000
# Backend:  http://localhost:8001
```
