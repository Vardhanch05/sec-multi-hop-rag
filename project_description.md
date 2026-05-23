# SEC Filing Multi-Hop RAG System: Detailed Project Description

## 1. Project Overview
The **SEC Filing Multi-Hop RAG (Retrieval-Augmented Generation) System** is a full-stack, AI-powered financial research application designed specifically for financial analysts, researchers, and developers. The system's primary goal is to allow users to ask complex, temporally-aware questions across a company's historical SEC EDGAR filings—specifically quarterly (10-Q) and annual (10-K) reports. 

Unlike standard RAG systems that simply retrieve similar text, this system intelligently "hops" across different time periods to track how narratives, financials, and risks evolve. Its core differentiator is an automated **Natural Language Inference (NLI)-based contradiction detection layer** that flags when management's statements in a recent period fundamentally conflict with statements made in previous filings.

## 2. Project Requirements
### Core Features:
- **Automated Ingestion Pipeline:** Capable of fetching metadata and downloading PDF filings directly from the SEC EDGAR database with built-in deduplication, retry, and backoff logic.
- **Smart Parsing & Chunking:** Cleanly extracting text from PDFs, identifying image-only documents, and segmenting the content using regex into standard SEC items (e.g., "Item 1A. Risk Factors", "Management's Discussion and Analysis").
- **Multi-Hop Temporal Retrieval:** Parsing queries to understand temporal references (e.g., "How did risk factors change from Q1 to Q3?") and orchestrating parallel vector searches to retrieve the necessary cross-period contexts.
- **Contradiction Detection:** Evaluating pairs of claims from different periods to automatically detect entailment, neutrality, or contradiction.
- **Structured Data Contracts:** Ensuring strict type safety across all layers using Python `dataclass` contracts (e.g., `FilingRef`, `HopSpec`, `ContradictionEvent`).
- **Property-Based Reliability:** Ensuring the logic is mathematically and logically sound through extensive property-based testing.

## 3. Technicalities & Architecture
The system is built with a highly modular, decoupled architecture following strict data contracts to ensure no raw dictionaries leak between internal layers:
- **Relational Data Layer:** Manages metadata, filing periods, and system state. Uses an interface that can switch between SQLite (local) and PostgreSQL (production).
- **Vector Data Layer:** Manages semantic embeddings for fast retrieval. Capable of switching between ChromaDB (local) and Qdrant (production).
- **Retrieval Hop Planner:** A specialized module (`retrieval/hop_planner.py`) that interprets the logic from the Query Classifier and constructs concrete retrieval specifications, allowing the system to query multiple timeframes independently before synthesizing the final response.
- **Orchestration:** The retrieval layer is designed to call the embedding model exactly once per query (producing a single query vector), then pass that vector to parallel hop threads where metadata filters handle the differentiation across time periods, minimizing latency.

## 4. Issues Encountered & Resolutions
**Issue:** **CI/CD Pipeline Automation Failures**
During the continuous integration and deployment phase, automated GitHub Actions workflows began failing consistently. 
*   **Root Cause 1:** Dependency resolution errors caused by `pdfplumber` and `qdrant-client` facing compatibility issues (e.g., fetching versions that conflicted with other dependencies, like `qdrant-client v1.9.1`).
*   **Root Cause 2:** PyTorch CPU wheels failing to resolve gracefully in a standard `pip install`.
*   **Root Cause 3:** The Python runtime in the GitHub Actions runner was set to `3.14`, which was a pre-release version and unavailable on standard Ubuntu runners, causing immediate setup failures.

**Resolution:**
*   Strictly pinned dependencies in `requirements.txt` to verified, stable PyPI versions (`pdfplumber==0.11.9`, `qdrant-client==1.13.3`). These version bumps were verified locally against existing tests and explicitly documented before being merged to resolve CI issues.
*   Configured pip to explicitly use the PyTorch extra-index-url (`--extra-index-url https://download.pytorch.org/whl/cpu`) to ensure a lightweight CPU-only wheel is fetched for CI, accelerating build times and preventing dependency conflicts.
*   Dropped the target Python runtime in `.github/workflows` from `3.14` back to `3.11`, adhering to the original design document, since Python 3.11 cleanly supports all required ML libraries (like `torch` and `sentence-transformers`) without the lagging compatibility issues often seen in newer Python releases.

## 5. Technology Stack
- **Primary LLM:** Llama 3.3 70B (via Groq API)
- **Fallback LLM:** Llama 3.1 8B (via Groq API)
- **Contradiction Detection Model:** `cross-encoder/nli-deberta-v3-base`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database:** ChromaDB (Development) / Qdrant Cloud (Production)
- **Relational Database:** SQLite (Development) / PostgreSQL (Production)
- **PDF Extraction:** `pdfplumber`
- **Backend API:** FastAPI
- **Frontend / UI:** Streamlit
- **Testing:** `pytest` + `hypothesis`
- **Evaluation:** RAGAS

## 6. Justification for the Tech Stack (The "Why")
- **Llama 3.3 70B + Groq:** SEC filings are dense, verbose, and packed with financial jargon. A 70B parameter model offers exceptional reasoning capabilities needed for synthesis, while Groq's LPU infrastructure guarantees ultra-low latency inference, creating a snappy user experience even when processing large contexts.
- **DeBERTa-v3-base (Cross-Encoder):** Instead of relying entirely on an LLM prompt to spot contradictions (which is prone to hallucination and expensive at scale), a dedicated NLI cross-encoder is fine-tuned on NLI datasets like SNLI and MultiNLI to classify sentence-pair relationships as entailment, neutral, or contradiction. It is much more accurate and cost-effective for this specific sub-task.
- **all-MiniLM-L6-v2:** An incredibly lightweight sentence transformer. Given the massive volume of text in 10-K/10-Q documents, a small, efficient embedding model reduces computation costs and speeds up ingestion without sacrificing too much semantic accuracy for simple chunk retrieval.
- **ChromaDB / Qdrant Split:** ChromaDB is perfect for zero-setup local development, while Qdrant provides the robust, cloud-native scalability and performance required for production vector search.
- **Hypothesis (Property-Based Testing):** Since the RAG system performs complex temporal hops and relational mapping, traditional unit tests (testing single inputs/outputs) are insufficient. Property-based testing generates hundreds of edge cases to verify the mathematical/logical invariants of the system, guaranteeing high reliability in parsing SEC filings.
- **Streamlit:** Allows for the rapid prototyping of interactive, Python-native data dashboards without the overhead of managing a separate frontend SPA (Single Page Application) repository.

## 7. Alternative Approaches Considered
- **LLMs (OpenAI GPT-4o / Anthropic Claude 3.5 Sonnet):** We could have utilized proprietary models. While they offer slightly higher out-of-the-box reasoning, they introduce higher latency and significantly higher API costs. Groq + Llama 3 offers a better speed-to-cost ratio.
- **Contradiction Detection via LLM Prompting:** We could have asked the main LLM to "find contradictions." However, LLMs often struggle with strict boolean entailment logic, sometimes hallucinating conflicts. A specialized NLI model is far more reliable and safer for this task.
- **Frontend (React / Next.js):** Building a custom Next.js frontend would allow for a highly tailored, beautiful UI. However, for a data-heavy, analytical MVP, Streamlit combined with a FastAPI backend provides a 10x faster time-to-market while keeping the entire architecture in Python.
- **Vector Stores (Pinecone / Milvus):** Pinecone is a strong fully-managed alternative to Qdrant, but Qdrant's open-source core and flexible deployment options (local Rust binary vs Cloud) aligned better with our dual-environment (dev/prod) setup.
- **Document Parsing (Unstructured.io / LlamaParse):** Instead of custom regex and `pdfplumber`, we could have used a managed parsing service. While these are powerful, they introduce external dependencies and costs. Writing a custom SEC Section Chunker ensures we perfectly isolate "Item 1A" (Risk Factors) based on standardized SEC formatting rules natively.
