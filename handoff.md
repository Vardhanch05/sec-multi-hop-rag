# SEC-RAG System: AI Agent Handoff Document

## Project Overview
The SEC Filing Multi-Hop RAG System is a full-stack AI financial research tool. It enables analysts to ask complex, temporally-aware questions across multiple SEC EDGAR quarterly (10-Q) and annual (10-K) filings. The core differentiator is an **NLI-based contradiction detection layer** (`cross-encoder/nli-deberta-v3-base`) that automatically flags when a company's statements in one period conflict with statements from a prior period.

## Current State
**Status:** In Progress (Week 3 of development).
**Completed Tasks:** Tasks 1 through 21.
- **Task 1 & 2:** Repository scaffolding, dependency pinning, and `config/settings.py` setup.
- **Task 3:** Relational Database Layer built (`db/schema.sql`, `db/connection.py`, `db/queries.py`).
- **Task 4:** SEC EDGAR Client (`ingestion/edgar_client.py`) implemented, supporting fetching metadata and downloading PDFs with retry/backoff and deduplication checks.
- **Task 5:** PDF Extractor (`ingestion/pdf_extractor.py`) built using `pdfplumber`, identifying image-only PDFs and cleanly extracting text.
- **Task 6:** Section Chunker (`ingestion/section_chunker.py`) built, segmenting text by regex into standard SEC items and producing chunks.
- **Task 7:** Embedder (`ingestion/embedder.py`) implemented with `sentence-transformers/all-MiniLM-L6-v2`.
- **Task 8:** Vector Store (`ingestion/vector_store.py`) implemented, supporting both ChromaDB and Qdrant.
- **Task 9-11:** Ingestion Orchestration and CI/CD (`ingestion/pipeline.py`, `.github/workflows/`) implemented with property tests and deduplication.
- **Task 12:** Query Classifier (`retrieval/query_classifier.py`) implemented to parse cross-company hop logic natively.
- **Task 13:** Hop Planner (`retrieval/hop_planner.py`) implemented to resolve temporal references and construct concrete retrieval specs.
- **Task 14:** Retriever (`retrieval/retriever.py`) implemented to execute parallel vector searches concurrently.
- **Task 15:** Claim Extractor (`retrieval/claim_extractor.py`) implemented structured LLM-based claim extraction.
- **Task 16:** Checkpoint — ensure all retrieval tests are currently passing (`pytest`).
- **Task 17:** Contradiction Data Contracts (`contradiction/contradiction_report.py`) implemented.
- **Task 18:** NLI Scorer (`contradiction/nli_scorer.py`) implemented with a 5-step pipeline including DeBERTa cross-encoder property tests.
- **Task 19:** Answer Synthesizer (`synthesis/answer_synthesizer.py`) implemented with Groq API integration, citation formatting, contradiction propagation, and RateLimitError fallback handling.
- **Task 20:** End-to-end query pipeline integration test (`tests/test_integration.py`) implemented, asserting 160 multi-hop combination retrieval operations execute completely.
- **Task 21:** Checkpoint — ensured all 47 component tests, unit tests, and property tests across the entire testing suite passed under pytest.
- **Task 22:** Implement `ui/app.py` — layout and sidebar implemented.
- **Task 23:** Implement contradiction cards and cold start polling in `ui/app.py` implemented.
- **Task 24:** Implement RAGAS dashboard tab in `ui/app.py` implemented.
- **CI/CD Fix (2026-05-18):** Resolved three broken package versions in `requirements.txt` (`pdfplumber`, `qdrant-client`) and dropped the Python runtime in workflows from `3.14` (pre-release, unavailable on GitHub runners) to `3.13` (latest stable LTS).

## Pinned Dependency Versions (Verified on PyPI)
| Package | Version | Notes |
|---|---|---|
| `pdfplumber` | `0.11.9` | Latest stable |
| `sentence-transformers` | `5.4.1` | |
| `chromadb` | `1.5.9` | |
| `qdrant-client` | `1.13.3` | v1.9.1 does **not** exist |
| `groq` | `0.9.0` | |
| `streamlit` | `1.57.0` | |
| `hypothesis` | `6.108.5` | |
| `pytest` | `8.2.2` | |
| `ragas` | `0.4.3` | |
| `transformers` | `5.8.0` | |
| `torch` | `2.11.0` | CPU wheel via PyTorch extra-index-url |
| `numpy` | `2.4.4` | |
| `python-dotenv` | `1.0.1` | |
| `requests` | `2.32.3` | |

## Architecture & Tech Stack
- **Primary LLM**: `llama-3.3-70b-versatile` (via Groq API).
- **Fallback LLM**: `llama-3.1-8b-instant` (via Groq API).
- **Contradiction Detection**: `cross-encoder/nli-deberta-v3-base`.
- **Vector Database**: ChromaDB (local dev) / Qdrant Cloud (prod).
- **Relational DB**: SQLite (local dev) / PostgreSQL (prod).
- **Python Runtime**: `3.13` (GitHub Actions `ubuntu-latest`). Do **not** bump to `3.14` — it is pre-release and not available on runners.

## Critical Guidelines for Agents
1. **Property-Based Testing**: This codebase relies heavily on property-based testing via `hypothesis` to validate 22 specific mathematical and logic properties defined in the system design. DO NOT bypass or remove these tests.
2. **Type Safety & Data Contracts**: All layers communicate via strict `dataclass` contracts (`FilingRef`, `FilingPeriod`, `HopSpec`, `ContradictionEvent`, etc.). The database layer (`db/queries.py`) must map raw rows into these dataclasses before returning them. No raw dictionaries should leak to higher-level modules.
3. **Embedder Design Requirement**: The retrieval layer MUST call `embedder.embed_query()` exactly once per query and pass that vector to the parallel hop threads.
4. **Context Maintenance**: Keep `full_context.md` perfectly updated after completing any task.

## Next Immediate Task
**Task 25: Implement evaluation/ragas_harness.py and evaluation/benchmark_questions.json**
- Write 50 single-hop, 100 multi-hop, and 50 contradiction-specific questions in `benchmark_questions.json`.
- Implement `ragas_harness.py` to run the 200-question benchmark against the query pipeline.
- Write tests for RAGAS results persistence and question counts.
