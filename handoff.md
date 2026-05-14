# SEC-RAG System: AI Agent Handoff Document

## Project Overview
The SEC Filing Multi-Hop RAG System is a full-stack AI financial research tool. It enables analysts to ask complex, temporally-aware questions across multiple SEC EDGAR quarterly (10-Q) and annual (10-K) filings. The core differentiator is an **NLI-based contradiction detection layer** (`cross-encoder/nli-deberta-v3-base`) that automatically flags when a company's statements in one period conflict with statements from a prior period.

## Current State
**Status:** In Progress (Week 2 of development).
**Completed Tasks:** Tasks 1 through 5.
- **Task 1 & 2:** Repository scaffolding, dependency pinning, and `config/settings.py` setup.
- **Task 3:** Relational Database Layer built (`db/schema.sql`, `db/connection.py`, `db/queries.py`).
- **Task 4:** SEC EDGAR Client (`ingestion/edgar_client.py`) implemented, supporting fetching metadata and downloading PDFs with retry/backoff and deduplication checks.
- **Task 5:** PDF Extractor (`ingestion/pdf_extractor.py`) built using `pdfplumber`, identifying image-only PDFs and cleanly extracting text.
- **Tests:** All 13 property and unit tests are currently passing (`pytest`).

## Architecture & Tech Stack
- **Primary LLM**: `llama-3.3-70b-versatile` (via Groq API).
- **Fallback LLM**: `llama-3.1-8b-instant` (via Groq API).
- **Contradiction Detection**: `cross-encoder/nli-deberta-v3-base`.
- **Vector Database**: ChromaDB (local dev) / Qdrant Cloud (prod).
- **Relational DB**: SQLite (local dev) / PostgreSQL (prod).

## Critical Guidelines for Agents
1. **Property-Based Testing**: This codebase relies heavily on property-based testing via `hypothesis` to validate 22 specific mathematical and logic properties defined in the system design. DO NOT bypass or remove these tests.
2. **Type Safety & Data Contracts**: All layers communicate via strict `dataclass` contracts (`FilingRef`, `FilingPeriod`, `HopSpec`, `ContradictionEvent`, etc.). The database layer (`db/queries.py`) must map raw rows into these dataclasses before returning them. No raw dictionaries should leak to higher-level modules.
3. **Embedder Design Requirement**: The retrieval layer MUST call `embedder.embed_query()` exactly once per query and pass that vector to the parallel hop threads.
4. **Context Maintenance**: Keep `full_context.md` perfectly updated after completing any task.

## Next Immediate Task
**Task 6: Implement `ingestion/section_chunker.py`**
- Segment the text by regex into standard SEC items (e.g., "Item 1A. Risk Factors", "Item 7. MD&A").
- Break each section into 1000-token chunks with 200-token overlap.
- Implement Property 1 (Metadata Completeness), Property 2 (Section Validity), and Property 22 (Serialization Round-trip).
