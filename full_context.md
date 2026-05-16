# SEC Multi-Hop RAG — Full Project Context

> This document is a handoff reference for any AI agent or developer picking up this project.
> It covers the project goal, architecture, decisions made, current state, and what comes next.

---

## 1. What This Project Is

**SEC Filing Multi-Hop RAG System** — an AI-powered financial research tool that lets analysts ask complex, temporally-aware questions across multiple SEC EDGAR quarterly and annual filings (10-Q and 10-K) and receive grounded, cited answers.

The core differentiator is an **NLI-based contradiction detection layer** using `cross-encoder/nli-deberta-v3-base` that automatically flags when a company's statements in one quarter conflict with statements from a prior quarter. This capability is absent from all known open-source RAG implementations.

**GitHub repo:** https://github.com/Vardhanch05/sec-multi-hop-rag

---

## 2. Tech Stack

| Layer | Dev | Prod |
|---|---|---|
| LLM (primary) | `llama-3.3-70b-versatile` via Groq API | same |
| LLM (fallback) | `llama-3.1-8b-instant` via Groq API | same |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | same |
| NLI / Contradiction | `cross-encoder/nli-deberta-v3-base` | same |
| Vector DB | ChromaDB | Qdrant Cloud |
| Relational DB | SQLite | PostgreSQL on Render |
| PDF parsing | pdfplumber | same |
| Frontend | Streamlit | Streamlit Community Cloud |
| CI/CD | GitHub Actions | same |
| Evaluation | RAGAS | same |
| Testing | pytest + hypothesis (PBT) | same |
| Python | 3.14.3 | same |

---

## 3. System Architecture

Two workflow tracks:

### Ingestion (daily, automated — GitHub Actions cron 06:00 UTC)
```
SEC EDGAR RSS → edgar_client.py → pdf_extractor.py (pdfplumber)
  → section_chunker.py (regex) → embedder.py (MiniLM) → vector_store.py (ChromaDB/Qdrant)
  → db/queries.py (filings table + ingestion_logs)
```

### Query (per user request, target < 15s end-to-end)
```
User Query → query_classifier.py (Groq → HopPlan JSON)
  → hop_planner.py (temporal resolution → HopSpec list, reads db/queries.py)
  → retriever.py (embed_query once, parallel metadata-filtered vector queries)
  → claim_extractor.py (single batched Groq call → 1 claim per chunk)
  → nli_scorer.py (DeBERTa pairwise NLI, capped at MAX_NLI_PAIRS=50)
  → answer_synthesizer.py (Groq Llama 3.3 70B, retry×3 + fallback)
  → ResponsePayload (answer + citations + contradictions + latency_ms)
  → ui/app.py (Streamlit)
```

---

## 4. Key Design Decisions (important for implementation)

### 4.1 Section-aware chunking
SEC filings have well-defined section headers. Chunks are bounded by section type (MD&A, Risk Factors, Forward Guidance, Financial Statements) using regex, not sliding windows. This enables metadata filtering by section and prevents financial statement data mixing with forward-looking statements.

Regex patterns:
- MD&A: `(?i)(item\s+2\.?\s*management.{0,30}discussion)`
- Risk Factors: `(?i)(item\s+1a\.?\s*risk\s+factors)`
- Forward Guidance: `(?i)(forward[- ]looking\s+statements?\|outlook\|guidance)`
- Financial Statements: `(?i)(item\s+[18]\.?\s*(financial\s+statements\|quantitative))`

### 4.2 NLI over vector similarity for contradiction detection
Two sentences like "We expect 10% growth" and "We revised growth guidance to 3%" have high cosine similarity but are semantically contradictory. DeBERTa NLI evaluates logical relationships (entailment / neutral / contradiction) — it cannot be replaced by similarity search.

### 4.3 Pair count cap (MAX_NLI_PAIRS = 50)
With 4 hops × 5 chunks = 20 claims → up to ~100 cross-period pairs. At 200–400ms per pair on CPU, uncapped = 20–40s (blows the 15s SLA). Capped at 50 pairs via random sampling → 10–20s, within the 30s NLI timeout.

### 4.4 Batched claim extraction (1 Groq call, not N)
For a 4-hop query with top_k=5, claim extraction sends all 20 chunks in a single prompt and gets back a JSON array of 20 claims. This avoids hitting Groq's 30 req/min free-tier limit.

### 4.5 top_k_per_hop = 5 (not 10)
Deliberately lowered from the naive default to keep NLI pair counts manageable.

### 4.6 embed_query() vs embed_chunks()
`embedder.py` has two functions:
- `embed_chunks(chunks)` — batch, used during ingestion
- `embed_query(text)` — single string, used by retriever before vector search

The retriever calls `embed_query()` once, then reuses the vector across all parallel hops.

### 4.7 db/queries.py returns list[FilingPeriod], never list[dict]
`get_filing_periods_for_ticker()` maps each DB row to a typed `FilingPeriod` dataclass before returning. This prevents implicit dict-to-dataclass conversion bugs in hop_planner.py.

### 4.8 Dev/prod switching via env vars only
`DB_BACKEND=sqlite` → SQLite. `DB_BACKEND=postgresql` → PostgreSQL. Same for vector store. No code changes needed between environments.

---

## 5. Data Contracts (key dataclasses)

```python
# ingestion/edgar_client.py
@dataclass
class FilingRef:
    ticker: str
    filing_type: str          # "10-Q" | "10-K"
    accession_number: str     # e.g. "0000320193-24-000123"
    filing_date: date
    source_url: str
    quarter: str | None       # "Q1"–"Q4" | None for 10-K
    fiscal_year: int

# ingestion/section_chunker.py
@dataclass
class Chunk:
    text: str
    ticker: str
    filing_type: str
    quarter: str | None
    fiscal_year: int
    section_type: str         # "MD&A"|"Risk Factors"|"Forward Guidance"|"Financial Statements"|"Other"
    chunk_index: int          # 0-based within filing
    filing_date: date
    accession_number: str
    source_url: str

# retrieval/hop_planner.py
@dataclass
class FilingPeriod:
    ticker: str
    quarter: str | None
    fiscal_year: int
    filing_type: str
    filing_date: date

@dataclass
class HopSpec:
    ticker: str
    quarter: str | None
    fiscal_year: int
    filing_type: str
    section_type: str | None  # None = no section filter

# retrieval/query_classifier.py — HopPlan JSON schema (LLM output)
{
  "hop_count": 4,
  "query_type": "cross_company",   # "single_hop"|"temporal_comparison"|"cross_company"|"trend_analysis"
  "tickers": ["MSFT", "GOOGL"],
  "periods": [{"ticker": "MSFT", "quarter": "Q1", "fiscal_year": 2024}, ...],
  "section_hint": "MD&A",
  "requires_contradiction_check": true
}

# contradiction/contradiction_report.py
@dataclass
class ContradictionEvent:
    ticker: str
    filing_ref_a: str         # accession_number
    filing_ref_b: str
    claim_a: str
    claim_b: str
    confidence_score: float
    query_id: str | None      # None for batch eval runs

@dataclass
class ContradictionReport:
    contradictions: list[ContradictionEvent]
    timed_out: bool

# synthesis/answer_synthesizer.py
@dataclass
class ResponsePayload:
    answer_text: str
    citations: list[Citation]
    contradictions: list[ContradictionEvent]
    contradiction_detection_skipped: bool
    latency_ms: int           # server-side pipeline time
    model_used: str           # which Groq model was actually used
```

---

## 6. Database Schema

### SQLite / PostgreSQL tables

```sql
CREATE TABLE filings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    filing_type TEXT NOT NULL CHECK (filing_type IN ('10-Q', '10-K')),
    quarter TEXT,                    -- NULL for 10-K
    fiscal_year INTEGER NOT NULL,
    filing_date DATE NOT NULL,
    accession_number TEXT NOT NULL UNIQUE,
    source_url TEXT NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ingestion_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TIMESTAMP NOT NULL,
    tickers_processed INTEGER NOT NULL,
    filings_added INTEGER NOT NULL,
    errors TEXT                      -- JSON array of error strings
);

CREATE TABLE ragas_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TIMESTAMP NOT NULL,
    faithfulness REAL NOT NULL,
    answer_relevance REAL NOT NULL,
    context_precision REAL NOT NULL,
    context_recall REAL NOT NULL,
    subset_breakdowns TEXT NOT NULL  -- JSON: {single_hop, multi_hop, contradiction}
);

CREATE TABLE contradiction_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT,                   -- NULL for batch eval runs
    ticker TEXT NOT NULL,
    filing_ref_a TEXT NOT NULL,
    filing_ref_b TEXT NOT NULL,
    claim_a TEXT NOT NULL,
    claim_b TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE benchmark_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    question_type TEXT NOT NULL CHECK (question_type IN ('single_hop', 'multi_hop', 'contradiction')),
    ground_truth_answer TEXT NOT NULL,
    filing_references TEXT NOT NULL  -- JSON array of accession_numbers
);
```

### ChromaDB / Qdrant collection: `sec_chunks`
Fields: `id` (`{accession_number}_{chunk_index}`), `embedding` (float[384]), `ticker`, `filing_type`, `fiscal_year`, `quarter`, `section_type`, `chunk_index`, `filing_date`, `accession_number`, `source_url`

---

## 7. Error Handling Rules

| Scenario | Behaviour |
|---|---|
| Groq rate limit | Retry ×3 with exponential backoff (0s, 1s, 2s), then fallback to `llama-3.1-8b-instant` |
| NLI timeout on CPU | Skip contradiction detection, set `contradiction_detection_skipped=True`, return answer with citations only |
| Filing PDF download fails | Log to `ingestion_logs.errors`, continue with next filing |
| Scanned/image PDF | Log as unparseable, skip, do NOT insert into `filings` table |
| Filing not in corpus | Raise `HopResolutionError` with descriptive message listing available periods |
| Render cold start | UI polls every 2s after 5s threshold, shows loading message, hard timeout at 120s |

---

## 8. UI Behaviour Notes

- Contradiction card color: amber (`#FFA500`) for score ≥ 0.75 and < 0.90, red (`#FF4444`) for score ≥ 0.90
- `contradiction_card_color()` returns `str | None` — callers MUST guard: `if color := contradiction_card_color(score): render_card(...)`
- RAGAS dashboard tab sources data from `ragas_results` table — seed a fixture row before testing
- Cold start polling: 5s trigger → show loading message → poll every 2s → 120s hard timeout → show error

---

## 9. Tickers Tracked (20 companies)

AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, BAC, GS, JNJ, PFE, UNH, XOM, CVX, WMT, HD, V, MA, NFLX

Sectors: Tech, Finance, Healthcare, Energy, Consumer/Retail

---

## 10. Success Metrics

- RAGAS faithfulness ≥ 0.85 on 200-question benchmark
- Contradiction detection F1 ≥ 0.80 on 50 contradiction-specific questions
- Full 200-question eval run completes within 60 minutes
- Handles 20 companies × 8 quarters without retrieval failure
- Live demo URL accessible (Streamlit Community Cloud)

---

## 11. Current Implementation State

### Completed
- [x] Task 1 — Repo scaffold: folder structure, `requirements.txt`, `.env.example`, `README.md`, `.gitignore`, all `__init__.py` files
- [x] Task 2 — `config/settings.py` (all env-var-backed constants), `config/tickers.json` (20 tickers), `tests/test_settings.py` (6 passing tests)
- [x] Task 3 — Database layer: `db/schema.sql`, `db/connection.py`, `db/queries.py`, `tests/test_db.py` (2 passing tests)
- [x] Task 4 — Implement `ingestion/edgar_client.py`
- [x] Task 5 — Implement `ingestion/pdf_extractor.py`
- [x] Task 6 — Implement `ingestion/section_chunker.py`
- [x] Task 7 — Implement `ingestion/embedder.py`

### In Progress / Next
- [ ] Task 8 — Implement `ingestion/vector_store.py`

### Remaining (Tasks 7–27)
See `tasks.md` for the full list. Summary:
- Tasks 4–10: Ingestion pipeline (EDGAR client, PDF extractor, section chunker, embedder, vector store, GitHub Actions)
- Tasks 12–15: Retrieval engine (query classifier, hop planner, retriever, claim extractor)
- Tasks 17–20: Contradiction detection + LLM synthesis + integration test
- Tasks 22–25: Streamlit UI + RAGAS evaluation harness
- Tasks 26–27: Final checkpoint + deployment

---

## 12. Installed Packages (venv — Python 3.14.3)

| Package | Version | Note |
|---|---|---|
| torch | 2.11.0+cpu | CPU build — Python 3.14 compatible |
| numpy | 2.4.4 | Pre-release wheel for Python 3.14 |
| sentence-transformers | 5.4.1 | Pinned 3.0.1 unavailable for Py3.14 |
| transformers | 5.8.0 | Latest |
| chromadb | 1.5.9 | Pinned 0.5.3 had numpy build issue |
| streamlit | 1.57.0 | Latest |
| ragas | 0.4.3 | Latest |
| pdfplumber | 0.11.0 | Exact pin |
| groq | 0.9.0 | Exact pin |
| qdrant-client | 1.9.1 | Exact pin |
| hypothesis | 6.108.5 | Exact pin |
| pytest | 8.2.2 | Exact pin |
| python-dotenv | 1.0.1 | Exact pin |

> Note: Several packages installed at newer versions than pinned in `requirements.txt` due to Python 3.14 compatibility. Functionally identical for this project.

---

## 13. File Structure (current)

```
sec-rag-system/
├── config/
│   ├── __init__.py
│   ├── settings.py          ✅ done
│   └── tickers.json         ✅ done
├── ingestion/
│   └── __init__.py
├── retrieval/
│   └── __init__.py
├── contradiction/
│   └── __init__.py
├── synthesis/
│   └── __init__.py
├── evaluation/
│   └── __init__.py
├── ui/
│   └── __init__.py
├── db/
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_settings.py     ✅ done (6 passing)
├── .github/
│   └── workflows/
│       └── .gitkeep
├── .env.example             ✅ done
├── .gitignore               ✅ done
├── README.md                ✅ skeleton
├── requirements.txt         ✅ done
└── full_context.md          ✅ this file
```

---

## 14. Spec Files Location

The full spec lives in `.kiro/specs/sec-rag-system/`:
- `requirements.md` — 11 requirements with EARS-format acceptance criteria
- `design.md` — full technical design: architecture diagrams (Mermaid), component interfaces, data contracts, 22 correctness properties, testing strategy
- `tasks.md` — 27 implementation tasks with sub-tasks, property test mappings, and requirement traceability
- `.config.kiro` — `{"specType": "feature", "workflowType": "requirements-first"}`
