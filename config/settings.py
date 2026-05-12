"""
config/settings.py
------------------
Central configuration for the SEC Multi-Hop RAG system.

All values are read from environment variables so the same codebase runs in
both local dev (SQLite + ChromaDB) and production (PostgreSQL + Qdrant Cloud)
without any code changes — just a different .env file.

Usage anywhere in the codebase:
    from config.settings import PRIMARY_LLM, CONTRADICTION_THRESHOLD
"""

import os
from dotenv import load_dotenv

# Load .env file if present (dev only — production uses real env vars)
load_dotenv()

# ---------------------------------------------------------------------------
# LLM — Groq API
# These are the exact model ID strings Groq expects in API calls.
# PRIMARY_LLM  : fast 70B model, used for query classification + synthesis
# FALLBACK_LLM : smaller 8B model, used after 3 rate-limit retries on primary
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
PRIMARY_LLM: str = os.getenv("PRIMARY_LLM", "llama-3.3-70b-versatile")
FALLBACK_LLM: str = os.getenv("FALLBACK_LLM", "llama-3.1-8b-instant")

# ---------------------------------------------------------------------------
# Contradiction detection — NLI (DeBERTa cross-encoder)
# CONTRADICTION_THRESHOLD : minimum probability to flag a pair as contradicting
# NLI_TIMEOUT_SECONDS     : CPU inference budget per query (skip if exceeded)
# MAX_NLI_PAIRS           : cap on pairwise comparisons to stay within SLA
#   Math: 4 hops × 5 chunks = 20 claims → up to ~100 cross-period pairs
#   Capped at 50 → 10–20s on CPU, safely within the 30s timeout
# ---------------------------------------------------------------------------
CONTRADICTION_THRESHOLD: float = float(os.getenv("CONTRADICTION_THRESHOLD", "0.75"))
NLI_TIMEOUT_SECONDS: float = float(os.getenv("NLI_TIMEOUT_SECONDS", "30.0"))
MAX_NLI_PAIRS: int = int(os.getenv("MAX_NLI_PAIRS", "50"))

# ---------------------------------------------------------------------------
# Vector store — ChromaDB (dev) or Qdrant Cloud (prod)
# Switch by setting VECTOR_STORE_BACKEND=qdrant in production .env
# ---------------------------------------------------------------------------
VECTOR_STORE_BACKEND: str = os.getenv("VECTOR_STORE_BACKEND", "chromadb")
QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# ---------------------------------------------------------------------------
# Relational database — SQLite (dev) or PostgreSQL (prod)
# Switch by setting DB_BACKEND=postgresql and DATABASE_URL=postgres://... in prod
# ---------------------------------------------------------------------------
DB_BACKEND: str = os.getenv("DB_BACKEND", "sqlite")
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./sec_rag.db")

# ---------------------------------------------------------------------------
# Ingestion
# TICKERS_CONFIG : path to the JSON file listing which companies to track
# ---------------------------------------------------------------------------
TICKERS_CONFIG: str = os.getenv("TICKERS_CONFIG", "./config/tickers.json")
