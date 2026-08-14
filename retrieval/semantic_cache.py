"""
retrieval/semantic_cache.py
---------------------------
Semantic cache with SQLite persistence so hits survive server restarts.
Falls back to in-memory only if the DB is unavailable.
"""

import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from synthesis.answer_synthesizer import ResponsePayload, Citation
from contradiction.contradiction_report import ContradictionEvent

logger = logging.getLogger(__name__)

_DB_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS semantic_cache (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT NOT NULL,
    embedding   TEXT NOT NULL,
    answer      TEXT NOT NULL,
    citations   TEXT NOT NULL,
    contradictions TEXT NOT NULL,
    model_used  TEXT NOT NULL,
    contradiction_detection_skipped INTEGER NOT NULL DEFAULT 0,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

def _payload_to_row(query: str, emb: List[float], payload: ResponsePayload) -> dict:
    return {
        "query": query,
        "embedding": json.dumps(emb),
        "answer": payload.answer,
        "citations": json.dumps([
            {"ticker": c.ticker, "fiscal_year": c.fiscal_year,
             "filing_type": c.filing_type, "section": c.section,
             "accession_number": c.accession_number}
            for c in payload.citations
        ]),
        "contradictions": json.dumps([
            {"ticker": c.ticker, "filing_ref_a": c.filing_ref_a, "claim_a": c.claim_a,
             "filing_ref_b": c.filing_ref_b, "claim_b": c.claim_b,
             "confidence_score": c.confidence_score, "query_id": c.query_id}
            for c in payload.contradictions
        ]),
        "model_used": payload.model_used,
        "contradiction_detection_skipped": int(payload.contradiction_detection_skipped),
    }

def _row_to_payload(row) -> ResponsePayload:
    citations = [Citation(**c) for c in json.loads(row["citations"])]
    contradictions = [ContradictionEvent(**c) for c in json.loads(row["contradictions"])]
    return ResponsePayload(
        answer=row["answer"],
        citations=citations,
        contradictions=contradictions,
        latency_ms=0,
        model_used=row["model_used"],
        contradiction_detection_skipped=bool(row["contradiction_detection_skipped"]),
    )


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.96, max_size: int = 200):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        # In-memory hot layer
        self._hot: List[Dict[str, Any]] = []
        self._db_available = False
        self._init_db()

    def _init_db(self):
        """Initialise SQLite persistence layer; gracefully degrade if unavailable."""
        try:
            from db.connection import get_connection
            with get_connection() as conn:
                conn.execute(_DB_TABLE_SQL)
                conn.commit()
                # Warm in-memory hot layer from persisted rows (latest max_size)
                rows = conn.execute(
                    "SELECT * FROM semantic_cache ORDER BY created_at DESC LIMIT ?",
                    (self.max_size,)
                ).fetchall()
            for row in reversed(rows):  # oldest first into hot layer
                try:
                    emb = json.loads(row["embedding"])
                    payload = _row_to_payload(row)
                    self._hot.append({"query": row["query"], "embedding": emb, "response": payload})
                except Exception:
                    pass
            self._db_available = True
            logger.info(f"Semantic cache warmed with {len(self._hot)} persisted entries.")
        except Exception as e:
            logger.warning(f"Semantic cache DB unavailable, using in-memory only: {e}")

    def get(self, query: str, query_embedding: List[float]) -> Optional[ResponsePayload]:
        if not self._hot:
            return None

        q_emb = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            return None

        best_sim = -1.0
        best_entry = None

        for entry in self._hot:
            entry_emb = np.array(entry["embedding"], dtype=np.float32)
            entry_norm = np.linalg.norm(entry_emb)
            if entry_norm == 0:
                continue
            sim = float(np.dot(q_emb, entry_emb) / (q_norm * entry_norm))
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_sim >= self.similarity_threshold and best_entry is not None:
            logger.info(
                f"Semantic cache HIT (sim={best_sim:.4f}): '{best_entry['query']}'"
            )
            return best_entry["response"]

        return None

    def set(self, query: str, query_embedding: List[float], response: ResponsePayload):
        # Never cache error/overloaded responses
        ans_lower = response.answer.lower()
        if "error" in ans_lower or "overloaded" in ans_lower or "rate limit" in ans_lower:
            return

        entry = {"query": query, "embedding": query_embedding, "response": response}

        # Update hot layer (FIFO eviction)
        if len(self._hot) >= self.max_size:
            self._hot.pop(0)
        self._hot.append(entry)

        # Persist to SQLite
        if self._db_available:
            try:
                from db.connection import get_connection
                row = _payload_to_row(query, query_embedding, response)
                with get_connection() as conn:
                    conn.execute(
                        """INSERT INTO semantic_cache
                           (query, embedding, answer, citations, contradictions,
                            model_used, contradiction_detection_skipped)
                           VALUES (:query, :embedding, :answer, :citations, :contradictions,
                                   :model_used, :contradiction_detection_skipped)""",
                        row,
                    )
                    # Keep the table bounded
                    conn.execute(
                        """DELETE FROM semantic_cache WHERE id NOT IN (
                               SELECT id FROM semantic_cache ORDER BY created_at DESC LIMIT ?
                           )""",
                        (self.max_size,),
                    )
                    conn.commit()
                logger.info(f"Persisted cache entry for: '{query}'")
            except Exception as e:
                logger.warning(f"Failed to persist cache entry: {e}")


_global_cache = SemanticCache()


def get_semantic_cache() -> SemanticCache:
    global _global_cache
    return _global_cache
