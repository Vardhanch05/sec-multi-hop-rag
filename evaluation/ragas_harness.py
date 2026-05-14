"""
evaluation/ragas_harness.py
---------------------------
Runner for the RAGAS evaluation benchmark.
"""

from dataclasses import dataclass
import datetime

@dataclass
class RagasResult:
    run_timestamp: datetime.datetime
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    subset_breakdowns: dict  # Serialized to JSON when saved
