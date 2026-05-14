-- filings: one row per ingested filing
CREATE TABLE IF NOT EXISTS filings (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker            TEXT NOT NULL,
    filing_type       TEXT NOT NULL CHECK (filing_type IN ('10-Q', '10-K')),
    quarter           TEXT,                    -- NULL for 10-K
    fiscal_year       INTEGER NOT NULL,
    filing_date       DATE NOT NULL,
    accession_number  TEXT NOT NULL UNIQUE,
    source_url        TEXT NOT NULL,
    ingested_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ingestion_logs: one row per daily ingestion run
CREATE TABLE IF NOT EXISTS ingestion_logs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp     TIMESTAMP NOT NULL,
    tickers_processed INTEGER NOT NULL,
    filings_added     INTEGER NOT NULL,
    errors            TEXT                     -- JSON array of error strings
);

-- ragas_results: one row per evaluation run
CREATE TABLE IF NOT EXISTS ragas_results (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp         TIMESTAMP NOT NULL,
    faithfulness          REAL NOT NULL,
    answer_relevance      REAL NOT NULL,
    context_precision     REAL NOT NULL,
    context_recall        REAL NOT NULL,
    subset_breakdowns     TEXT NOT NULL        -- JSON: {single_hop, multi_hop, contradiction}
);

-- contradiction_events: one row per detected contradiction
CREATE TABLE IF NOT EXISTS contradiction_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id          TEXT,                            -- NULL for batch eval runs
    ticker            TEXT NOT NULL,
    filing_ref_a      TEXT NOT NULL,           -- accession_number
    filing_ref_b      TEXT NOT NULL,
    claim_a           TEXT NOT NULL,
    claim_b           TEXT NOT NULL,
    confidence_score  REAL NOT NULL,
    detected_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- benchmark_questions: 200-question evaluation set
CREATE TABLE IF NOT EXISTS benchmark_questions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    question          TEXT NOT NULL,
    question_type     TEXT NOT NULL CHECK (question_type IN ('single_hop', 'multi_hop', 'contradiction')),
    ground_truth_answer TEXT NOT NULL,
    filing_references TEXT NOT NULL            -- JSON array of accession_numbers
);
