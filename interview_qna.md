# SEC Filing Multi-Hop RAG System
## Complete Project Deep-Dive Interview Preparation
**Vardhan Chilakamarri | 50 Questions**

**Topics:** Project Overview · Multi-Hop RAG · NLI & Contradiction Detection · Vector DB & Embeddings · PDF Parsing & Chunking · RAGAS Evaluation · Tech Stack Decisions · Testing & CI/CD · Deployment · Edge Cases & Improvements

---

### — PROJECT OVERVIEW —

**Q1. Walk me through your SEC Filing Multi-Hop RAG System — what does it do and why did you build it?**
**Answer:**
It's a financial research tool that lets users ask complex, temporally-aware questions across 160+ SEC EDGAR filings (10-Q/10-K) from 20 companies across 8 quarters. The key insight: standard RAG just finds similar text. This system intelligently hops across time periods to track how narratives, financials, and risks evolve — for example, 'How did Apple's risk factors change from Q1 2023 to Q3 2023?' Beyond retrieval, it has a contradiction detection layer using a DeBERTa-v3 NLI model that automatically flags when management quietly walks back prior guidance. Deployed at $0/month on Render + Streamlit Cloud.
*Interview Tip: Open with the core differentiator: multi-hop temporal retrieval + contradiction detection. Don't just say 'it answers questions about SEC filings.'*

**Q2. What is a SEC filing? What are 10-K and 10-Q reports?**
**Answer:**
SEC (Securities and Exchange Commission) filings are mandatory documents that public companies submit. 10-K: annual report — comprehensive overview of financial performance, risk factors, business description, audited financials. Filed once a year. 10-Q: quarterly report — unaudited financials and business updates for Q1, Q2, Q3 (Q4 is covered by 10-K). Together they form a longitudinal record of a company's financial narrative. My system ingests both types to enable temporal analysis across 8 quarters per company.

**Q3. What is EDGAR and how did you use it?**
**Answer:**
EDGAR (Electronic Data Gathering, Analysis, and Retrieval) is the SEC's public database where all mandated filings are submitted and publicly accessible. I built an automated ingestion pipeline that fetches filing metadata directly from EDGAR's REST API, downloads the PDF filings, deduplicates them, and handles failures with retry and exponential backoff logic. This gave me 160+ PDFs across 20 companies with no manual downloading — fully automated data collection.

**Q4. What makes your system different from a standard RAG system?**
**Answer:**
Three key differentiators: (1) Multi-hop temporal retrieval — standard RAG retrieves from one pool. Mine uses a hop planner to decompose a query into multiple time-specific retrieval specs and runs them in parallel, then synthesizes across results. (2) NLI-based contradiction detection — standard RAG has no mechanism to detect when a company changes its story. DeBERTa-v3 cross-encoder classifies claim pairs as entailment, neutral, or contradiction — a capability no similarity-based system can replicate. (3) RAGAS evaluation — I rigorously benchmarked the system on 200 questions with quantitative targets (Faithfulness ≥0.85, Context Precision ≥0.88).
*Interview Tip: These three points are your strongest differentiators. Memorize them.*

---

### — MULTI-HOP RAG —

**Q5. What is multi-hop RAG and how is it different from standard RAG?**
**Answer:**
Standard RAG: single retrieval step — embed query, find top-k similar chunks, pass to LLM. Works for single-document or single-timepoint questions. Multi-hop RAG: decomposes a complex query into multiple retrieval steps ('hops'), each targeting different documents or time periods, then synthesizes results. Example: 'How did risk factors change from Q1 to Q3?' requires two hops — one to Q1 filing, one to Q3 filing — before comparison. The hops can be sequential (output of one feeds the next) or parallel (independent hops merged at synthesis).
*Interview Tip: Be ready to draw the pipeline: Query → Hop Planner → Parallel Retrieval Threads → Context Aggregation → LLM Synthesis.*

**Q6. Explain the architecture of your retrieval pipeline end-to-end.**
**Answer:**
Pipeline: (1) User query arrives at FastAPI endpoint. (2) Query Classifier identifies temporal references and query intent. (3) Hop Planner (`hop_planner.py`) converts the classified query into concrete `HopSpec` dataclass objects — each specifying which filing(s) to retrieve from. (4) The query is embedded ONCE using `all-MiniLM-L6-v2` — single vector for all hops. (5) Parallel threads execute each hop: same query vector + different metadata filters (company, period, filing type) sent to ChromaDB. (6) Retrieved chunks from all hops are aggregated into a combined context. (7) Llama 3.3 70B via Groq synthesizes the final answer. (8) Contradiction detection runs pairwise NLI on cross-period claims.

**Q7. What is the hop planner? How does it work?**
**Answer:**
The hop planner (`retrieval/hop_planner.py`) is the brain of the temporal reasoning. It takes the output of the Query Classifier — which has already identified temporal references — and converts them into concrete `HopSpec` dataclass objects. Each `HopSpec` says: 'retrieve from company X, filing period Y, with these metadata filters.' For example, 'last 4 quarters' is resolved into 4 specific filing identifiers (Q2-2023, Q3-2023, Q4-2023, Q1-2024). These specs are then dispatched to parallel retrieval threads. Key design: the hop planner never touches embeddings — it purely handles the logical decomposition.

**Q8. Why do you embed the query only once even for multiple hops?**
**Answer:**
The semantic meaning of the query doesn't change between hops — 'risk factors' means the same thing whether we're searching Q1 or Q3. Embedding is a computationally non-trivial operation. Calling the embedding model once and reusing the resulting vector across all parallel hop threads eliminates redundant computation and reduces latency. The differentiation between hops is handled entirely by metadata filters in ChromaDB — not by different query vectors. This single embedding + parallel metadata-filtered retrieval design is what enables sub-15-second response times despite searching across multiple documents.

**Q9. How did you handle temporal references like 'last 4 quarters' in natural language?**
**Answer:**
The Query Classifier identifies temporal expression patterns in the query (regex + NLP heuristics). The hop planner then resolves them against the filing metadata database: 'last 4 quarters' → queries the SQLite/PostgreSQL metadata store for the 4 most recent filings for the specified company → extracts their filing identifiers. The resolution is relative to whatever filings exist in the system, not hardcoded dates. This means 'last 4 quarters' gives correct results whether you're asking in 2024 or 2025. The resolved filing IDs become the metadata filter parameters for ChromaDB retrieval.
*Interview Tip: Emphasize that the resolution is dynamic (against the actual database), not hardcoded.*

**Q10. What are the data contracts (dataclasses) you used and why?**
**Answer:**
I defined strict Python dataclass contracts for every inter-module boundary: `FilingRef` (filing identifier, company, period, type, source URL), `HopSpec` (retrieval specification — query vector, filing refs, top-k, metadata filters), `ContradictionEvent` (claim A, claim B, NLI label, confidence score, filing periods). Why: prevents raw dictionaries from leaking between layers — a dict is untyped and can silently have wrong/missing keys. Dataclasses give type safety, IDE autocompletion, and validation. If a module receives a `HopSpec`, it knows exactly what fields exist and their types. This is production engineering practice.
*Interview Tip: This shows software engineering maturity beyond just ML. Interviewers love this.*

---

### — NLI & CONTRADICTION DETECTION —

**Q11. What is Natural Language Inference (NLI)?**
**Answer:**
NLI (also called Recognizing Textual Entailment) is an NLP task that classifies the logical relationship between a premise and a hypothesis. Three labels: Entailment (hypothesis logically follows from premise — same claim), Neutral (hypothesis is neither confirmed nor contradicted), Contradiction (hypothesis conflicts with premise — opposite claims). Example: Premise: 'Supply chain risks are minimal.' Hypothesis: 'Supply chain disruptions pose a significant threat.' → Contradiction. NLI models are trained on datasets like SNLI and MultiNLI — millions of labeled sentence pairs.
*Interview Tip: NLI is a key concept — understand all three labels with examples.*

**Q12. What is a cross-encoder and how is it different from a bi-encoder?**
**Answer:**
Bi-encoder: encodes the two sentences independently into separate vectors, then computes similarity (cosine). Fast — vectors can be precomputed. Used for retrieval (FAISS, ChromaDB). Cross-encoder: takes both sentences as a single concatenated input and produces a single score/classification. The model attends to both sentences simultaneously — much richer interaction. Slower (must run for every pair) but significantly more accurate for classification tasks. DeBERTa-v3 cross-encoder is the right choice for contradiction detection where accuracy matters more than speed.
*Interview Tip: Bi-encoder = fast retrieval, Cross-encoder = accurate reranking/classification. Classic interview question.*

**Q13. Why did you choose DeBERTa-v3 specifically for NLI?**
**Answer:**
DeBERTa-v3 (Decoding-enhanced BERT with disentangled attention v3) consistently achieves state-of-the-art performance on NLI benchmarks (MNLI, RTE). Key improvements over BERT: disentangled attention mechanism (content and position encoded separately — better at capturing relative positions), enhanced mask decoder, virtual adversarial training for better generalization. The `cross-encoder/nli-deberta-v3-base` from Sentence Transformers is specifically fine-tuned on SNLI+MultiNLI for NLI classification — making it directly applicable to my contradiction detection task without additional fine-tuning.

**Q14. Why not just use the main LLM (Llama 3.3 70B) for contradiction detection?**
**Answer:**
Three reasons: (1) Reliability — LLMs sometimes hallucinate contradictions that don't exist, or miss real ones. A dedicated NLI model fine-tuned specifically on entailment classification is more reliable for this boolean task. (2) Cost — running a 70B model inference for every pairwise claim comparison across 8 quarters of filings would be prohibitively expensive. The DeBERTa-v3-base is a small model (~180M parameters) run locally at near-zero cost. (3) Speed — a cross-encoder inference takes milliseconds; a 70B LLM call takes seconds. For pairwise evaluation of many claims, this matters significantly.
*Interview Tip: This is one of the most likely deep-dive questions. You have a strong 3-part answer: reliability, cost, speed.*

**Q15. How does pairwise contradiction detection work in your system?**
**Answer:**
After the LLM synthesizes an answer using chunks from multiple periods, the system extracts the key claims (factual statements) from each period's retrieved chunks. It then runs DeBERTa-v3 cross-encoder on pairs of claims from different time periods: (claim from Q1, claim from Q3) → [entailment, neutral, contradiction] with a confidence score. If the label is 'contradiction' and confidence exceeds a threshold, a `ContradictionEvent` dataclass is created and surfaced to the user with the conflicting statements highlighted. This is automated — no human needs to compare filings manually.
*Interview Tip: Walk through a concrete example: 'Q1: supply chain risks are minimal' vs 'Q3: supply chain disruptions are our primary risk.'*

**Q16. What is the difference between entailment, neutral, and contradiction in your system's context?**
**Answer:**
In the SEC context: Entailment means the company's guidance is consistent — Q3 statement is aligned with Q1 (e.g., both describe strong revenue growth). Neutral means the statements are on different topics or simply don't address each other — not informative for contradiction detection. Contradiction is the signal we care about — management's Q3 statement fundamentally conflicts with Q1 (e.g., Q1: 'We see no material cybersecurity risks' vs Q3: 'We experienced a significant cybersecurity incident'). Contradiction events are what get flagged and shown to the analyst as a signal of guidance walk-back.

**Q17. Why is contradiction detection valuable for financial analysis?**
**Answer:**
Management commentary in SEC filings is often strategically worded. Companies sometimes quietly reverse prior guidance without explicitly saying so — downplaying risks that were previously highlighted, or vice versa. Manually comparing hundreds of pages across 8 quarters for 20 companies is impractical. Automated NLI-based detection surfaces these inconsistencies instantly. For analysts: this is an early warning signal — if a company contradicts itself between Q1 and Q3, it may indicate deteriorating conditions being downplayed. This is a capability financial data vendors charge thousands of dollars for.

---

### — VECTOR DB & EMBEDDINGS —

**Q18. What is ChromaDB and why did you choose it for development?**
**Answer:**
ChromaDB is an open-source, embedded vector database designed for AI applications. Zero setup — runs in-process with Python, no separate server needed. Perfect for local development and prototyping. Stores embeddings + metadata + documents together. Supports metadata filtering (essential for hop-based retrieval by company/period). Persistent storage to disk. Why for dev: zero infrastructure overhead, fast iteration, no Docker or cloud account needed. For production I switched to Qdrant Cloud for better scalability, performance, and cloud-native operations.

**Q19. What is the difference between ChromaDB and Qdrant?**
**Answer:**
ChromaDB: embedded (runs in-process), Python-native, minimal setup, great for prototyping and small datasets. Limited in performance and scalability. Qdrant: standalone server (Rust-based — very fast), cloud-native, production-grade with distributed deployment, advanced filtering, payload indexing, and quantization for memory efficiency. Supports gRPC for lower latency. My architecture abstracts the vector store behind an interface — I can switch between ChromaDB locally and Qdrant Cloud in production by changing a config variable, not code.
*Interview Tip: The key point: same interface, swappable backends. This is clean software design.*

**Q20. Why did you choose all-MiniLM-L6-v2 as your embedding model?**
**Answer:**
`all-MiniLM-L6-v2` is a sentence-transformer model that maps text to 384-dimensional vectors. Reasons for choosing it: Extremely lightweight (22M parameters) — fast inference, low memory, cheap to run at scale across 160+ large filings. Despite its small size, it performs well on semantic similarity tasks — sufficient accuracy for chunk retrieval. Runs locally (no API calls, no cost per embedding). The tradeoff: it's less powerful than OpenAI's `text-embedding-3-large` or `E5-large`, but for the retrieval step (finding topically relevant chunks), it's more than adequate. Accuracy-critical tasks (contradiction detection) use a dedicated cross-encoder.

**Q21. How did you implement metadata filtering in ChromaDB for multi-hop retrieval?**
**Answer:**
Every chunk stored in ChromaDB has associated metadata: company ticker, filing period (e.g. '2023-Q2'), filing type ('10-K' or '10-Q'), SEC section ('Item 1A', 'MD&A'), and chunk index. When a `HopSpec` is dispatched, the ChromaDB query includes both the query vector AND a where filter: `{'company': 'AAPL', 'period': '2023-Q2'}`. This means the vector similarity search only considers chunks matching that metadata — you get the most semantically relevant chunks from that specific filing, not from the entire corpus. Without metadata filtering, a query about Q1 risk factors might return the most similar text from Q3.

**Q22. What is the difference between FAISS and ChromaDB?**
**Answer:**
FAISS (Facebook AI Similarity Search): pure vector similarity search library — extremely fast, supports billion-scale vectors, but no built-in metadata storage, no persistence layer, no server. You manage storage yourself. ChromaDB: full vector database — stores vectors + documents + metadata together, built-in persistence, metadata filtering, Python API. Built on top of FAISS/HNSW internally. For my Tech Support Agent (internship), FAISS was sufficient — just needed fast retrieval. For the SEC system with complex metadata filtering (by company/period), ChromaDB was better suited.

---

### — PDF PARSING & CHUNKING —

**Q23. How did you parse SEC filings from PDF?**
**Answer:**
Used `pdfplumber` (Python library) which handles complex PDF layouts, tables, and multi-column text better than PyPDF2. Pipeline: download PDF from EDGAR → pdfplumber extracts raw text page by page → detect image-only pages (PDFs where text extraction returns empty strings — scanned documents with no selectable text) and handle them separately → clean the extracted text (remove headers/footers/page numbers) → pass to the SEC section chunker.
*Interview Tip: Be ready to explain why you chose pdfplumber over alternatives.*

**Q24. How did you identify and segment SEC sections (Item 1A, MD&A, etc.)?**
**Answer:**
SEC filings follow a standardized structure defined by the SEC. I wrote a custom regex-based SEC Section Chunker that identifies section headers: patterns like `r'Item\s+1A\.?\s+Risk Factors'` or `r'Management.{0,20}Discussion'`. The chunker scans the extracted text, identifies section boundaries, and splits the document into section-labeled segments. This ensures chunks are semantically coherent (all about risk factors, or all about MD&A) rather than arbitrary fixed-size windows. Semantic chunking at section level significantly improves retrieval precision.
*Interview Tip: Contrast with naive fixed-size chunking — explain why section-aware chunking is better for structured documents.*

**Q25. Why did you use custom parsing instead of Unstructured.io or LlamaParse?**
**Answer:**
Two reasons: (1) Cost — managed parsing services charge per page or per document. At 160+ filings with hundreds of pages each, costs would be significant. Custom parsing with `pdfplumber` is free. (2) Control — SEC filings have a known, standardized structure (all 10-Ks have Item 1, Item 1A, Item 7, etc.). A custom regex chunker exploits this structure perfectly — I can guarantee that 'Item 1A' chunks contain only risk factor content. Managed services are general-purpose and may not respect SEC-specific section boundaries. The tradeoff: more dev time, but better result for this specific document type.

**Q26. How did you handle image-only PDFs (scanned documents)?**
**Answer:**
Some older SEC filings are scanned images stored as PDFs with no selectable text. Detection: after `pdfplumber` extraction, if the text length is below a threshold (e.g. less than 100 characters per page on average), the document is flagged as image-only. Resolution options: (1) Skip with a warning logged to the user (current approach), (2) Use OCR (Tesseract/pytesseract) to extract text from scanned pages. For the current system, most modern filings are text-based — image-only is rare. The deduplication logic also prevents reprocessing known image-only filings.

**Q27. What is chunk size and overlap, and how did you choose them for SEC filings?**
**Answer:**
Chunk size determines how much text each vector represents. Too large: embeddings are diluted — a chunk covering 3 topics will match queries about all 3 poorly. Too small: chunks lose context — a single sentence about 'debt covenants' without surrounding context is ambiguous. For SEC filings, which have long, dense paragraphs with dense financial reasoning, I used larger chunks (~500-800 tokens) with ~100 token overlap. The overlap ensures a statement split across a chunk boundary doesn't lose context. Section-level chunking (Item 1A as a unit) was also used to preserve semantic coherence for broad section-level queries.

---

### — RAGAS EVALUATION —

**Q28. What is RAGAS and what metrics does it measure?**
**Answer:**
RAGAS (Retrieval Augmented Generation Assessment) is an evaluation framework specifically designed for RAG systems. Key metrics: Faithfulness — does the answer contain only information grounded in the retrieved context? (measures hallucination). Answer Relevance — is the answer relevant to the question? Context Precision — are the retrieved chunks actually relevant to the question? Context Recall — does the retrieved context cover all the ground truth answer information? My targets: Faithfulness ≥0.85 (answer is grounded), Context Precision ≥0.88 (retrieval is precise). Evaluated on a 200-question benchmark.
*Interview Tip: Know all 4 RAGAS metrics by name and definition. Very likely to be asked.*

**Q29. Why is Faithfulness an important metric for your system?**
**Answer:**
Faithfulness measures whether the LLM's answer is grounded in the retrieved context — detecting hallucination. In a financial analysis system, hallucination is dangerous: if the LLM fabricates a risk factor or financial figure not present in the actual SEC filing, an analyst might make investment decisions based on false information. A Faithfulness score of 0.85+ means 85% of claims in the answer can be directly attributed to retrieved context. It's the most critical metric for a system handling financial documents where factual accuracy is non-negotiable.

**Q30. What is Context Precision and why did you target ≥0.88?**
**Answer:**
Context Precision measures the fraction of retrieved chunks that are actually relevant to answering the question. High precision = retrieval is tight and relevant, not noisy. If precision is low, the LLM is getting lots of irrelevant chunks — wasting context window space and increasing chances of confusion or hallucination. I targeted ≥0.88 because SEC filings are dense — a query about 'liquidity risk' should retrieve liquidity-related chunks, not general business description sections. High context precision was achievable because of section-level metadata filtering — retrieval was constrained to semantically coherent document sections.

**Q31. How did you build the 200-question benchmark for RAGAS evaluation?**
**Answer:**
RAGAS provides a test set generation utility that uses an LLM to automatically generate question-answer-context triplets from the corpus documents. I generated questions across different categories: simple factual (single document), temporal comparison (multi-hop across periods), and contradiction-detection questions. For each question: the ground truth answer was generated by the LLM from the source chunks, forming the reference. The system then answered each question, and RAGAS compared the system answer against the ground truth for each metric. 200 questions provided statistically meaningful coverage across all 20 companies.

**Q32. How would you improve the system's RAGAS scores further?**
**Answer:**
Faithfulness: implement answer attribution — post-process the LLM's answer to cite specific chunks, use a faithfulness-focused prompt that explicitly instructs the model to only use provided context. Context Precision: improve the retrieval step with a reranking stage — after initial vector retrieval, use a cross-encoder reranker (like `ms-marco-MiniLM`) to rerank chunks by relevance, keeping only the top-k most relevant. Context Recall: increase the number of retrieved chunks per hop (top-k) to ensure better coverage. Also: fine-tune the embedding model on financial domain text for better semantic matching.

---

### — TECH STACK DECISIONS —

**Q33. Why did you use Llama 3.3 70B via Groq instead of GPT-4o or Claude?**
**Answer:**
Three reasons: (1) Cost — OpenAI and Anthropic charge per token. SEC filings are very long; processing 160+ filings with a proprietary model would cost significantly more. Groq + Llama 3 is essentially free at my scale. (2) Speed — Groq's LPU (Language Processing Unit) delivers extremely low-latency inference even for 70B models — competitive with or faster than GPT-4 Turbo for simple queries. (3) Capability — Llama 3.3 70B is competitive with GPT-4o on reasoning benchmarks and handles financial jargon well. The cost-to-performance ratio clearly favored open-source + Groq for an MVP.
*Interview Tip: Never say 'GPT-4 was too expensive' without explaining what you got instead. Show you evaluated the tradeoff.*

**Q34. What is Groq and what makes it different from standard LLM APIs?**
**Answer:**
Groq is an AI inference company that built a custom LPU (Language Processing Unit) chip specifically optimized for LLM inference — not a GPU. LPUs eliminate the memory bandwidth bottleneck that makes GPU-based LLM inference slow. Result: Groq serves models like Llama 3.3 70B at 500-800 tokens/second — roughly 10-20x faster than typical GPU-hosted endpoints. For a user-facing application where sub-15 second response time is a requirement, Groq's low latency is critical. I use it as the primary LLM with Llama 3.1 8B as a fallback for simpler or faster queries.

**Q35. Why did you use FastAPI as the backend?**
**Answer:**
FastAPI is a modern Python web framework with automatic request/response validation via Pydantic (aligns well with my dataclass-heavy architecture), native async support (important for parallel hop execution), auto-generated OpenAPI docs, and high performance. For this system specifically: the parallel hop execution benefits from async — multiple ChromaDB queries dispatched concurrently using `asyncio.gather`. Synchronous frameworks (Flask) would execute hops sequentially, significantly increasing latency. FastAPI's type system also integrates cleanly with Python dataclasses.

**Q36. Why use Streamlit for the frontend instead of React/Next.js?**
**Answer:**
For an analytical MVP: Streamlit allows building interactive dashboards in pure Python — no HTML/CSS/JS required. A data analyst using this tool doesn't need a polished SPA; they need clear query input, answer display, and contradiction event visualization. Streamlit provides all of this with ~50 lines of Python vs hundreds of lines of React. The entire stack stays Python — no context switching between languages. Tradeoff: less customizable UI, not suitable for a production consumer product. For this research/analytics tool, Streamlit + FastAPI was the right balance of speed-to-market vs feature quality.

**Q37. What is SQLite used for in this system? Why switch to PostgreSQL in production?**
**Answer:**
SQLite stores the relational metadata: filing registry (company, period, filing type, download URL, processing status), filing-to-chunk mappings, and system state. In development: SQLite is file-based — zero setup, no server, no credentials. Ideal for local development. In production: PostgreSQL handles concurrent writes from multiple workers, has better performance at scale, supports more advanced indexing, and is the standard for production cloud deployments (AWS RDS, Render). The database layer is abstracted behind an interface — same query code works for both SQLite and PostgreSQL.

**Q38. What is pdfplumber and why did you choose it over PyPDF2 or PyMuPDF?**
**Answer:**
`pdfplumber` is built on `pdfminer` and specializes in extracting structured content — text with positional information, tables with cell boundaries, and layout-aware extraction. PyPDF2: simpler but poor at complex layouts, tables, and multi-column text common in SEC filings. PyMuPDF (`fitz`): faster and handles more PDF types, but less Pythonic API. For SEC filings which often contain financial tables and multi-column layouts, `pdfplumber`'s table extraction and layout awareness give cleaner text output. The final pinned version (`pdfplumber==0.11.9`) was chosen after resolving CI compatibility issues.

---

### — TESTING & CI/CD —

**Q39. What is property-based testing and why did you use Hypothesis?**
**Answer:**
Traditional unit tests test specific known inputs/outputs. Property-based testing (Hypothesis library) automatically generates hundreds of random inputs and verifies that logical properties (invariants) always hold — regardless of input. Why needed here: the hop planner and temporal resolver must handle arbitrary date ranges, filing counts, and company combinations correctly. A traditional test for 'last 4 quarters' with one example might pass but fail for edge cases (company with only 2 filings, leap years, missing quarters). Hypothesis discovers these edge cases automatically. For a financial system where correctness guarantees matter, this provides much stronger reliability assurance than example-based tests.
*Interview Tip: Hypothesis is niche — explaining it clearly shows engineering depth.*

**Q40. What CI/CD issues did you encounter and how did you fix them?**
**Answer:**
Three issues: (1) `pdfplumber` and `qdrant-client` version conflicts — fixed by strictly pinning all dependencies in `requirements.txt` (`pdfplumber==0.11.9`, `qdrant-client==1.13.3`) after testing locally. (2) PyTorch CPU wheel failing to resolve — fixed by adding the PyTorch extra-index-url (`--extra-index-url https://download.pytorch.org/whl/cpu`) to pip install commands, ensuring the lightweight CPU-only wheel is fetched instead of the GPU version. (3) Python 3.14 set in GitHub Actions — dropped back to Python 3.11 since 3.14 was a pre-release unavailable on standard Ubuntu runners, and 3.11 cleanly supports all required ML libraries.
*Interview Tip: Real CI/CD debugging experience is very impressive in an interview. Walk through each root cause clearly.*

**Q41. Why pin exact dependency versions in requirements.txt?**
**Answer:**
In ML systems, unpinned dependencies are a major source of bugs: a new version of a library can change behavior silently. Examples: `sentence-transformers` changing default pooling methods, `chromadb` changing its API, `numpy` changing default dtypes. Pinned versions guarantee reproducibility — the same environment on every machine, every CI run, every deployment. The tradeoff: you must manually update pins to get security patches. Best practice: pin everything in `requirements.txt` for applications; use loose constraints only for libraries. I pinned after discovering that `qdrant-client` v1.9.1 conflicted with other dependencies — the new pin (1.13.3) was verified against the test suite.

**Q42. How did you implement the deduplication and retry logic for EDGAR ingestion?**
**Answer:**
Deduplication: before downloading a filing, check the SQLite metadata store for a record with the same `company+period+filing_type` combination. If a record exists with `status='completed'`, skip the download entirely. This prevents re-downloading 160 files every time the pipeline runs. Retry with backoff: the EDGAR API rate-limits requests. I implemented exponential backoff — on HTTP 429 or 5xx errors, wait 2^attempt seconds before retrying (1s, 2s, 4s, 8s...) up to a max of 5 retries. After max retries, log the failure and mark the filing as 'failed' in the metadata store for later review.

---

### — DEPLOYMENT —

**Q43. How did you deploy the system at $0/month?**
**Answer:**
FastAPI backend: deployed on Render's free tier — provides a persistent web service with 512MB RAM. Cold starts on free tier (~30 seconds on first request) are acceptable for a research tool. Streamlit frontend: deployed on Streamlit Community Cloud (free) — connects to the FastAPI backend. ChromaDB: for production demo, persisted to Render's ephemeral disk (acceptable for demo; for real production, Qdrant Cloud has a free tier). SQLite metadata: also on Render's disk. LLM: Groq API (free tier — 14,400 requests/day). Embeddings: `sentence-transformers` run locally on the server CPU. Total: $0/month for an MVP serving real queries.
*Interview Tip: The $0/month figure is memorable. Know exactly how each component is free.*

**Q44. What are the limitations of the current $0 deployment and how would you productionize it?**
**Answer:**
Current limitations: Render free tier has cold starts and limited RAM — not suitable for concurrent users or large filings. SQLite doesn't handle concurrent writes. ChromaDB on ephemeral disk loses data on service restart. Production improvements: (1) Move to PostgreSQL (Render paid tier or AWS RDS) for reliable metadata storage. (2) Use Qdrant Cloud (free tier supports 1GB) for persistent vector storage. (3) Add a job queue (Celery + Redis) for async ingestion — don't block the API during filing download/processing. (4) Add Redis caching for repeated queries. (5) Containerize with Docker for consistent deployments. (6) Add authentication for the API.

**Q45. How would you containerize this system with Docker?**
**Answer:**
Multi-container Docker Compose setup: (1) `api` service: Dockerfile with `python:3.11-slim`, install requirements, copy source, uvicorn FastAPI on port 8000. (2) `streamlit` service: Dockerfile for Streamlit UI, connects to api. (3) `qdrant` service: `qdrant/qdrant` official Docker image. (4) `postgres` service: `postgres:15` official image with volume for data persistence. Networks: all services on a shared bridge network. Volumes: named volumes for postgres data and qdrant storage. The current dev setup (SQLite + ChromaDB) becomes production-ready (PostgreSQL + Qdrant) just by switching Docker Compose profiles.

---

### — EDGE CASES & IMPROVEMENTS —

**Q46. What were the hardest engineering challenges in this project?**
**Answer:**
Three main challenges: (1) Temporal query parsing — resolving 'last 4 quarters' to specific filing IDs reliably across different companies with different filing histories required careful design of the hop planner and thorough property-based testing. (2) CI/CD pipeline stability — PyTorch CPU wheel resolution and Python version compatibility issues took significant debugging time. The lesson: always pin dependencies and test CI from the start, not at the end. (3) Keeping response time under 15 seconds despite multi-hop retrieval — solved by embedding once and running parallel metadata-filtered searches rather than sequential hops.

**Q47. What would you add to the system if you had more time?**
**Answer:**
High priority: (1) Fine-tune `all-MiniLM-L6-v2` on SEC financial text — domain adaptation would significantly improve retrieval quality for financial jargon. (2) Add a cross-encoder reranker between initial retrieval and LLM synthesis — improves context precision. (3) Structured data extraction — extract financial tables (revenue, EPS, guidance) into PostgreSQL for precise numerical queries alongside semantic RAG. (4) Multi-company comparison — 'Compare Apple and Microsoft's Q2 2023 risk factors' requires cross-company multi-hop. (5) Alert system — notify analysts when a contradiction is detected in a new filing automatically.

**Q48. How does your system handle a company with fewer than 4 quarters of data when asked 'last 4 quarters'?**
**Answer:**
The hop planner queries the metadata store for available filings for that company. If only 2 quarters exist, it returns 2 `HopSpecs` (not 4) and proceeds with what's available. The LLM synthesis prompt is aware of the number of hops executed and explicitly states which periods were covered in the response — so the user knows the answer is based on 2 quarters, not 4. The system never silently fabricates missing data. This edge case was specifically tested with Hypothesis by generating companies with randomly sized filing histories (1 to N quarters).
*Interview Tip: Handling edge cases gracefully and communicating limitations to users is a sign of production maturity.*

**Q49. How would you handle a query that doesn't involve temporal comparison?**
**Answer:**
The Query Classifier identifies whether a query has temporal references. If no temporal reference is detected ('What are Apple's main risk factors?' with no time reference), the hop planner generates a single `HopSpec` targeting the most recent available filing. The system falls back to standard single-hop RAG for these queries — no unnecessary multi-hop overhead. This classification is important for both performance (avoid unnecessary parallel searches) and answer quality (don't confuse the LLM with multi-period context when a single period is sufficient).

**Q50. How would you scale this system to 1000 companies instead of 20?**
**Answer:**
Ingestion: parallelize EDGAR downloads with async workers (Celery), respect rate limits per company. Storage: migrate from SQLite to PostgreSQL with proper indexing on company+period+filing_type. Vector store: Qdrant Cloud scales horizontally — add more nodes, use collection sharding by company. Embeddings: batch-embed chunks using `sentence-transformers`' batch inference API instead of one-by-one. Contradiction detection: queue pairwise NLI jobs asynchronously — results stored in DB, surfaced when ready instead of blocking the query response. The modular architecture (swappable DB and vector store interfaces) means most of this is configuration, not code changes.
