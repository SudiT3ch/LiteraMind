# LiteraMind

**An Evidence-Grounded Multi-Agent AI Framework for Automated Scientific Literature Analysis and Research Gap Identification**

Ask a research question → LiteraMind plans the search, discovers papers (arXiv · Semantic Scholar · OpenAlex), indexes full text, extracts methods/datasets/metrics/limitations, compares papers, proposes **candidate** research gaps, **checks every generated claim against source passages**, and writes a downloadable literature review (PDF / DOCX / Markdown).

> **Honesty notes (read before demoing).** (1) Without an LLM key the system runs in a transparent *rule-based* mode (regex/cue-phrase extraction, lexical verification). It is fully functional but less nuanced than LLM mode. (2) Live academic APIs, Sentence-Transformers model download, real LLM calls and cloud deployment could **not** be exercised in the development sandbox (outbound network was restricted) — those paths are covered by format fixtures, fake providers and mocks only. (3) Reported experiment numbers come from a synthetic fixture corpus — see [Evaluation](#evaluation). Details: [Known limitations](#known-limitations).

---
## 1. Problem statement
Researchers spend long hours discovering papers, extracting methodology/results, comparing approaches, spotting limitations and writing reviews. Search engines retrieve documents; plain RAG chatbots answer questions. Neither produces *structured, cross-paper, verified* analysis.

## 2. Objectives
1. Turn a research question into a search plan and ranked, de-duplicated papers.  2. Extract structured findings with source location (paper, page, section).  3. Compare papers without mixing incomparable metrics.  4. Identify recurring limitations and *candidate* gaps with evidence strength.  5. Verify claims against source text and flag unsupported ones.  6. Generate a cited review.  7. Measure all of this (baseline vs multi-agent, ablations).

## 3. Key features & unique contributions
* **Six-agent LangGraph workflow** with shared structured state, retries, bounded expansion loop, DB-logged execution (the UI shows real status).
* **Evidence-first design**: every finding keeps `paper_id + page + section + passage`; LLM output is accepted only if its quotes occur verbatim in the retrieved text.
* **Metric-safe comparison**: numbers compared only for identical metric *and* dataset.
* **Candidate gaps, never "novel" claims**: Strong/Moderate/Weak evidence + mandatory coverage caveat.
* **Verifier with metric-aware numeric check, negation check, missing-citation and conflicting-evidence detection** (fault-injection tested).
* Hybrid retrieval (ChromaDB dense + BM25 → RRF → rerank), knowledge graph (NetworkX/Plotly), PDF/DOCX/MD reports, baseline RAG + ablation framework.

## 4. Architecture
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (diagram, orchestration rules, agent table, security design).

```
Query → Planner → Discovery → Retrieval → Analyst → Comparison → Gap → Verifier → (expand?) → Report → Dashboard
```

| Agent | Responsibility |
|---|---|
| 1 Research Planner | domain, keywords, related terms, queries, information needs |
| 2 Paper Discovery | multi-source search, de-dup, relevance ranking, metadata storage |
| 3 Research Analyst | problem, objectives, methods, algorithms, datasets, setup, metrics, results, limitations, future work |
| 4 Comparison | comparison matrix, comparable-metric groups, similarities/differences, trends, divergent findings |
| 5 Research Gap | recurring limitations, dataset concentration, method imbalance, sparse coverage → candidate gaps |
| 6 Evidence Verification | Supported / Partially supported / Contradicted / Insufficient evidence; DOI check (optional) |

## 5. Technology stack
Python 3.10+ · Streamlit · FastAPI · LangGraph · SQLAlchemy (SQLite → PostgreSQL) · ChromaDB · PyMuPDF · Sentence-Transformers (optional; hashing fallback) · NetworkX · Plotly · ReportLab · python-docx · rouge-score · PyTest. *spaCy/NLTK are not used (not needed; the lexicon/regex layer covers the required NLP). RAGAS is not integrated (needs an LLM judge) — custom metrics are used.*

## 6. Folder structure
```
LiteraMind/
├── frontend/            app.py (Dashboard) · common.py · pages/1..7 (Discovery, Documents, Analysis, Comparison, Gaps, Verification, Report)
├── backend/             main.py · routes/{projects,documents,system}.py
├── agents/              planner · discovery · analyst · comparison · gap_identifier · verifier · lexicon · base
├── orchestration/       state.py · graph.py (LangGraph) · workflow.py
├── retrieval/           embeddings · vector_store (Chroma) · retriever (hybrid) · reranker
├── document_processing/ pdf_extractor · chunker · ingest
├── academic_sources/    arxiv_client · semantic_scholar · openalex · crossref · aggregator · base
├── llm/                 provider.py (anthropic | openai | ollama | heuristic)
├── evaluation/          retrieval/citation/summarization metrics · baseline · datasets · experiments · fixtures · results
├── reports/             generator.py (MD/PDF/DOCX)
├── database/            models.py · database.py · repository.py
├── visualization/       knowledge_graph.py · plotting.py
├── tests/               69 tests · docs/ API.md DATABASE.md ARCHITECTURE.md EVALUATION.md DEPLOYMENT.md
└── run.py · config.py · requirements.txt · .env.example · Dockerfile · render.yaml · PROGRESS.md
```

## 7. Data sources
| Type | Source | Use |
|---|---|---|
| Live publications | arXiv API, Semantic Scholar Graph API, OpenAlex, Crossref (DOI validation) | discovery & metadata (no fabrication; missing fields stay empty) |
| User documents | uploaded PDFs | analysed like discovered papers |
| Evaluation (real) | QASPER, SciTLDR (HF `datasets`), S2ORC / ACL Anthology (reference) | loaders provided, **pending** (network) |
| Evaluation (offline) | synthetic fixture corpus (6 fictional documents) | regression/sanity only — never shown as real papers |
Please check each API's current terms/rate limits; set `CONTACT_EMAIL` for OpenAlex/Crossref polite pools and `S2_API_KEY` for higher Semantic Scholar limits.

## 8. Installation
```bash
git clone <your-repo> && cd LiteraMind
python -m venv venv && source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Lightweight install (no PyTorch): delete the `sentence-transformers` line and set `EMBEDDING_BACKEND=hashing`.

## 9. Configuration (`.env`, copy from `.env.example`)
| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `heuristic` (default, no key) · `anthropic` · `openai` (any OpenAI-compatible URL) · `ollama` (local) |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `LLM_MODEL` | needed only for the matching provider (falls back to heuristic if missing) |
| `S2_API_KEY`, `CONTACT_EMAIL`, `ENABLED_SOURCES` | optional academic API settings |
| `EMBEDDING_BACKEND` | `auto` (try Sentence-Transformers, fall back) · `sentence-transformers` · `hashing` |
| `DATABASE_URL` | default SQLite file in `data/`; PostgreSQL URL supported |
| `MAX_UPLOAD_MB`, `MAX_PAPERS`, `MIN_PAPERS`, `MIN_SUPPORTED_RATIO`, … | limits and workflow thresholds (see `config.py`) |
**Requires an external key:** LLM modes (Anthropic/OpenAI). **No key needed:** arXiv, OpenAlex, Crossref, Semantic Scholar (low limits), PDF/RAG/embeddings, rule-based agents.

## 10. Run
```bash
python run.py                  # API http://127.0.0.1:8000 (docs at /docs) + UI http://localhost:8501
python run.py --backend-only   # or --frontend-only, --api-port, --ui-port
```
Manual: `uvicorn backend.main:app` and `streamlit run frontend/app.py` (set `BACKEND_URL` if not default).

## 11. Using the app
1. **Dashboard** – enter a research question → project created, planner + discovery run.  2. **Research Discovery** – inspect ranked papers (relevance, DOI/source/PDF links, abstracts), tick papers, *Analyse selected* (human review step).  3. **Document Management** – upload your own PDFs, inspect extracted text with page/section tags, delete.  4. **Multi-Agent Analysis** – start/ablate agents, watch real per-agent status, warnings, outputs, findings table.  5. **Paper Comparison** – matrix, comparable-metric charts, similarities/differences/trends, knowledge graph.  6. **Research Gap Analysis** – candidate gaps with evidence, strength, directions, caveat.  7. **Evidence Verification** – claims, status, passage, page/section, conflicting evidence.  8. **Generated Report** – tabs + PDF/DOCX/MD download.
API usage: [docs/API.md](docs/API.md). Schema: [docs/DATABASE.md](docs/DATABASE.md). Deployment: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## 12. Testing
```bash
pytest tests -q                       # 69 tests, ~25 s, offline
python -m evaluation.experiments      # experiments -> evaluation/results/
```
Covers: PDF extraction/invalid/empty/oversize/truncated PDFs, chunking & metadata, embeddings, Chroma CRUD, retrieval modes, API clients (parsers, de-dup, 429/timeouts, graceful degradation), all six agents (incl. fake-LLM hallucination rejection), workflow (full run, ablation, no-papers/insufficient evidence, agent failure+retry with secret redaction, two-phase review), FastAPI endpoints & upload restrictions, report PDF/DOCX validity, LLM missing-key/timeouts/HTTP errors, evaluation metrics, and all 8 Streamlit pages via `AppTest` against a live server.

## 13. Evaluation
Methodology: [docs/EVALUATION.md](docs/EVALUATION.md). Latest committed run (`evaluation/results/results.md`; synthetic fixture corpus, rule-based mode, hashing embedder — **sanity check, not a benchmark**):

| | citation support | unsupported claims | evidence coverage | analysis completeness | comparison F1 | gap-theme F1 |
|---|---|---|---|---|---|---|
| Baseline (single-pass extractive RAG) | 1.00 † | 0.00 † | 0.75 | 0.23 | 0.00 | 0.00 (none produced) |
| Multi-agent (full) | 0.86 | 0.11 | 1.00 | 0.90 | 1.00 | 0.67 |

† tautological for an extractive baseline (its claims *are* retrieved sentences). Ablation (full vs −Comparison/−Gap/−Verification) and a verifier fault-injection test (5/5 numerically-altered claims flagged, 5/5 true claims accepted — tiny sample) are in `results.md`. Also measured and **not favourable to the proposed system**: baseline ROUGE-L (0.493) ≈ multi-agent summary (0.486); BM25 alone beat hybrid on P@5. SciTLDR/QASPER, LLM-mode runs and human evaluation are **PENDING**.

## 14. Known limitations
* Rule-based mode is cue-phrase driven: it misses implicit statements; section detection depends on standard headings; scanned PDFs (no OCR) are rejected as "no extractable text".
* Verification is lexical entailment (+ numeric/negation rules), not an NLI model; paraphrase-heavy claims can be marked *Insufficient evidence* (seen in the fixture run) — conservative by design.
* The hashing fallback embedder is lexical, not semantic; install Sentence-Transformers for real semantic retrieval (untested in the sandbox).
* Live arXiv/S2/OpenAlex/Crossref calls, real LLM calls, the cross-encoder reranker, Docker/Render deployment are implemented but **not verified live**.
* Uploaded PDFs get title from PDF metadata/first line; authors/year are not extracted (not invented).
* Candidate gaps reflect only the retrieved papers; they are prompts for investigation, not novelty claims. No authentication (single-user). Background runs use threads (single-process).
* Benchmarks on real datasets, human evaluation, and token/cost comparison in LLM mode are pending.

## 15. Future enhancements
NLI-based verifier (e.g., DeBERTa-MNLI) · OCR for scans · citation-graph expansion via Semantic Scholar references · PostgreSQL + job queue (Celery/RQ) · user accounts · Neo4j knowledge graph · RAGAS integration with a judge LLM · multilingual papers.

## 16. Screenshots
`docs/screenshots/` — _placeholders:_ Dashboard · Discovery · Analysis monitor · Comparison · Gap analysis · Verification · Report. (Capture with your own data; the sandbox had no browser.)

## 17. Author contribution
_B.Tech CSE final-year project — Author: ⟨your name⟩, Roll no. ⟨…⟩, Guide: ⟨…⟩. Replace with your contribution statement._
