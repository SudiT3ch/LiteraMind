# LiteraMind — Development Progress (final)

_Only items marked "tested" were executed. **Sandbox limitation:** outbound network was restricted (arXiv / Semantic Scholar / OpenAlex / Crossref / HuggingFace / PyPI-only egress), so live API calls, Sentence-Transformers model download, real LLM calls and cloud deployment were **not** verified; those paths are covered by format fixtures, fake providers and mocks._

| Phase | Status | Verification actually performed |
|---|---|---|
| 1 Setup, config, DB (12 tables), logging w/ secret redaction | DONE | DB smoke + all tests |
| 2 Academic APIs (arXiv, S2, OpenAlex, Crossref) + de-dup | DONE (parsers/errors tested; **live calls not verified**) | parser fixtures, 429/timeout/network mocks; live attempt returned graceful warnings |
| 3 PDF extraction + chunking | DONE | real generated PDFs; invalid/empty/oversize/truncated cases |
| 4 Embeddings + ChromaDB | DONE (hashing embedder tested; **Sentence-Transformers path untested** – model download blocked) | CRUD/upsert/filter tests |
| 5 Hybrid retrieval (dense + BM25 + RRF + rerank) | DONE | mode tests, traceability, quality sanity on fixtures |
| 6 LangGraph orchestration | DONE | full run, retries, ablation, expansion loop, failure isolation |
| 7 Six agents | DONE (rule-based path fully tested; LLM paths tested with **fake providers only**) | unit + workflow tests |
| 8 Comparison | DONE | comparable only when metric+dataset match |
| 9 Research gaps | DONE | evidence/strength/caveat assertions |
| 10 Verification | DONE | statuses, missing citation, conflicting evidence, fault injection |
| 11 Backend + frontend | DONE | 19 API endpoints tested via TestClient; 8 Streamlit pages via AppTest against a live uvicorn server; `python run.py` start-up probed (API + UI health OK) |
| 12 Reports (MD/PDF/DOCX) + knowledge graph | DONE | PDF reopened with PyMuPDF, DOCX reopened with python-docx |
| 13 Testing / debugging | DONE | **69 tests passed** (`pytest tests -q`) |
| 14 Baseline vs multi-agent, ablation, metrics | DONE offline on **synthetic fixture corpus**; real benchmarks **PENDING** | `python -m evaluation.experiments` → `evaluation/results/` |
| 15 Docs + deployment files | DONE (Dockerfile/render.yaml **not deploy-tested**) | README, docs/*.md (API & schema generated from code) |

## Bugs found by testing and fixed
1. LLM analyst dropped valid quotes when the model cited the wrong chunk index → now searches all retrieved chunks.
2. Chroma `EphemeralClient` leaked state across stores → unique collection per ephemeral store.
3. API startup re-initialised the DB over the test DB → idempotent init.
4. **Verifier accepted a false claim** ("accuracy 90%" vs "drops from 90% to 83%"), found by fault injection (4/5 detected) → metric-aware numeric check; re-measured 5/5 detected, 5/5 true claims accepted (tiny synthetic sample).

## Pending / not done
* SciTLDR & QASPER evaluation (need network + `pip install datasets`); human evaluation (template generated); LLM-mode experiments and token-cost comparison; RAGAS integration (needs an LLM judge).
* Live verification of arXiv/S2/OpenAlex/Crossref, Sentence-Transformers, cross-encoder reranker, Docker/Render deployment.
* Authentication/`users` table (optional in spec); OCR for scanned PDFs; spaCy/NLTK (not needed by the current design).

## Results caveats (see evaluation/results/results.md)
Synthetic corpus + author-made gold labels → sanity checks only. Baseline citation-support 1.0 is tautological for an extractive baseline. BM25 beat hybrid on P@5; baseline ROUGE-L (0.493) ≈ multi-agent (0.486).
