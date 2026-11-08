# Architecture

```
                        ┌──────────────────────── Streamlit UI (8 pages) ────────────────────────┐
                        │  talks to the backend ONLY over HTTP (frontend/common.py)              │
                        └───────────────────────────────┬────────────────────────────────────────┘
                                                        │ REST /api
                        ┌───────────────────────────────▼────────────────────────────────────────┐
                        │ FastAPI  (backend/): projects · documents · runs · results · downloads │
                        └───────────────┬───────────────────────────────┬────────────────────────┘
                                        │                               │ background thread
                          ┌─────────────▼────────────┐        ┌─────────▼──────────────────────────────┐
                          │ SQLite/PostgreSQL (ORM)  │◄───────┤ LangGraph workflow (orchestration/)     │
                          │ 12 tables, AgentRun log  │        │ shared WorkflowState, retries, loops    │
                          └──────────────────────────┘        └─────────┬──────────────────────────────┘
                                                                        │
 START → Planner → Discovery → [expand?] → Retrieval(full text) → Analyst → Comparison → Gap → Verifier → [expand? | Report] → END
          (1)        (2)                        │                   (3)        (4)        (5)      (6)
                                                ▼
              PDF (PyMuPDF) → chunks (page/section/paper_id) → embeddings → ChromaDB ──► hybrid retriever (dense + BM25 → RRF → rerank)
```

## Orchestration (`orchestration/graph.py`)
* **State** (`state.py`): a `TypedDict`; agents exchange structured data (plan, candidates, analyses, findings, comparison, gaps, verifications), never free chat. `warnings`/`errors` use list-append reducers.
* **Node wrapper** (`make_node`): writes an `AgentRun` row (running → completed/failed/skipped) with attempts, duration, summary, warnings, output; retries once (`AGENT_RETRIES`); catches exceptions so a failing agent degrades the run instead of crashing it; honours ablation flags (`enable_comparison/gap/verification`). **UI status is read from these rows — nothing is simulated.**
* **Conditional edges**: planner failure → END; too few papers → one bounded **expansion** round (extra queries from plan concepts, +3 paper budget); after verification, if supported-claim ratio < `MIN_SUPPORTED_RATIO` → one expansion round; otherwise → report. If evidence is still insufficient the report says so explicitly. Loop caps (`MAX_EXPANSION_LOOPS`, LangGraph `recursion_limit`) prevent runaway loops.
* **Human review**: two-phase API — `discover` (planner + discovery) → user selects papers → `run` with the selection (`skip_search`).

## Agents
| # | Agent | Rule-based path (always available) | LLM path (optional) |
|---|---|---|---|
| 1 | Planner | stopword-aware phrase extraction, domain lexicon, synonym expansion, query generation | JSON plan |
| 2 | Discovery | arXiv/S2/OpenAlex search, DOI/arXiv/title de-dup, relevance = 0.5·dense + 0.5·BM25 + bonuses | — |
| 3 | Analyst | cue-phrase + section scoring over sentences of the paper's indexed chunks; lexicons for algorithms/datasets/metrics | one JSON call per paper; every `quote` must occur **verbatim** in retrieved text or the finding is dropped |
| 4 | Comparison | deterministic table; metrics comparable **only if metric and dataset match** | optional narrative (verified later by Agent 6) |
| 5 | Gap | recurring limitation themes, embedding clusters, dataset concentration, method imbalance, sparse concept coverage; Strong/Moderate/Weak by #papers; mandatory coverage caveat | rewrites "future direction" wording only |
| 6 | Verifier | re-retrieves from the cited paper; judge = stemmed content coverage + metric-aware numeric check + negation check → Supported / Partially / Contradicted / Insufficient; missing-citation handling; cross-paper conflicting evidence; optional Crossref DOI check | may revise non-Supported verdicts, only with evidence quoted from the passage |

## Security design
* Secrets only via env vars; `/api/config` never returns keys; logs and API errors pass through `redact()`.
* PDFs are untrusted: extension + `%PDF-` magic + streamed size cap + page cap + encrypted/corrupt rejection; filenames sanitised; instruction-like text is flagged in logs/UI.
* Prompt-injection hardening: document text is wrapped in `<document>` tags with a guard system prompt; LLM output is *validated against the source text* (quotes) rather than trusted; the rule-based agents never execute document text.

## Report generation
`reports/generator.py` assembles the review from stored findings with page/section locations and verification tags; no free-form LLM prose is added. Exporters: Markdown, PDF (ReportLab), DOCX (python-docx).
