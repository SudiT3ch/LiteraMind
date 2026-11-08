"""Agent 2 — Paper Discovery: multi-source search, de-duplication, relevance ranking, metadata storage."""
from __future__ import annotations
from .base import Agent
from academic_sources.aggregator import search_all
from academic_sources.base import PaperRecord
from retrieval.embeddings import get_embedder, cosine
from retrieval.retriever import BM25
from config import get_settings
from database import repository as repo


def rank_records(query: str, records: list[PaperRecord]) -> list[PaperRecord]:
    """Relevance = 0.5*dense cosine + 0.5*normalised BM25 over title+abstract, + small multi-source/abstract bonus."""
    if not records:
        return []
    emb = get_embedder()
    texts = [f"{r.title}. {r.title}. {r.abstract}" for r in records]
    qv = emb.encode([query])[0]
    dv = emb.encode(texts)
    dense = [max(0.0, cosine(qv, d)) for d in dv]
    bm = BM25(texts).scores(query)
    mx = max(bm) or 1.0
    for r, d, b in zip(records, dense, bm):
        bonus = min(0.1, 0.05 * (len(r.sources) - 1)) + (0.03 if r.abstract else 0.0)
        r.relevance = round(min(1.0, 0.5 * d + 0.5 * (b / mx) + bonus), 4)
    return sorted(records, key=lambda r: r.relevance, reverse=True)


class DiscoveryAgent(Agent):
    name = "Paper Discovery"

    def run(self, state: dict) -> dict:
        st = get_settings()
        pid = state["project_id"]
        plan = state.get("plan") or {}
        cfg = state.get("config") or {}
        max_papers = int(cfg.get("max_papers") or st.max_papers)
        warnings: list[str] = []
        queries = list(dict.fromkeys((plan.get("search_queries") or [state.get("question", "")]) + (state.get("extra_queries") or [])))
        pre = state.get("preselected_ids") or []
        uploaded = state.get("uploaded_ids") or []
        records, ws = ([], []) if (pre and cfg.get("skip_search")) else search_all(queries, per_source=int(cfg.get("per_source", 10)))
        warnings += ws
        ranked = rank_records(" ".join([state.get("question", "")] + (plan.get("keywords") or [])[:6]), records)
        for r in ranked:
            repo.upsert_paper(r.to_dict())
            repo.link_paper(pid, r.paper_id, r.relevance, selected=False)
        for u in uploaded:
            repo.link_paper(pid, u, 1.0, selected=True)
        existing = {p["paper_id"]: p for p in repo.project_papers(pid)}
        if pre:
            selected = [p for p in pre if p in existing]
        else:
            usable = [p for p in existing.values() if not p["is_uploaded"] and (p["abstract"] or p["pdf_url"]) and p["relevance"] > 0.05]
            selected = [p["paper_id"] for p in sorted(usable, key=lambda p: -p["relevance"])[:max_papers]]
        selected = list(dict.fromkeys(selected + [u for u in uploaded if u in existing]))
        repo.set_selection(pid, selected)
        cands = repo.project_papers(pid)
        if not cands:
            warnings.append("No papers were found. Check network access/API availability or upload PDFs manually.")
        return {"candidates": cands, "selected_ids": selected, "warnings": warnings}
