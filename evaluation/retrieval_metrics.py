"""Retrieval metrics at paper level: Precision@k, Recall@k, MRR, plus mean 'relevance' (fraction of hits that are relevant)."""
from __future__ import annotations
from typing import Callable, Iterable


def dedupe_ranking(paper_ids: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for p in paper_ids:
        if p not in seen: seen.add(p); out.append(p)
    return out


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    return sum(1 for p in top if p in relevant) / len(top) if top else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, p in enumerate(ranked, 1):
        if p in relevant: return 1.0 / i
    return 0.0


def evaluate_retrieval(queries: dict[str, set[str]], search: Callable[[str], list[str]], p_k: int = 5, r_k: int = 10) -> dict:
    """search(query) -> ranked paper ids (chunk-level results should be de-duplicated by the caller or here)."""
    per_q, P, R, M = [], [], [], []
    for q, rel in queries.items():
        ranked = dedupe_ranking(search(q))
        p, r, m = precision_at_k(ranked, rel, p_k), recall_at_k(ranked, rel, r_k), reciprocal_rank(ranked, rel)
        P.append(p); R.append(r); M.append(m)
        per_q.append({"query": q, f"P@{p_k}": round(p, 3), f"R@{r_k}": round(r, 3), "RR": round(m, 3)})
    n = len(queries) or 1
    return {f"P@{p_k}": round(sum(P) / n, 4), f"R@{r_k}": round(sum(R) / n, 4), "MRR": round(sum(M) / n, 4), "n_queries": len(queries), "per_query": per_q}
