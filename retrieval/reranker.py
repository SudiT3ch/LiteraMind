"""Lightweight reranker: blends dense/fused score with query-term coverage and phrase hits.
Optional cross-encoder is used only if sentence-transformers and the model are available."""
from __future__ import annotations
from .embeddings import tokenize


def lexical_rerank(query: str, items: list, text_attr: str = "text", score_attr: str = "score", alpha: float = 0.7) -> list:
    q = set(tokenize(query))
    if not q or not items:
        return items
    rescored = []
    for it in items:
        toks = set(tokenize(getattr(it, text_attr)))
        coverage = len(q & toks) / len(q)
        base = float(getattr(it, score_attr))
        rescored.append((alpha * base + (1 - alpha) * coverage, it))
    rescored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, it in rescored:
        it.score = float(s)
        out.append(it)
    return out


def cross_encoder_rerank(query: str, items: list, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(model_name)
        scores = ce.predict([(query, i.text) for i in items])
        for it, s in zip(items, scores):
            it.score = float(s)
        return sorted(items, key=lambda x: x.score, reverse=True)
    except Exception:
        return lexical_rerank(query, items)
