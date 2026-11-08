"""Hybrid retriever: dense (Chroma) + BM25 keyword retrieval fused with Reciprocal Rank Fusion, then reranked.
Every result keeps paper_id / page / section for traceability."""
from __future__ import annotations
import math
from dataclasses import dataclass
from .embeddings import tokenize
from .reranker import lexical_rerank
from .vector_store import VectorStore, Hit, get_store


@dataclass
class Retrieved:
    chunk_uid: str
    text: str
    paper_id: str
    page: int | None
    section: str
    score: float
    semantic: float = 0.0
    keyword: float = 0.0


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.toks = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = (sum(len(t) for t in self.toks) / len(self.toks)) if self.toks else 0.0
        df: dict[str, int] = {}
        for t in self.toks:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        n = len(self.toks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for t in self.toks:
            tf: dict[str, int] = {}
            for w in t:
                tf[w] = tf.get(w, 0) + 1
            s = 0.0
            for w in q:
                if w in tf:
                    s += self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * len(t) / (self.avg or 1)))
            out.append(s)
        return out


EXCLUDE_DEFAULT = {"references", "acknowledgments", "acknowledgements"}


def retrieve(query: str, k: int = 8, paper_ids: list[str] | None = None, mode: str = "hybrid",
             store: VectorStore | None = None, rerank: bool = True, exclude_sections: set[str] | None = None) -> list[Retrieved]:
    """mode: 'semantic' | 'keyword' | 'hybrid'."""
    store = store or get_store()
    exclude = EXCLUDE_DEFAULT if exclude_sections is None else exclude_sections
    pool_k = max(k * 4, 20)
    sem: list[Hit] = store.query(query, pool_k, paper_ids) if mode in ("semantic", "hybrid") else []
    kw: list[tuple[Hit, float]] = []
    if mode in ("keyword", "hybrid"):
        allc = [h for h in store.get_all(paper_ids)]
        if allc:
            sc = BM25([h.text for h in allc]).scores(query)
            kw = sorted([(h, s) for h, s in zip(allc, sc) if s > 0], key=lambda x: x[1], reverse=True)[:pool_k]
    fused: dict[str, Retrieved] = {}
    C = 60.0
    for rank, h in enumerate(sem):
        r = fused.setdefault(h.chunk_uid, Retrieved(h.chunk_uid, h.text, h.paper_id, h.page, h.section, 0.0))
        r.semantic = h.score
        r.score += 1.0 / (C + rank + 1)
    mx = max([s for _, s in kw], default=1.0) or 1.0
    for rank, (h, s) in enumerate(kw):
        r = fused.setdefault(h.chunk_uid, Retrieved(h.chunk_uid, h.text, h.paper_id, h.page, h.section, 0.0))
        r.keyword = s / mx
        r.score += 1.0 / (C + rank + 1)
    items = [r for r in fused.values() if (r.section or "").lower() not in exclude]
    if not items:
        return []
    top = max(r.score for r in items) or 1.0
    for r in items:
        r.score = r.score / top  # normalise fused score to [0,1]
    items.sort(key=lambda r: r.score, reverse=True)
    items = items[: max(k * 2, k)]
    if rerank:
        items = lexical_rerank(query, items)
    return items[:k]
