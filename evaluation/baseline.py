"""BASELINE: traditional single-pass RAG (retrieve top-k chunks -> one answer). With an LLM it makes ONE generation call;
without an LLM it is an extractive RAG (answer = top retrieved sentences with citations) — documented, never presented as LLM output."""
from __future__ import annotations
from agents.verifier import best_sentence
from document_processing.chunker import split_sentences
from llm.provider import get_provider, wrap_untrusted, LLMError
from retrieval.retriever import retrieve


def baseline_rag(question: str, paper_ids: list[str] | None = None, k: int = 6) -> dict:
    chunks = retrieve(question, k, paper_ids=paper_ids)
    llm = get_provider()
    if llm.available and chunks:
        ctx = "\n".join(f"[{i+1}] (paper {c.paper_id}, p.{c.page}) {c.text}" for i, c in enumerate(chunks))
        try:
            text = llm.complete(f"Answer the research question using only the excerpts; cite as [n].\nQuestion: {question}\n" + wrap_untrusted(ctx), max_tokens=700)
            return {"mode": "llm", "answer": text, "citations": [{"n": i + 1, "paper_id": c.paper_id, "page": c.page, "section": c.section} for i, c in enumerate(chunks)], "usage": llm.usage()}
        except LLMError:
            pass
    sents = []
    for c in chunks:
        for s in split_sentences(c.text):
            if len(s.split()) >= 6: sents.append({"text": s, "paper_id": c.paper_id, "page": c.page, "section": c.section, "score": c.score})
    top = sorted(sents, key=lambda s: -s["score"])[:6]
    return {"mode": "extractive", "answer": " ".join(s["text"] for s in top), "citations": top, "usage": llm.usage()}


def baseline_claims(result: dict) -> list[dict]:
    """Per-sentence claims with their cited source, so the same citation metrics apply to the baseline."""
    out = []
    for c in result["citations"]:
        if "text" in c:
            out.append({"claim": c["text"], "check": c["text"], "paper_id": c["paper_id"]})
    return out
