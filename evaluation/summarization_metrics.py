"""ROUGE-L (rouge-score if installed, own LCS otherwise) and verifier-based factual consistency of generated summaries."""
from __future__ import annotations
import re


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> dict:
    try:
        from rouge_score import rouge_scorer
        s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(reference, candidate)["rougeL"]
        return {"precision": round(s.precision, 4), "recall": round(s.recall, 4), "f1": round(s.fmeasure, 4), "impl": "rouge-score"}
    except ImportError:
        c, r = re.findall(r"\w+", candidate.lower()), re.findall(r"\w+", reference.lower())
        l = _lcs(c, r); p, rc = (l / len(c) if c else 0), (l / len(r) if r else 0)
        return {"precision": round(p, 4), "recall": round(rc, 4), "f1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0, "impl": "builtin-lcs"}


def factual_consistency(summary: str, paper_ids: list[str]) -> dict:
    """Fraction of summary sentences judged Supported/Partially supported by the source text (lexical entailment)."""
    from agents.verifier import best_sentence
    from document_processing.chunker import split_sentences
    sents = [s for s in split_sentences(summary) if len(s.split()) >= 5]
    if not sents: return {"consistency": None, "n_sentences": 0}
    ok = 0
    for s in sents:
        _, j = best_sentence(s, paper_ids)
        ok += 1 if j and j["status"] in ("Supported", "Partially supported") else 0
    return {"consistency": round(ok / len(sents), 4), "n_sentences": len(sents)}
