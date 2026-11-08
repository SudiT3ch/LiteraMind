"""Citation / claim verification metrics computed from verifier outputs."""
from __future__ import annotations
from retrieval.vector_store import get_store


def citation_metrics(verifications: list[dict], require_page: bool = True) -> dict:
    n = len(verifications)
    if n == 0:
        return {"n_claims": 0, "citation_support_rate": None, "unsupported_claim_rate": None, "citation_completeness": None, "source_traceability": None}
    supported = sum(1 for v in verifications if v["status"] == "Supported")
    partial = sum(1 for v in verifications if v["status"] == "Partially supported")
    unsupported = sum(1 for v in verifications if v["status"] in ("Insufficient evidence", "Contradicted"))
    cited = sum(1 for v in verifications if v.get("paper_id") and v.get("passage"))
    store = get_store(); trace = 0
    for v in verifications:
        if not (v.get("paper_id") and v.get("passage")): continue
        texts = [h.text for h in store.get_all([v["paper_id"]])]
        in_src = any(v["passage"][:60] in t for t in texts)
        if in_src and (v.get("section") or not require_page) and (v.get("page") is not None or not require_page): trace += 1
    return {"n_claims": n, "citation_support_rate": round(supported / n, 4), "partial_support_rate": round(partial / n, 4),
            "unsupported_claim_rate": round(unsupported / n, 4), "citation_completeness": round(cited / n, 4), "source_traceability": round(trace / n, 4)}
