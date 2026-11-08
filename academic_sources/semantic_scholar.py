"""Semantic Scholar Graph API client. Optional S2_API_KEY raises rate limits."""
from __future__ import annotations
from config import get_settings
from .base import PaperRecord, http_get, normalize_doi

URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,year,authors,venue,externalIds,openAccessPdf,url,citationCount"


def parse(payload: dict) -> list[PaperRecord]:
    out = []
    for p in payload.get("data") or []:
        if not p.get("title"):
            continue
        ext = p.get("externalIds") or {}
        oa = p.get("openAccessPdf") or {}
        out.append(PaperRecord(
            title=p["title"].strip(), authors=[a.get("name", "") for a in p.get("authors") or [] if a.get("name")],
            year=p.get("year"), abstract=(p.get("abstract") or "").strip(),
            doi=normalize_doi(ext.get("DOI")), arxiv_id=ext.get("ArXiv"), s2_id=p.get("paperId"),
            venue=p.get("venue") or None, url=p.get("url"), pdf_url=oa.get("url"),
            sources=["semantic_scholar"], citation_count=p.get("citationCount")))
    return out


def search(query: str, limit: int = 10) -> list[PaperRecord]:
    st = get_settings()
    headers = {"x-api-key": st.s2_api_key} if st.s2_api_key else None
    r = http_get(URL, params={"query": query, "limit": min(limit, 100), "fields": FIELDS}, headers=headers)
    return parse(r.json())
