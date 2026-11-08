"""OpenAlex works API client (no key; CONTACT_EMAIL enables the polite pool)."""
from __future__ import annotations
import re
from config import get_settings
from .base import PaperRecord, http_get, normalize_doi

URL = "https://api.openalex.org/works"


def _abstract(inv: dict | None) -> str:
    """OpenAlex ships abstracts as an inverted index; rebuild the text."""
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def parse(payload: dict) -> list[PaperRecord]:
    out = []
    for w in payload.get("results") or []:
        title = (w.get("title") or w.get("display_name") or "").strip()
        if not title:
            continue
        loc = w.get("primary_location") or {}
        best = w.get("best_oa_location") or {}
        src = (loc.get("source") or {}).get("display_name")
        oa_id = (w.get("id") or "").rsplit("/", 1)[-1] or None
        arx = None
        for l in w.get("locations") or []:
            m = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+?)(?:\.pdf)?$", l.get("landing_page_url") or l.get("pdf_url") or "")
            if m:
                arx = re.sub(r"v\d+$", "", m.group(1)); break
        out.append(PaperRecord(
            title=title,
            authors=[(a.get("author") or {}).get("display_name", "") for a in w.get("authorships") or [] if (a.get("author") or {}).get("display_name")],
            year=w.get("publication_year"), abstract=_abstract(w.get("abstract_inverted_index")),
            doi=normalize_doi(w.get("doi")), arxiv_id=arx, openalex_id=oa_id, venue=src,
            url=loc.get("landing_page_url") or w.get("doi"), pdf_url=best.get("pdf_url") or loc.get("pdf_url"),
            sources=["openalex"], citation_count=w.get("cited_by_count")))
    return out


def search(query: str, limit: int = 10) -> list[PaperRecord]:
    params = {"search": query, "per-page": min(limit, 50)}
    if get_settings().contact_email:
        params["mailto"] = get_settings().contact_email
    return parse(http_get(URL, params=params).json())
