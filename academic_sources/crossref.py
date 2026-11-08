"""Crossref client: DOI validation and bibliographic metadata lookup."""
from __future__ import annotations
from config import get_settings
from .base import PaperRecord, http_get, SourceError, normalize_doi

URL = "https://api.crossref.org/works"


def _headers() -> dict:
    e = get_settings().contact_email
    return {"User-Agent": f"LiteraMind/1.0 (mailto:{e})" if e else "LiteraMind/1.0"}


def parse_item(item: dict) -> PaperRecord | None:
    titles = item.get("title") or []
    if not titles:
        return None
    issued = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
    authors = [" ".join(x for x in (a.get("given"), a.get("family")) if x) for a in item.get("author") or []]
    return PaperRecord(title=titles[0].strip(), authors=[a for a in authors if a],
                       year=issued[0] if issued and issued[0] else None, doi=normalize_doi(item.get("DOI")),
                       venue=((item.get("container-title") or [None])[0]), url=item.get("URL"), sources=["crossref"])


def lookup_doi(doi: str) -> PaperRecord | None:
    """Return metadata for a DOI, or None if Crossref says it does not exist."""
    doi = normalize_doi(doi)
    if not doi:
        return None
    try:
        r = http_get(f"{URL}/{doi}", headers=_headers(), retries=1)
    except SourceError as e:
        if "HTTP 404" in str(e):
            return None
        raise
    return parse_item(r.json().get("message") or {})


def validate_doi(doi: str) -> bool | None:
    """True = registered, False = not found, None = could not check (network/rate limit)."""
    try:
        return lookup_doi(doi) is not None
    except SourceError:
        return None


def search_bibliographic(title: str) -> PaperRecord | None:
    r = http_get(URL, params={"query.bibliographic": title, "rows": 1}, headers=_headers(), retries=1)
    items = (r.json().get("message") or {}).get("items") or []
    return parse_item(items[0]) if items else None
