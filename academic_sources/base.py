"""Shared paper record, HTTP helper with retry/rate-limit handling, and de-duplication."""
from __future__ import annotations
import hashlib, re, time
from dataclasses import dataclass, field, asdict
from typing import Any
import httpx
from config import get_settings
from logging_utils import get_logger, redact

log = get_logger("sources")


class SourceError(Exception):
    """Raised when an academic source fails after retries (caller degrades gracefully)."""


@dataclass
class PaperRecord:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    s2_id: str | None = None
    openalex_id: str | None = None
    venue: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    sources: list[str] = field(default_factory=list)
    citation_count: int | None = None
    relevance: float = 0.0

    @property
    def paper_id(self) -> str:
        if self.arxiv_id:
            return "arxiv_" + re.sub(r"v\d+$", "", self.arxiv_id).replace("/", "_")
        if self.doi:
            return "doi_" + re.sub(r"[^a-z0-9]+", "_", self.doi.lower()).strip("_")
        return "h_" + hashlib.sha1(normalize_title(self.title).encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["paper_id"] = self.paper_id
        return d


def normalize_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def normalize_doi(d: str | None) -> str | None:
    if not d:
        return None
    d = d.strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", d)
    return d or None


def http_get(url: str, params: dict | None = None, headers: dict | None = None,
             timeout: int | None = None, retries: int | None = None) -> httpx.Response:
    """GET with timeouts, exponential back-off, and 429/5xx handling. Raises SourceError."""
    st = get_settings()
    timeout = timeout or st.http_timeout
    retries = st.http_retries if retries is None else retries
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = httpx.get(url, params=params, headers=headers, timeout=timeout, follow_redirects=True)
            if r.status_code == 429 or r.status_code >= 500:
                wait = min(float(r.headers.get("Retry-After", 2 ** attempt)), 8.0) if r.headers.get("Retry-After", "1").replace(".", "").isdigit() else 2 ** attempt
                last = SourceError(f"HTTP {r.status_code} from {httpx.URL(url).host}")
                if attempt < retries:
                    time.sleep(wait)
                    continue
                raise last
            r.raise_for_status()
            return r
        except httpx.HTTPStatusError as e:
            raise SourceError(redact(f"HTTP {e.response.status_code} from {httpx.URL(url).host}")) from e
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = SourceError(redact(f"Network error contacting {httpx.URL(url).host}: {type(e).__name__}"))
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise last from e
    raise last or SourceError("unknown error")


def merge_records(a: PaperRecord, b: PaperRecord) -> PaperRecord:
    """Merge b into a, keeping the most complete metadata (never inventing values)."""
    for f in ("year", "doi", "arxiv_id", "s2_id", "openalex_id", "venue", "url", "pdf_url", "citation_count"):
        if getattr(a, f) in (None, "") and getattr(b, f) not in (None, ""):
            setattr(a, f, getattr(b, f))
    if len(b.abstract or "") > len(a.abstract or ""):
        a.abstract = b.abstract
    if len(b.authors) > len(a.authors):
        a.authors = b.authors
    a.sources = sorted(set(a.sources) | set(b.sources))
    return a


def deduplicate(records: list[PaperRecord]) -> list[PaperRecord]:
    """De-duplicate by DOI, arXiv id, then normalised title."""
    by_key: dict[str, PaperRecord] = {}
    out: list[PaperRecord] = []
    for r in records:
        if not r.title or not r.title.strip():
            continue
        r.doi = normalize_doi(r.doi)
        keys = []
        if r.doi: keys.append("doi:" + r.doi)
        if r.arxiv_id: keys.append("arxiv:" + re.sub(r"v\d+$", "", r.arxiv_id))
        keys.append("title:" + normalize_title(r.title))
        hit = next((by_key[k] for k in keys if k in by_key), None)
        if hit is not None:
            merge_records(hit, r)
            target = hit
        else:
            out.append(r)
            target = r
        for k in keys:
            by_key.setdefault(k, target)
        if target.doi: by_key.setdefault("doi:" + target.doi, target)
        if target.arxiv_id: by_key.setdefault("arxiv:" + re.sub(r"v\d+$", "", target.arxiv_id), target)
    return out
