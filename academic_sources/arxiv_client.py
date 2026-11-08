"""arXiv API client (Atom feed, no key required). https://info.arxiv.org/help/api/"""
from __future__ import annotations
import re
import xml.etree.ElementTree as ET
from .base import PaperRecord, http_get, SourceError

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
URL = "http://export.arxiv.org/api/query"


def parse_feed(xml_text: str) -> list[PaperRecord]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise SourceError(f"arXiv returned invalid XML: {e}") from e
    out: list[PaperRecord] = []
    for e in root.findall("a:entry", NS):
        title = re.sub(r"\s+", " ", (e.findtext("a:title", "", NS) or "")).strip()
        if not title:
            continue
        abs_url = (e.findtext("a:id", "", NS) or "").strip()
        m = re.search(r"arxiv\.org/abs/(.+)$", abs_url)
        aid = re.sub(r"v\d+$", "", m.group(1)) if m else None
        published = e.findtext("a:published", "", NS) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        pdf = None
        for l in e.findall("a:link", NS):
            if l.get("title") == "pdf" or l.get("type") == "application/pdf":
                pdf = l.get("href")
        if not pdf and aid:
            pdf = f"https://arxiv.org/pdf/{aid}"
        doi = e.findtext("arxiv:doi", None, NS)
        out.append(PaperRecord(
            title=title,
            authors=[(a.findtext("a:name", "", NS) or "").strip() for a in e.findall("a:author", NS)],
            year=year,
            abstract=re.sub(r"\s+", " ", e.findtext("a:summary", "", NS) or "").strip(),
            doi=doi, arxiv_id=aid, url=abs_url or None, pdf_url=pdf, sources=["arxiv"],
            venue="arXiv"))
    return out


def search(query: str, limit: int = 10) -> list[PaperRecord]:
    q = re.sub(r"[^\w\s\-]", " ", query).strip()
    if not q:
        return []
    params = {"search_query": "all:" + " AND all:".join(q.split()[:8]), "start": 0,
              "max_results": limit, "sortBy": "relevance"}
    return parse_feed(http_get(URL, params=params).text)
