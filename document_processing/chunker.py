"""Configurable overlapping chunker that preserves paper id, page number and section."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from config import get_settings
from .pdf_extractor import ExtractedDocument


@dataclass
class TextChunk:
    chunk_uid: str
    paper_id: str
    idx: int
    text: str
    page: int | None
    section: str
    metadata: dict = field(default_factory=dict)


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\(\[])")


def split_sentences(text: str) -> list[str]:
    parts = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s*\n\s*", " ", para).strip()
        if para:
            parts.extend(s.strip() for s in _SENT.split(para) if s.strip())
    return parts


def chunk_text(text: str, chunk_words: int, overlap: int) -> list[str]:
    """Sentence-aware windows of ~chunk_words words with `overlap` words carried over."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    overlap = max(0, min(overlap, chunk_words - 1))
    chunks: list[list[str]] = []
    cur: list[str] = []
    count = 0
    for sent in split_sentences(text):
        words = sent.split()
        # very long "sentence" (tables, bad extraction): hard-split
        while len(words) > chunk_words:
            if cur:
                chunks.append(cur); cur, count = [], 0
            chunks.append(words[:chunk_words]); words = words[chunk_words - overlap:]
        if count + len(words) > chunk_words and cur:
            chunks.append(cur)
            tail = " ".join(cur).split()[-overlap:] if overlap else []
            cur, count = ([" ".join(tail)] if tail else []), len(tail)
        cur.append(" ".join(words)); count += len(words)
    if cur:
        chunks.append(cur)
    out, seen = [], set()
    for c in chunks:
        t = " ".join(c).strip()
        if len(t.split()) >= 5 and t not in seen:
            seen.add(t); out.append(t)
    return out


def chunk_document(doc: ExtractedDocument, paper_id: str, chunk_words: int | None = None,
                   overlap: int | None = None, extra_meta: dict | None = None) -> list[TextChunk]:
    st = get_settings()
    cw, ov = chunk_words or st.chunk_words, st.chunk_overlap if overlap is None else overlap
    out: list[TextChunk] = []
    for page in doc.pages:
        for section, seg_text in page.segments:
            for t in chunk_text(seg_text, cw, ov):
                i = len(out)
                out.append(TextChunk(f"{paper_id}::c{i}", paper_id, i, t, page.page, section,
                                     {"paper_id": paper_id, "page": page.page, "section": section, **(extra_meta or {})}))
    return out


def chunk_abstract(paper_id: str, abstract: str, extra_meta: dict | None = None) -> list[TextChunk]:
    """Fallback when only an abstract is available (flagged via section='Abstract')."""
    return [TextChunk(f"{paper_id}::c{i}", paper_id, i, t, None, "Abstract",
                      {"paper_id": paper_id, "page": 0, "section": "Abstract", "abstract_only": True, **(extra_meta or {})})
            for i, t in enumerate(chunk_text(abstract, get_settings().chunk_words, 0))] if (abstract or "").strip() else []
