"""PDF text extraction with PyMuPDF. Uploaded PDFs are untrusted: validated, size/page-limited, never executed."""
from __future__ import annotations
import hashlib, re
from dataclasses import dataclass, field
from pathlib import Path
from config import get_settings

SECTION_NAMES = ("abstract", "introduction", "background", "related work", "literature review", "preliminaries",
                 "method", "methods", "methodology", "approach", "materials and methods", "model", "experiments",
                 "experimental setup", "experimental results", "evaluation", "results", "results and discussion",
                 "discussion", "limitations", "conclusion", "conclusions", "future work", "acknowledgments",
                 "acknowledgements", "references", "appendix", "dataset", "datasets", "data")
_HEAD_RE = re.compile(r"^(?:(?:\d+(?:\.\d+){0,2}|[IVX]{1,5})[\.\)]?\s+)?([A-Za-z][A-Za-z \-&]{2,60})$")
INJECTION_PATTERNS = re.compile(
    r"(ignore (all |any )?(previous|prior|above) (instructions|prompts)|disregard (the )?(system|previous)|"
    r"you are now|system prompt|reveal your (instructions|prompt)|act as (an? )?(?!researcher)\w+ and)", re.I)


class PDFError(Exception):
    """Invalid, oversized, encrypted or unreadable PDF."""


@dataclass
class PageText:
    page: int  # 1-based
    text: str
    section: str = ""  # section active at the END of the page's first heading, see segments
    segments: list[tuple[str, str]] = field(default_factory=list)  # (section, text) in reading order


@dataclass
class ExtractedDocument:
    pages: list[PageText]
    n_pages: int
    title: str = ""
    sha256: str = ""
    metadata: dict = field(default_factory=dict)
    injection_flags: int = 0

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def is_empty(self) -> bool:
        return len(self.full_text.strip()) < 50


def clean_text(t: str) -> str:
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", t)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)            # de-hyphenate line wraps
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def heading_of(line: str) -> str | None:
    s = line.strip()
    if not (2 < len(s) <= 70):
        return None
    m = _HEAD_RE.match(s)
    if not m:
        return None
    name = m.group(1).strip().lower()
    return name.title() if name in SECTION_NAMES else None


def split_sections(text: str, current: str) -> tuple[list[tuple[str, str]], str]:
    """Split page text into (section, text) segments; carries the active section across pages."""
    segs: list[tuple[str, list[str]]] = [(current, [])]
    for line in text.splitlines():
        h = heading_of(line)
        if h:
            segs.append((h, []))
        else:
            segs[-1][1].append(line)
    out = [(s, "\n".join(ls).strip()) for s, ls in segs if "\n".join(ls).strip()]
    return out, segs[-1][0]


def validate_pdf_bytes(data: bytes) -> None:
    st = get_settings()
    if not data:
        raise PDFError("The file is empty.")
    if len(data) > st.max_upload_mb * 1024 * 1024:
        raise PDFError(f"File exceeds the {st.max_upload_mb} MB upload limit.")
    if not data[:1024].lstrip().startswith(b"%PDF-"):
        raise PDFError("Not a valid PDF file (missing %PDF header).")


def extract_pdf(path: str | Path | None = None, data: bytes | None = None) -> ExtractedDocument:
    """Extract page-wise text. Raises PDFError on invalid/encrypted/unreadable input."""
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24.3
    except ImportError:  # pragma: no cover
        import fitz
    st = get_settings()
    if data is None:
        if path is None:
            raise PDFError("No input provided.")
        try:
            data = Path(path).read_bytes()
        except OSError as e:
            raise PDFError(f"Cannot read file: {e.__class__.__name__}") from e
    validate_pdf_bytes(data)
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise PDFError("The PDF is corrupted or cannot be opened.") from e
    try:
        if doc.needs_pass or doc.is_encrypted:
            raise PDFError("The PDF is password-protected.")
        n = doc.page_count
        if n == 0:
            raise PDFError("The PDF has no pages.")
        if n > st.max_pdf_pages:
            raise PDFError(f"The PDF has {n} pages; the limit is {st.max_pdf_pages}.")
        pages, section, flags = [], "Front Matter", 0
        for i in range(n):
            try:
                raw = doc[i].get_text("text")
            except Exception:
                raw = ""
            txt = clean_text(raw)
            if INJECTION_PATTERNS.search(txt):
                flags += 1
            segs, section = split_sections(txt, section)
            first = segs[0][0] if segs else section
            pages.append(PageText(page=i + 1, text=txt, section=first, segments=segs))
        meta = {k: v for k, v in (doc.metadata or {}).items() if v}
        first_lines = [l.strip() for l in (pages[0].text.splitlines() if pages else []) if l.strip()]
        title = (meta.get("title") or (first_lines[0] if first_lines else ""))[:300]
        return ExtractedDocument(pages=pages, n_pages=n, title=title, sha256=hashlib.sha256(data).hexdigest(),
                                 metadata=meta, injection_flags=flags)
    finally:
        doc.close()


def detect_injection(text: str) -> bool:
    return bool(INJECTION_PATTERNS.search(text or ""))
