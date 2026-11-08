"""Ingestion: PDF bytes/files and open-access downloads -> DB chunks + Chroma vectors. Falls back to abstract-only."""
from __future__ import annotations
import hashlib, os, re
from pathlib import Path
import httpx
from config import get_settings
from logging_utils import get_logger
from database.database import session_scope
from database.models import Paper, Document, Chunk
from retrieval.vector_store import get_store
from .pdf_extractor import extract_pdf, PDFError
from .chunker import chunk_document, chunk_abstract, TextChunk

log = get_logger("ingest")


def _persist_chunks(paper_id: str, chunks: list[TextChunk]) -> None:
    with session_scope() as s:
        s.query(Chunk).filter(Chunk.paper_id == paper_id).delete()
        for c in chunks:
            s.add(Chunk(chunk_uid=c.chunk_uid, paper_id=paper_id, idx=c.idx, page=c.page, section=c.section, text=c.text))


def index_chunks(paper_id: str, chunks: list[TextChunk]) -> int:
    store = get_store()
    store.delete_paper(paper_id)
    _persist_chunks(paper_id, chunks)
    return store.add_chunks(chunks)


def index_abstract(paper_id: str) -> int:
    with session_scope() as s:
        p = s.get(Paper, paper_id)
        if p is None:
            return 0
        abstract, title = p.abstract or "", p.title
    chunks = chunk_abstract(paper_id, abstract or title)
    n = index_chunks(paper_id, chunks)
    with session_scope() as s:
        p = s.get(Paper, paper_id)
        if p and p.fulltext_status != "fulltext":
            p.fulltext_status = "abstract_only" if n else "none"
    return n


def safe_filename(name: str) -> str:
    name = os.path.basename(name or "upload.pdf")
    name = re.sub(r"[^A-Za-z0-9._\- ]", "_", name)[:120]
    return name if name.lower().endswith(".pdf") else name + ".pdf"


def ingest_pdf_bytes(data: bytes, filename: str, paper_id: str | None = None, meta: dict | None = None,
                     uploaded: bool = True) -> dict:
    """Validate, store, extract, chunk and index a PDF. Returns a status dict; raises PDFError for invalid input."""
    st = get_settings()
    doc = extract_pdf(data=data)  # validates; raises PDFError
    fname = safe_filename(filename)
    paper_id = paper_id or "upload_" + doc.sha256[:12]
    meta = meta or {}
    Path(st.upload_dir).mkdir(parents=True, exist_ok=True)
    path = Path(st.upload_dir) / f"{doc.sha256[:16]}_{fname}"
    path.write_bytes(data)
    with session_scope() as s:
        p = s.get(Paper, paper_id)
        if p is None:
            p = Paper(paper_id=paper_id, title=(meta.get("title") or doc.title or fname)[:300], authors=meta.get("authors") or [],
                      year=meta.get("year"), abstract=meta.get("abstract") or "", is_uploaded=uploaded, sources=["upload"] if uploaded else [])
            s.add(p)
        s.flush()
        d = Document(paper_id=paper_id, filename=fname, path=str(path), sha256=doc.sha256, size_bytes=len(data), n_pages=doc.n_pages,
                     status="uploaded")
        s.add(d); s.flush(); doc_id = d.id
    if doc.is_empty:
        _set_doc(doc_id, "failed", "No extractable text (scanned PDF? OCR is not supported).", paper_id, "failed")
        return {"paper_id": paper_id, "document_id": doc_id, "status": "failed", "chunks": 0,
                "error": "No extractable text (possibly a scanned PDF)."}
    chunks = chunk_document(doc, paper_id)
    n = index_chunks(paper_id, chunks)
    _set_doc(doc_id, "processed", None, paper_id, "fulltext")
    if doc.injection_flags:
        log.warning("Document %s contains instruction-like text on %d page(s); treated as data only", fname, doc.injection_flags)
    return {"paper_id": paper_id, "document_id": doc_id, "status": "processed", "chunks": n, "pages": doc.n_pages,
            "injection_flags": doc.injection_flags}


def _set_doc(doc_id: int, status: str, error: str | None, paper_id: str, ft: str) -> None:
    with session_scope() as s:
        d = s.get(Document, doc_id)
        if d: d.status, d.error = status, error
        p = s.get(Paper, paper_id)
        if p: p.fulltext_status = ft


def download_pdf(url: str) -> bytes:
    """Stream a PDF with a hard size cap. Raises PDFError on any problem."""
    st = get_settings()
    cap = st.max_upload_mb * 1024 * 1024
    try:
        with httpx.stream("GET", url, timeout=st.http_timeout, follow_redirects=True,
                          headers={"User-Agent": "LiteraMind/1.0"}) as r:
            r.raise_for_status()
            buf = bytearray()
            for part in r.iter_bytes():
                buf += part
                if len(buf) > cap:
                    raise PDFError("Remote PDF exceeds the size limit.")
            return bytes(buf)
    except PDFError:
        raise
    except httpx.HTTPError as e:
        raise PDFError(f"Download failed: {type(e).__name__}") from e


def fetch_and_index(paper_id: str) -> str:
    """Try open-access PDF full text; fall back to abstract chunks. Returns resulting fulltext_status."""
    with session_scope() as s:
        p = s.get(Paper, paper_id)
        url = p.pdf_url if p else None
    store = get_store()
    if store.has_paper(paper_id):
        with session_scope() as s:
            return s.get(Paper, paper_id).fulltext_status
    if url:
        try:
            data = download_pdf(url)
            r = ingest_pdf_bytes(data, f"{paper_id}.pdf", paper_id=paper_id, uploaded=False)
            if r["status"] == "processed":
                return "fulltext"
        except PDFError as e:
            log.info("Full text unavailable for %s: %s", paper_id, e)
    index_abstract(paper_id)
    with session_scope() as s:
        return s.get(Paper, paper_id).fulltext_status


def delete_document(paper_id: str) -> bool:
    """Remove an uploaded document, its chunks, vectors and file."""
    with session_scope() as s:
        p = s.get(Paper, paper_id)
        if p is None:
            return False
        for d in s.query(Document).filter(Document.paper_id == paper_id).all():
            try: Path(d.path).unlink(missing_ok=True)
            except OSError: pass
            s.delete(d)
        s.query(Chunk).filter(Chunk.paper_id == paper_id).delete()
        from database.models import ProjectPaper
        s.query(ProjectPaper).filter(ProjectPaper.paper_id == paper_id).delete()
        s.delete(p)
    get_store().delete_paper(paper_id)
    return True
