from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from config import get_settings
from database import repository as repo
from document_processing.ingest import ingest_pdf_bytes, delete_document
from document_processing.pdf_extractor import PDFError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("")
async def upload(file: UploadFile = File(...), project_id: int | None = Form(None), title: str | None = Form(None)):
    st = get_settings()
    name = file.filename or "upload.pdf"
    if not name.lower().endswith(".pdf"):
        raise HTTPException(415, "Only .pdf files are accepted.")
    cap = st.max_upload_mb * 1024 * 1024
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data += chunk
        if len(data) > cap:
            raise HTTPException(413, f"File exceeds the {st.max_upload_mb} MB limit.")
    res = ingest_pdf_bytes(bytes(data), name, meta={"title": title} if title else None)  # PDFError -> 422
    if project_id is not None:
        if repo.get_query(project_id) is None:
            raise HTTPException(404, "Unknown project")
        repo.link_paper(project_id, res["paper_id"], 1.0, True)
    return res


@router.get("")
def list_docs():
    return repo.list_documents()


@router.delete("/{paper_id}")
def delete(paper_id: str):
    if not delete_document(paper_id):
        raise HTTPException(404, "Document not found")
    return {"deleted": paper_id}


@router.get("/{paper_id}/text")
def text(paper_id: str, limit: int = 20000):
    if repo.get_paper(paper_id) is None:
        raise HTTPException(404, "Document not found")
    return {"paper": repo.get_paper(paper_id), "text": repo.document_text(paper_id, min(limit, 200000))}
