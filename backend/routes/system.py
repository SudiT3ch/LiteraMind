from fastapi import APIRouter
from config import get_settings
from database import repository as repo
from llm.provider import get_provider
from retrieval.embeddings import get_embedder

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/stats")
def stats():
    return repo.stats()


@router.get("/config")
def config():
    """Non-secret runtime configuration (keys are never returned)."""
    st = get_settings(); llm = get_provider()
    return {"llm_provider": llm.name, "llm_available": llm.available, "embedding_backend": get_embedder().name,
            "sources": st.sources(), "max_papers": st.max_papers, "max_upload_mb": st.max_upload_mb,
            "s2_key_configured": bool(st.s2_api_key)}
