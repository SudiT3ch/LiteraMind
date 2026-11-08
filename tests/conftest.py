import os, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_tmp = tempfile.mkdtemp(prefix="lm_test_")
os.environ.update({"DATABASE_URL": f"sqlite:///{_tmp}/t.db", "CHROMA_DIR": f"{_tmp}/chroma", "UPLOAD_DIR": f"{_tmp}/up",
                   "REPORT_DIR": f"{_tmp}/rep", "EMBEDDING_BACKEND": "hashing", "LLM_PROVIDER": "heuristic",
                   "HTTP_RETRIES": "0"})
import pytest


@pytest.fixture(autouse=True)
def isolated(tmp_path):
    from config import reset_settings
    from database.database import init_db
    from retrieval import embeddings, vector_store
    reset_settings(); embeddings.reset_embedder()
    init_db(f"sqlite:///{tmp_path}/db.sqlite")
    vector_store.set_store(vector_store.VectorStore(ephemeral=True, embedder=embeddings.HashingEmbedder()))
    from llm import provider
    provider.reset_provider()
    yield
    vector_store.set_store(None)


@pytest.fixture
def corpus_store():
    """Vector store loaded with the SYNTHETIC fixture corpus."""
    from evaluation.fixtures.synthetic_corpus import DOCS
    from document_processing.pdf_extractor import ExtractedDocument, PageText, split_sections
    from document_processing.chunker import chunk_document
    from retrieval.vector_store import get_store
    store = get_store()
    for pid, d in DOCS.items():
        pages, sec = [], "Front Matter"
        for i, t in enumerate(d["pages"], 1):
            segs, sec = split_sections(t, sec)
            pages.append(PageText(i, t, segs[0][0], segs))
        store.add_chunks(chunk_document(ExtractedDocument(pages, len(pages)), pid))
    return store
