"""ChromaDB persistent vector store. Stores embeddings, chunk text and source metadata."""
from __future__ import annotations
from dataclasses import dataclass
import chromadb
from config import get_settings
from document_processing.chunker import TextChunk
from .embeddings import get_embedder


@dataclass
class Hit:
    chunk_uid: str
    text: str
    paper_id: str
    page: int | None
    section: str
    score: float  # cosine similarity in [-1,1]


class VectorStore:
    def __init__(self, path: str | None = None, embedder=None, ephemeral: bool = False, collection: str | None = None):
        self.embedder = embedder or get_embedder()
        self.client = chromadb.EphemeralClient() if ephemeral else chromadb.PersistentClient(path=path or get_settings().chroma_dir)
        import uuid
        # Ephemeral clients share process-wide state in Chroma, so each gets a unique collection name.
        name = collection or (f"lm_eph_{uuid.uuid4().hex[:10]}" if ephemeral else f"literamind_{self.embedder.name}")
        self.col = self.client.get_or_create_collection(name.replace("_-", "_")[:60],
                                                        metadata={"hnsw:space": "cosine"})

    def add_chunks(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            return 0
        embs = self.embedder.encode([c.text for c in chunks])
        self.col.upsert(ids=[c.chunk_uid for c in chunks], embeddings=embs.tolist(),
                        documents=[c.text for c in chunks],
                        metadatas=[{"paper_id": c.paper_id, "page": int(c.page or 0), "section": c.section or "",
                                    "idx": c.idx} for c in chunks])
        return len(chunks)

    @staticmethod
    def _where(paper_ids: list[str] | None):
        if not paper_ids:
            return None
        return {"paper_id": paper_ids[0]} if len(paper_ids) == 1 else {"paper_id": {"$in": list(paper_ids)}}

    def query(self, text: str, k: int = 8, paper_ids: list[str] | None = None) -> list[Hit]:
        n = self.count(paper_ids)
        if n == 0 or not (text or "").strip():
            return []
        emb = self.embedder.encode([text])[0].tolist()
        res = self.col.query(query_embeddings=[emb], n_results=min(k, n), where=self._where(paper_ids))
        hits = []
        for uid, doc, md, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(uid, doc, md["paper_id"], md.get("page") or None, md.get("section", ""), 1.0 - float(dist)))
        return hits

    def get_all(self, paper_ids: list[str] | None = None) -> list[Hit]:
        res = self.col.get(where=self._where(paper_ids), include=["documents", "metadatas"])
        return [Hit(i, d, m["paper_id"], m.get("page") or None, m.get("section", ""), 0.0)
                for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def count(self, paper_ids: list[str] | None = None) -> int:
        if not paper_ids:
            return self.col.count()
        return len(self.col.get(where=self._where(paper_ids), include=[])["ids"])

    def delete_paper(self, paper_id: str) -> None:
        self.col.delete(where={"paper_id": paper_id})

    def has_paper(self, paper_id: str) -> bool:
        return self.count([paper_id]) > 0


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def set_store(s: VectorStore | None) -> None:
    global _store
    _store = s
