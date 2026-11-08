import numpy as np
from retrieval.embeddings import HashingEmbedder, cosine, tokenize
from retrieval.vector_store import VectorStore, get_store
from retrieval.retriever import retrieve, BM25
from document_processing.chunker import TextChunk
from evaluation.fixtures.synthetic_corpus import QUERIES


def test_embedder_deterministic_normalised_and_similarity():
    e = HashingEmbedder()
    v = e.encode(["explainable AI for medical diagnosis", "explainable AI for medical diagnosis", "stock market volatility"])
    assert np.allclose(v[0], v[1]) and abs(np.linalg.norm(v[0]) - 1) < 1e-5
    assert cosine(v[0], v[1]) > cosine(v[0], v[2])
    assert e.encode([""]).shape == (1, 512) and np.linalg.norm(e.encode([""])[0]) == 0


def test_vector_store_crud_and_filter():
    s = get_store()
    s.add_chunks([TextChunk("a::c0", "a", 0, "neural networks for chest x-ray", 3, "Methods"),
                  TextChunk("b::c0", "b", 0, "federated learning across hospitals", 1, "Intro")])
    assert s.count() == 2 and s.has_paper("a")
    h = s.query("chest x-ray", 5, ["a"])
    assert len(h) == 1 and h[0].paper_id == "a" and h[0].page == 3 and h[0].section == "Methods"
    s.add_chunks([TextChunk("a::c0", "a", 0, "neural networks for chest x-ray", 3, "Methods")])  # upsert idempotent
    assert s.count() == 2
    s.delete_paper("a"); assert not s.has_paper("a") and s.count() == 1
    assert s.query("", 3) == [] and s.add_chunks([]) == 0


def test_retrieval_modes_return_traceable_chunks(corpus_store):
    for mode in ("semantic", "keyword", "hybrid"):
        r = retrieve("federated learning privacy hospitals", 5, mode=mode)
        assert r and all(x.paper_id and x.section for x in r)
        assert r[0].paper_id in {"syn_fl1", "syn_fl2"}, mode


def test_references_excluded_and_paper_filter(corpus_store):
    r = retrieve("accuracy", 10, paper_ids=["syn_xai1"])
    assert r and {x.paper_id for x in r} == {"syn_xai1"}
    assert retrieve("anything", 5, paper_ids=["nonexistent"]) == []


def test_empty_store_returns_empty():
    assert retrieve("query", 5) == []


def test_bm25_prefers_matching_doc():
    b = BM25(["cats and dogs", "federated learning hospitals privacy", "weather today"])
    s = b.scores("federated privacy")
    assert s.index(max(s)) == 1


def test_retrieval_quality_on_fixture_corpus(corpus_store):
    """Sanity check on SYNTHETIC corpus: top-3 chunk papers should contain a relevant paper for every query."""
    for q, rel in QUERIES.items():
        r = retrieve(q, 5)
        assert {x.paper_id for x in r[:3]} & rel, q
