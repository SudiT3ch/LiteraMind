from agents.planner import PlannerAgent
from agents.discovery import DiscoveryAgent, rank_records
from academic_sources.base import PaperRecord
from academic_sources import aggregator
from database import repository as repo


def test_planner_heuristic_example_query():
    r = PlannerAgent().run({"question": "How can Explainable AI improve the reliability of medical diagnosis systems?"})
    p = r["plan"]
    assert p["domain"] == "Medicine & Healthcare" and p["method"] == "heuristic"
    assert any("Explainable AI" in k for k in p["keywords"]) and p["search_queries"]
    assert any("interpretab" in c.lower() or "XAI" in c for c in p["concepts"])
    assert len(p["search_queries"]) <= 6 and len(set(p["search_queries"])) == len(p["search_queries"])


def test_planner_rejects_unsupported_query():
    r = PlannerAgent().run({"question": "??"})
    assert r["status"] == "failed" and r["errors"]


def test_planner_llm_path_and_fallback(monkeypatch):
    from llm import provider
    class Fake(provider.LLMProvider):
        name = "fake"
        def __init__(self, out): super().__init__(); self.out = out
        def _complete(self, s, p, m): return self.out
    provider.set_provider(Fake('{"domain":"NLP","keywords":["a b"],"concepts":["c"],"search_queries":["q1","q2"]}'))
    p = PlannerAgent().run({"question": "retrieval augmented generation for science"})["plan"]
    assert p["method"] == "llm" and p["search_queries"] == ["q1", "q2"]
    provider.set_provider(Fake("garbage not json"))
    r = PlannerAgent().run({"question": "retrieval augmented generation for science"})
    assert r["plan"]["method"] == "heuristic" and r["warnings"]


def test_ranking_orders_by_relevance():
    recs = [PaperRecord("Stock market volatility models", abstract="finance returns"),
            PaperRecord("Explainable AI for medical diagnosis", abstract="explainability in clinical diagnosis models", sources=["arxiv", "openalex"])]
    out = rank_records("explainable AI medical diagnosis", recs)
    assert out[0].title.startswith("Explainable") and out[0].relevance > out[1].relevance


def test_discovery_stores_dedupes_and_handles_failure(monkeypatch):
    pid = repo.create_project("t", "q")
    recs = [PaperRecord("Explainable AI in diagnosis", doi="10.1/a", abstract="explainable ai diagnosis", sources=["arxiv"]),
            PaperRecord("explainable ai in diagnosis", doi="10.1/A", abstract="explainable ai diagnosis longer text", sources=["openalex"]),
            PaperRecord("No abstract paper", sources=["arxiv"])]
    from academic_sources.base import deduplicate
    monkeypatch.setattr("agents.discovery.search_all", lambda q, per_source=10: (deduplicate(recs), ["Source 'x' failed: HTTP 429"]))
    out = DiscoveryAgent().run({"project_id": pid, "question": "explainable ai diagnosis", "plan": {"search_queries": ["a"], "keywords": []}})
    assert len(out["candidates"]) == 2 and len(out["selected_ids"]) == 1 and out["warnings"]
    monkeypatch.setattr("agents.discovery.search_all", lambda q, per_source=10: ([], ["all failed"]))
    pid2 = repo.create_project("t2", "q")
    out = DiscoveryAgent().run({"project_id": pid2, "question": "x", "plan": {"search_queries": ["a"]}})
    assert out["selected_ids"] == [] and any("No papers" in w for w in out["warnings"])
