import pytest
from agents.analyst import AnalystAgent
from agents.comparison import ComparisonAgent
from agents.gap_identifier import GapAgent
from agents.verifier import VerifierAgent, judge
from database import repository as repo
from academic_sources.base import PaperRecord
from evaluation.fixtures.synthetic_corpus import DOCS


@pytest.fixture
def state(corpus_store):
    pid = repo.create_project("t", "q")
    for k, d in DOCS.items():
        repo.upsert_paper({"paper_id": k, "title": d["title"], "authors": d["authors"], "year": d["year"], "abstract": "a", "sources": ["fixture"]})
        repo.link_paper(pid, k, 0.5, True)
    s = {"project_id": pid, "plan": {"keywords": ["explainable AI", "medical diagnosis", "quantum blockchain"]}, "config": {}}
    s.update(AnalystAgent().run(s)); s.update(ComparisonAgent().run(s)); s.update(GapAgent().run(s))
    return s


def test_gap_agent_recurring_limitations_with_evidence(state):
    gaps = state["gaps"]
    ext = next(g for g in gaps if g.get("theme") == "external_validation")
    assert ext["confidence"] == "Strong" and len(ext["supporting_papers"]) >= 3
    for g in gaps:
        assert {"description", "supporting_papers", "evidence", "why_it_matters", "future_direction", "confidence", "caveat"} <= set(g)
        assert "novel" in g["caveat"] and g["evidence"]
        assert g["confidence"] in ("Strong", "Moderate", "Weak")
    assert all(e["paper_id"] in DOCS for g in gaps for e in g["evidence"])
    assert gaps == sorted(gaps, key=lambda g: ["Strong", "Moderate", "Weak"].index(g["confidence"]))
    assert repo.list_gaps(state["project_id"])
    cost = next((g for g in gaps if g.get("theme") == "cost"), None)
    assert cost and len(cost["supporting_papers"]) == 2 and cost["confidence"] == "Moderate"


def test_gap_agent_insufficient_papers():
    out = GapAgent().run({"project_id": repo.create_project("t", "q"), "analyses": {"a": {"title": "x", "datasets": [], "algorithms": [], "fulltext_status": "fulltext"}}, "findings": []})
    assert out["gaps"] == [] and any("insufficient" in w.lower() for w in out["warnings"])


def test_gap_agent_flags_abstract_only(state):
    state["analyses"]["syn_xai1"]["fulltext_status"] = "abstract_only"
    assert any("abstract" in w for w in GapAgent().run(state)["warnings"])


def test_judge_statuses():
    p = "The dataset comes from a single hospital and lacks external validation."
    assert judge("lacks external validation", p)["status"] == "Supported"
    assert judge("no external validation", "External validation was performed on three hospitals.")["status"] in ("Contradicted", "Partially supported", "Insufficient evidence")
    assert judge("Accuracy 91% SynthChest", "The model achieves an accuracy of 91% on SynthChest.")["status"] == "Supported"
    assert judge("Accuracy 95% SynthChest", "The model achieves an accuracy of 91% on SynthChest.")["status"] == "Contradicted"
    # metric-aware: the START value of "drops from 90% to 83%" is not the reported accuracy
    drop = "Accuracy drops from 90% to 83% on SynthChest as the privacy budget tightens."
    assert judge("Accuracy 83% SynthChest", drop)["status"] == "Supported"
    assert judge("Accuracy 90% SynthChest", drop)["status"] == "Contradicted"
    assert judge("quantum blockchain consensus", p)["status"] == "Insufficient evidence"
    assert judge("", p)["status"] == "Insufficient evidence"


def test_verifier_end_to_end(state):
    out = VerifierAgent().run(state)
    v = out["verifications"]
    assert v and {x["claim_source"] for x in v} >= {"gap", "comparison", "finding"}
    assert all(x["status"] in ("Supported", "Partially supported", "Contradicted", "Insufficient evidence") for x in v)
    sup = [x for x in v if x["status"] == "Supported"]
    assert sup and all(x["passage"] and x["paper_id"] for x in sup)
    assert out["supported_ratio"] > 0.5 and out["evidence_status"] == "sufficient"
    assert all(r["doi_status"] == "no_doi" for r in out["reference_checks"])
    assert repo.list_verifications(state["project_id"])


def test_verifier_flags_false_claim_and_missing_citation(state):
    ag = VerifierAgent()
    fake = {"claim": "x", "check": "Accuracy 99% SynthChest", "paper_id": "syn_xai1", "claim_source": "test"}
    assert ag.verify_claim(fake, list(state["analyses"]))["status"] == "Contradicted"
    unsup = {"claim": "x", "check": "quantum blockchain consensus protocol", "paper_id": None, "claim_source": "narrative"}
    r = ag.verify_claim(unsup, list(state["analyses"]))
    assert r["status"] == "Insufficient evidence" and r["paper_id"] is None
    unattributed = {"claim": "x", "check": "federated averaging simulated hospitals", "paper_id": None, "claim_source": "narrative"}
    r = ag.verify_claim(unattributed, list(state["analyses"]))
    assert r["paper_id"] == "syn_fl1" and "attributed" in r["note"]


def test_verifier_empty_and_llm_review_rejects_fake_evidence(state):
    assert VerifierAgent().run({"project_id": 1, "analyses": {}})["evidence_status"] == "insufficient"
    from llm import provider
    class Fake(provider.LLMProvider):
        name = "fake"
        def _complete(self, s, p, m): return '[{"i":0,"verdict":"Supported","evidence":"words that never appear in the passage at all"}]'
    provider.set_provider(Fake())
    items = [{"check": "c", "passage": "Some real passage text that is here.", "status": "Insufficient evidence", "method": "heuristic"}]
    VerifierAgent().llm_review(items, [])
    assert items[0]["status"] == "Insufficient evidence"
