import pytest
from database import repository as repo
from orchestration import workflow, graph
from orchestration.state import WorkflowState
from academic_sources.base import PaperRecord
from evaluation.fixtures.synthetic_corpus import DOCS
from tests.helpers import make_pdf
from document_processing.ingest import ingest_pdf_bytes


@pytest.fixture
def uploaded(tmp_path):
    ids = []
    for k, d in DOCS.items():
        p = make_pdf(tmp_path / f"{k}.pdf", d["pages"])
        r = ingest_pdf_bytes(open(p, "rb").read(), f"{k}.pdf", meta={"title": d["title"], "year": d["year"], "authors": d["authors"]})
        assert r["status"] == "processed"; ids.append(r["paper_id"])
    return ids


def offline(monkeypatch):
    monkeypatch.setattr("agents.discovery.search_all", lambda q, per_source=10: ([], ["offline test"]))


def test_full_workflow_with_uploaded_pdfs(monkeypatch, uploaded):
    offline(monkeypatch)
    pid = workflow.new_project("How can explainable AI improve the reliability of medical diagnosis?")
    final = workflow.run_analysis(pid, uploaded_ids=uploaded)
    assert len(final["analyses"]) == 6 and final["gaps"] and final["verifications"] and final["report_id"]
    runs = repo.list_runs(pid)
    names = [r["agent"] for r in runs]
    for a in ["Research Planner", "Paper Discovery", "Academic Paper Retrieval", "Research Analyst", "Comparison Agent",
              "Research Gap Agent", "Evidence Verification Agent", "Report Generation"]:
        assert a in names, a
    assert all(r["status"] == "completed" for r in runs if r["agent"] != "Search Expansion")
    rep = repo.latest_report(pid)
    for h in ["Abstract", "Introduction", "Literature Review", "Comparative Analysis", "Research Limitations", "Candidate Research Gaps", "Future Directions", "References"]:
        assert f"## {h}" in rep["markdown"]
    assert "[SYNTHETIC]" in rep["markdown"] and "not proof of novelty" in rep["markdown"].lower() or "novel" in rep["markdown"]
    assert repo.list_projects()[0]["status"] in ("completed", "insufficient_evidence")


def test_ablation_flags_skip_agents(monkeypatch, uploaded):
    offline(monkeypatch)
    pid = workflow.new_project("explainable AI for medical diagnosis reliability")
    final = workflow.run_analysis(pid, uploaded_ids=uploaded, config={"enable_comparison": False, "enable_gap": False, "enable_verification": False})
    skipped = [r["agent"] for r in repo.list_runs(pid) if r["status"] == "skipped"]
    assert set(skipped) == {"Comparison Agent", "Research Gap Agent", "Evidence Verification Agent"}
    assert not final.get("gaps") and not final.get("verifications")
    assert "[not verified]" in repo.latest_report(pid)["markdown"]


def test_no_papers_found_ends_gracefully_with_report(monkeypatch):
    offline(monkeypatch)
    pid = workflow.new_project("quantum blockchain consensus for protein folding")
    final = workflow.run_analysis(pid)
    assert final.get("expansion_loops") == 1  # bounded expansion happened once
    assert not final.get("analyses")
    rep = repo.latest_report(pid)
    assert "No papers were analysed" in rep["markdown"] and "could not be found" in rep["markdown"]
    assert repo.list_projects()[0]["status"] == "insufficient_evidence"


def test_agent_failure_is_retried_and_isolated(monkeypatch, uploaded):
    offline(monkeypatch)
    calls = {"n": 0}
    def boom(self, state): calls["n"] += 1; raise RuntimeError("secret sk-abcdefghijklmnop exploded")
    monkeypatch.setattr("agents.comparison.ComparisonAgent.run", boom)
    pid = workflow.new_project("explainable AI for medical diagnosis reliability")
    final = workflow.run_analysis(pid, uploaded_ids=uploaded)
    assert calls["n"] == 2  # 1 try + 1 retry
    failed = [r for r in repo.list_runs(pid) if r["status"] == "failed"]
    assert len(failed) == 2 and all("sk-abc" not in (r["error"] or "") for r in failed)
    assert final["report_id"] and final["gaps"]  # downstream continues


def test_invalid_questions():
    with pytest.raises(ValueError): workflow.new_project("hi")
    with pytest.raises(ValueError): workflow.new_project("x" * 2000)


def test_two_phase_discovery_then_selected_analysis(monkeypatch, uploaded):
    recs = [PaperRecord("Explainable AI for diagnosis", doi="10.1/a", abstract="explainable artificial intelligence improves medical diagnosis reliability", sources=["arxiv"])]
    monkeypatch.setattr("agents.discovery.search_all", lambda q, per_source=10: (recs, []))
    pid = workflow.new_project("explainable AI medical diagnosis reliability")
    st = workflow.discover(pid)
    assert len(st["candidates"]) == 1 and repo.get_query(pid).plan
    final = workflow.run_analysis(pid, selected_ids=[], uploaded_ids=uploaded[:3])
    assert len(final["analyses"]) >= 3
