import pytest
from agents.analyst import AnalystAgent, paper_sentences
from agents.comparison import ComparisonAgent
from agents.lexicon import find_metrics, find_datasets, find_algorithms, themes_of
from database import repository as repo
from academic_sources.base import PaperRecord
from evaluation.fixtures.synthetic_corpus import DOCS


@pytest.fixture
def project(corpus_store):
    pid = repo.create_project("t", "q")
    for k, d in DOCS.items():
        repo.upsert_paper(PaperRecord(d["title"], d["authors"], d["year"], "abs", doi=None, sources=["fixture"]).to_dict() | {"paper_id": k})
        repo.link_paper(pid, k, 0.5, True)
    return pid


def test_lexicon_extractors():
    assert find_metrics("The model achieves an accuracy of 91% on X.")[0] == {"name": "Accuracy", "value": 0.91, "raw": "91%"}
    m = find_metrics("Accuracy drops from 90% to 83% on SynthChest as budgets tighten.")
    assert m[0]["value"] == 0.83 and "from 90" in m[0]["note"]
    assert find_metrics("AUC of 0.87")[0]["value"] == 0.87
    assert "SynthChest" in find_datasets("Experiments use the SynthChest dataset.") and find_datasets("This dataset is large.") == []
    assert "SHAP" in find_algorithms("We compute SHAP values") and "LIME" not in find_algorithms("a lime green box")
    assert "external_validation" in themes_of("The dataset comes from a single hospital and lacks external validation.")


def test_analyst_extracts_traceable_findings(project, corpus_store):
    out = AnalystAgent().run({"project_id": project})
    a = out["analyses"]["syn_xai1"]
    fields = {f["field"] for f in a["findings"]}
    assert {"methodology", "results", "limitations", "future_work", "algorithm", "dataset", "metric"} <= fields
    lim = [f for f in a["findings"] if f["field"] == "limitations"]
    assert lim and all(f["page"] == 2 and f["section"] == "Limitations" and f["paper_id"] == "syn_xai1" and f["passage"] for f in lim)
    assert a["datasets"] == ["SynthChest"] and any(m["name"] == "Accuracy" and m["value"] == 0.91 and m["dataset"] == "SynthChest" for m in a["metrics"])
    assert repo.list_findings(project)  # persisted


def test_analyst_handles_missing_text(project):
    repo.upsert_paper({"paper_id": "nofile", "title": "No text", "sources": []}); repo.link_paper(project, "nofile", 0.1, True)
    out = AnalystAgent().run({"project_id": project})
    assert any("nofile" in w for w in out["warnings"]) and "nofile" not in out["analyses"]


def test_analyst_llm_drops_hallucinated_quotes(project, corpus_store):
    from llm import provider
    class Fake(provider.LLMProvider):
        name = "fake"
        def _complete(self, s, p, m):
            return ('{"results":[{"text":"x","quote":"The model achieves an accuracy of 91% on SynthChest.","chunk":0},'
                    '{"text":"fake","quote":"This sentence was invented by the model entirely.","chunk":0}],'
                    '"limitations":[{"text":"y","quote":"The dataset comes from a single hospital and lacks external validation.","chunk":0}],'
                    '"methodology":[{"text":"z","quote":"We train a ResNet-50 convolutional neural network and generate Grad-CAM saliency maps.","chunk":0}]}')
    provider.set_provider(Fake())
    warnings = []
    r = AnalystAgent().analyse_llm("syn_xai1", "t", warnings)
    assert r is not None
    llm_f = [f for f in r["findings"] if f["method"] == "llm"]
    assert all("invented" not in f["passage"] for f in llm_f) and any("dropped 1" in w for w in warnings)


def test_comparison_only_compares_matching_metric_and_dataset(project, corpus_store):
    st = AnalystAgent().run({"project_id": project}); st["project_id"] = project
    out = ComparisonAgent().run(st)["comparison"]
    assert len(out["table"]) == 6
    for g in out["comparable_metrics"]:
        assert g["dataset"] and len({e["paper_id"] for e in g["entries"]}) >= 2
    # AUC on SynthEHR: syn_xai2 (0.87) vs syn_fl1 (0.84, 0.87 centralised mention) -> comparable group exists
    assert any(g["metric"] == "AUC" and g["dataset"] == "SynthEHR" for g in out["comparable_metrics"])
    # Accuracy on SynthChest in xai1 (91) and fl2 (83) comparable; BLEU (xai3) is never mixed with accuracy
    assert all(len({e["paper_id"] for e in g["entries"]}) >= 1 for g in out["non_comparable_metrics"])
    assert not any(g["metric"] == "BLEU" for g in out["comparable_metrics"])
    assert out["similarities"] and out["methodological_trends"]
    assert repo.get_comparison(project)


def test_comparison_needs_two_papers():
    out = ComparisonAgent().run({"project_id": repo.create_project("t", "q"), "analyses": {"a": {}}, "findings": []})
    assert out["comparison"]["table"] == [] and out["warnings"]
