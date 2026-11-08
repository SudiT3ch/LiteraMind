import io, pytest
from fastapi.testclient import TestClient
from tests.helpers import make_pdf
from evaluation.fixtures.synthetic_corpus import DOCS
from reports import generator
from visualization import knowledge_graph as kg


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("agents.discovery.search_all", lambda q, per_source=10: ([], ["offline"]))
    from backend.main import app
    from database.database import init_db
    with TestClient(app) as c:
        yield c


def upload(client, tmp_path, key, project_id=None):
    p = make_pdf(tmp_path / f"{key}.pdf", DOCS[key]["pages"])
    return client.post("/api/documents", files={"file": (f"{key}.pdf", open(p, "rb"), "application/pdf")},
                       data={"title": DOCS[key]["title"], **({"project_id": str(project_id)} if project_id else {})})


def test_health_config_never_leaks_keys(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecretvalue123")
    assert client.get("/api/health").json() == {"status": "ok"}
    body = client.get("/api/config").text
    assert "supersecret" not in body and "llm_provider" in body


def test_document_lifecycle_and_upload_restrictions(client, tmp_path):
    r = upload(client, tmp_path, "syn_xai1"); assert r.status_code == 200 and r.json()["status"] == "processed"
    pid = r.json()["paper_id"]
    docs = client.get("/api/documents").json(); assert len(docs) == 1 and docs[0]["chunks"] > 0
    t = client.get(f"/api/documents/{pid}/text").json(); assert "[p.2" in t["text"]
    assert client.post("/api/documents", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/api/documents", files={"file": ("a.pdf", b"not really a pdf", "application/pdf")}).status_code == 422
    assert client.post("/api/documents", files={"file": ("../../etc/x.pdf", b"", "application/pdf")}).status_code == 422
    assert client.delete(f"/api/documents/{pid}").status_code == 200 and client.get("/api/documents").json() == []
    assert client.delete(f"/api/documents/{pid}").status_code == 404


def test_oversize_upload_rejected(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    from config import reset_settings; reset_settings()
    r = client.post("/api/documents", files={"file": ("big.pdf", b"%PDF-" + b"0" * (2 * 1024 * 1024), "application/pdf")})
    assert r.status_code == 413


def test_full_api_workflow_and_exports(client, tmp_path):
    pid = client.post("/api/projects", json={"question": "How can explainable AI improve the reliability of medical diagnosis?"}).json()["project_id"]
    assert client.post("/api/projects", json={"question": "hi"}).status_code == 422
    for k in DOCS: assert upload(client, tmp_path, k, pid).status_code == 200
    r = client.post(f"/api/projects/{pid}/run", json={"background": False}).json()
    assert r["status"] in ("completed", None) or r.get("warnings") is not None
    p = client.get(f"/api/projects/{pid}").json()
    assert p["status"] in ("completed", "insufficient_evidence") and len(p["papers"]) == 6 and p["runs"]
    assert client.get(f"/api/projects/{pid}/gaps").json() and client.get(f"/api/projects/{pid}/verifications").json()
    assert client.get(f"/api/projects/{pid}/comparison").json()["table"] and client.get(f"/api/projects/{pid}/findings").json()
    g = client.get(f"/api/projects/{pid}/graph").json()
    kinds = {n["kind"] for n in g["nodes"]}
    assert kinds >= {"paper", "algorithm", "dataset", "limitation"} and g["edges"]
    assert "author" not in kinds  # uploads carry no author metadata; none is invented
    from database.database import session_scope
    from database.models import Paper
    with session_scope() as s:
        s.query(Paper).filter(Paper.paper_id.like("upload_%")).first().authors = ["Ada Lovelace"]
    assert "author" in {n["kind"] for n in client.get(f"/api/projects/{pid}/graph").json()["nodes"]}
    md = client.get(f"/api/projects/{pid}/report/download?format=md"); assert md.status_code == 200 and b"## References" in md.content
    pdf = client.get(f"/api/projects/{pid}/report/download?format=pdf"); assert pdf.content[:5] == b"%PDF-" and len(pdf.content) > 3000
    docx = client.get(f"/api/projects/{pid}/report/download?format=docx"); assert docx.content[:2] == b"PK"
    import docx as D; d = D.Document(io.BytesIO(docx.content)); assert any("Candidate Research Gaps" in p.text for p in d.paragraphs)
    import fitz
    assert "Literature Review" in fitz.open(stream=pdf.content, filetype="pdf")[0].get_text()
    assert client.get(f"/api/projects/{pid}/report/download?format=exe").status_code == 400


def test_404s_and_errors(client):
    assert client.get("/api/projects/999").status_code == 404
    assert client.get("/api/projects/999/report").status_code == 404
    pid = client.post("/api/projects", json={"question": "explainable AI medical diagnosis"}).json()["project_id"]
    assert client.get(f"/api/projects/{pid}/report").status_code == 404


def test_background_run_and_status(client, tmp_path):
    import time
    pid = client.post("/api/projects", json={"question": "explainable AI for medical diagnosis reliability"}).json()["project_id"]
    for k in list(DOCS)[:3]: upload(client, tmp_path, k, pid)
    assert client.post(f"/api/projects/{pid}/run", json={}).json()["started"] is True
    for _ in range(60):
        s = client.get(f"/api/projects/{pid}/runs").json()
        if not s["running"]: break
        time.sleep(0.5)
    assert not s["running"] and any(r["agent"] == "Report Generation" for r in s["runs"])


def test_pdf_export_handles_special_characters(tmp_path):
    md = "# T & <x>\n\n## S\n\nText with **bold** and _it_ ⚠ <b>\n\n| a | b |\n|---|---|\n| 1 | <2> |\n\n- item & more"
    assert open(generator.to_pdf(md, tmp_path / "a.pdf"), "rb").read()[:5] == b"%PDF-"
    assert generator.to_docx(md, tmp_path / "a.docx")
