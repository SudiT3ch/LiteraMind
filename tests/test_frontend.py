"""Frontend smoke/integration tests: Streamlit AppTest against a REAL uvicorn server (HTTP), no mocks of the API layer."""
import socket, sys, threading, time
from pathlib import Path
import pytest
import uvicorn
from streamlit.testing.v1 import AppTest
from tests.helpers import make_pdf
from evaluation.fixtures.synthetic_corpus import DOCS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontend"))
PAGES = ["frontend/app.py"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "frontend" / "pages").glob("*.py"))


@pytest.fixture
def live(monkeypatch, tmp_path):
    import httpx
    monkeypatch.setattr("agents.discovery.search_all", lambda q, per_source=10: ([], ["offline test"]))
    from backend.main import app
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True); t.start()
    for _ in range(50):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=1).status_code == 200: break
        except Exception: time.sleep(0.1)
    import common
    monkeypatch.setattr(common, "API_URL", f"http://127.0.0.1:{port}")
    yield f"http://127.0.0.1:{port}", tmp_path
    server.should_exit = True; t.join(5)


def test_all_pages_render_with_empty_state(live):
    for p in PAGES:
        at = AppTest.from_file(str(ROOT / p), default_timeout=30).run()
        assert not at.exception, (p, [e.value for e in at.exception])


def test_backend_down_is_reported_not_crashing(monkeypatch):
    import common
    monkeypatch.setattr(common, "API_URL", "http://127.0.0.1:9")
    for p in PAGES:
        at = AppTest.from_file(str(ROOT / p), default_timeout=30).run()
        assert not at.exception, p
    assert any("not reachable" in e.value for e in AppTest.from_file(str(ROOT / "frontend/app.py")).run().error)


def test_pages_with_real_analysis_data(live):
    import httpx
    url, tmp = live
    pid = httpx.post(f"{url}/api/projects", json={"question": "How can explainable AI improve the reliability of medical diagnosis?"}).json()["project_id"]
    for k, d in DOCS.items():
        p = make_pdf(tmp / f"{k}.pdf", d["pages"])
        assert httpx.post(f"{url}/api/documents", files={"file": (f"{k}.pdf", open(p, "rb"), "application/pdf")}, data={"project_id": str(pid), "title": d["title"]}, timeout=60).status_code == 200
    assert httpx.post(f"{url}/api/projects/{pid}/run", json={"background": False}, timeout=180).status_code == 200
    seen = {}
    for p in PAGES:
        at = AppTest.from_file(str(ROOT / p), default_timeout=60).run()
        assert not at.exception, (p, [e.value for e in at.exception])
        seen[p] = at
    gap = seen["frontend/pages/5_Research_Gap_Analysis.py"]
    assert len(gap.expander) >= 3 and any("candidate" in w.value for w in gap.warning)
    ver = seen["frontend/pages/6_Evidence_Verification.py"]
    assert len(ver.metric) == 4 and len(ver.dataframe) >= 1
    cmp_ = seen["frontend/pages/4_Paper_Comparison.py"]
    assert len(cmp_.dataframe) >= 1
    ana = seen["frontend/pages/3_Multi_Agent_Analysis.py"]
    assert any("Research Analyst" in e.label for e in ana.expander)
    rep = seen["frontend/pages/7_Generated_Report.py"]
    assert len(rep.tabs) == 10
    dash = seen["frontend/app.py"]
    assert dash.metric[0].value == "1"


def test_dashboard_create_project_form(live):
    at = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=60).run()
    at.text_area[0].set_value("explainable AI for medical diagnosis reliability")
    at.button[0].click().run()
    import httpx
    url, _ = live
    assert len(httpx.get(f"{url}/api/projects").json()) == 1
    at2 = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=30).run()
    at2.text_area[0].set_value("x"); at2.button[0].click().run()
    assert any("at least 5" in e.value for e in at2.error)


def test_embedded_backend_mode_starts_api_inside_ui(monkeypatch):
    """EMBEDDED_BACKEND=1 lets a single-process host (Streamlit Cloud) run UI + API together."""
    import httpx
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    import common
    monkeypatch.setattr(common, "API_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("EMBEDDED_BACKEND", "1")
    monkeypatch.setattr("agents.discovery.search_all", lambda q, per_source=10: ([], []))
    at = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=3).json() == {"status": "ok"}
    assert not any("not reachable" in e.value for e in at.error)
    at.text_area[0].set_value("explainable AI for medical diagnosis reliability"); at.button[0].click().run()
    assert len(httpx.get(f"http://127.0.0.1:{port}/api/projects").json()) == 1
