"""Shared Streamlit helpers: API client (talks to the FastAPI backend), theme, sidebar, project selection."""
from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import httpx
import streamlit as st

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
STATUS_ICON = {"completed": "✅", "running": "⏳", "failed": "❌", "skipped": "⏭️", "insufficient_evidence": "⚠️", "created": "🆕", "discovered": "🔎"}
CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1, h2, h3 {color: #1f3a5f; letter-spacing: -0.01em;}
.lm-hero {background: linear-gradient(135deg,#1f3a5f 0%,#2f5d8a 100%); color: #fff; padding: 1.6rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem;}
.lm-hero h1 {color:#fff; margin:0 0 .3rem 0; font-size: 2rem;} .lm-hero p {margin:0; opacity:.9;}
.lm-badge {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600; color:#fff;}
.lm-Strong,.lm-Supported {background:#2a9d57;} .lm-Moderate,.lm-Partially {background:#d49a1a;} .lm-Weak,.lm-Insufficient {background:#7a8ba3;} .lm-Contradicted {background:#b3261e;}
.lm-card {border: 1px solid #dfe5ee; border-radius: 12px; padding: .9rem 1.1rem; margin-bottom: .7rem; background:#fff;}
.lm-muted {color:#667085; font-size:.88rem;}
.lm-quote {border-left: 3px solid #2f5d8a; padding: .3rem .8rem; background:#f3f6fa; border-radius: 0 8px 8px 0; font-size:.9rem;}
</style>
"""


def api(method: str, path: str, timeout: float = 60, **kw):
    """Returns (data, error_message). Never raises; error messages are safe to show."""
    try:
        r = httpx.request(method, f"{API_URL}/api{path}", timeout=timeout, **kw)
    except httpx.ConnectError:
        return None, f"Backend not reachable at {API_URL}. Start it with `python run.py` (or `uvicorn backend.main:app`)."
    except httpx.TimeoutException:
        return None, "The request timed out. The analysis may still be running; check the Multi-Agent Analysis page."
    except httpx.HTTPError as e:
        return None, f"Network error: {type(e).__name__}"
    if r.status_code >= 400:
        try: detail = r.json().get("detail", r.text)
        except Exception: detail = r.text
        if isinstance(detail, list): detail = "; ".join(str(d.get("msg", d)) for d in detail)
        return None, str(detail)[:400]
    ctype = r.headers.get("content-type", "")
    return (r.json() if "json" in ctype else r.content), None


@st.cache_resource(show_spinner="Starting the analysis engine…")
def _start_embedded_backend(url: str):
    """Run the FastAPI backend inside this Streamlit process (for hosts that run a single app, e.g. Streamlit Community Cloud)."""
    import threading, time
    import uvicorn
    from urllib.parse import urlparse
    from backend.main import app
    port = urlparse(url).port or 8000
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="embedded-backend").start()
    for _ in range(100):  # wait up to ~10 s for startup
        try:
            if httpx.get(f"{url}/api/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    return server


def ensure_backend() -> None:
    """Opt-in: set EMBEDDED_BACKEND=1 (env var or Streamlit secret) to host the API inside the UI process."""
    if os.getenv("EMBEDDED_BACKEND", "0") == "1":
        _start_embedded_backend(API_URL)


def setup(title: str, icon: str = "📚"):
    st.set_page_config(page_title=f"{title} · LiteraMind", page_icon=icon, layout="wide")
    ensure_backend()
    st.markdown(CSS, unsafe_allow_html=True)


def badge(label: str, cls: str | None = None) -> str:
    return f'<span class="lm-badge lm-{(cls or label).split()[0]}">{label}</span>'


def sidebar() -> int | None:
    """Sidebar: branding, backend/LLM status, project selector. Returns the active project id."""
    with st.sidebar:
        st.markdown("### 📚 LiteraMind")
        st.caption("Evidence-grounded multi-agent literature analysis")
        cfg, err = api("GET", "/config", timeout=5)
        if err: st.error(err)
        else:
            st.caption(f"LLM: **{cfg['llm_provider']}**" + ("" if cfg["llm_available"] else " (rule-based mode)") + f" · Embeddings: **{cfg['embedding_backend']}**")
        projects, _ = api("GET", "/projects", timeout=10)
        projects = projects or []
        if projects:
            labels = {p["id"]: f"#{p['id']} · {p['title'][:38]} {STATUS_ICON.get(p['status'], '')}" for p in projects}
            ids = list(labels)
            cur = st.session_state.get("project_id")
            idx = ids.index(cur) if cur in ids else 0
            st.session_state["project_id"] = st.selectbox("Active project", ids, index=idx, format_func=lambda i: labels[i])
        else:
            st.info("No projects yet. Create one on the Dashboard.")
            st.session_state["project_id"] = None
    return st.session_state.get("project_id")


def need_project() -> int | None:
    pid = sidebar()
    if pid is None:
        st.info("Create or select a research project from the Dashboard first.")
    return pid
