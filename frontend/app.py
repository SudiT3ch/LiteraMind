"""Page 1 — Research Dashboard."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pandas as pd
import plotly.express as px
import streamlit as st
import common as C

C.setup("Dashboard", "🏠")
C.sidebar()
st.markdown('<div class="lm-hero"><h1>LiteraMind</h1><p>Discover papers, compare methods, surface candidate research gaps — every claim checked against source passages.</p></div>', unsafe_allow_html=True)

st.subheader("Start a research project")
with st.form("new_project"):
    q = st.text_area("Research question", placeholder="How can Explainable AI improve the reliability of medical diagnosis systems?", height=90)
    go = st.form_submit_button("Create project & discover papers", type="primary")
if go:
    if len(q.strip()) < 5:
        st.error("Please enter a research question (at least 5 characters).")
    else:
        res, err = C.api("POST", "/projects", json={"question": q.strip()})
        if err: st.error(err)
        else:
            st.session_state["project_id"] = res["project_id"]
            with st.spinner("Planning the search and querying academic sources…"):
                d, err = C.api("POST", f"/projects/{res['project_id']}/discover", timeout=180, json={})
            if err: st.warning(f"Project created, but discovery failed: {err}")
            else:
                for w in d.get("warnings", []): st.warning(w)
                st.session_state["discovered"] = res["project_id"]
                st.switch_page("pages/1_Research_Discovery.py")

stats, err = C.api("GET", "/stats")
if err: st.error(err)
else:
    c = st.columns(4)
    c[0].metric("Research projects", stats["projects"]); c[1].metric("Completed analyses", stats["completed"])
    c[2].metric("Papers analysed", stats["papers"]); c[3].metric("Claims verified", stats["claims_verified"])

st.subheader("Recent research projects")
projects, err = C.api("GET", "/projects")
if not projects:
    st.info("No projects yet — ask a research question above to begin.")
else:
    df = pd.DataFrame(projects)
    view = df[["id", "title", "query", "n_papers", "status", "created_at"]].rename(columns={"n_papers": "papers", "created_at": "created"})
    view["status"] = view["status"].map(lambda s: f"{C.STATUS_ICON.get(s, '')} {s}")
    st.dataframe(view.head(10), use_container_width=True, hide_index=True)
    st.subheader("Research activity")
    df["day"] = pd.to_datetime(df["created_at"]).dt.date
    act = df.groupby("day").size().reset_index(name="projects")
    fig = px.bar(act, x="day", y="projects", color_discrete_sequence=["#1f3a5f"])
    fig.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
