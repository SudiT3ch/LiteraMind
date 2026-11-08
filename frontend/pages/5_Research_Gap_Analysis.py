"""Page 6 — Research Gap Analysis."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import plotly.express as px
import streamlit as st
import common as C

C.setup("Research Gap Analysis", "🧭")
st.title("🧭 Research Gap Analysis")
pid = C.need_project()
if pid:
    gaps, err = C.api("GET", f"/projects/{pid}/gaps")
    if err: st.error(err); st.stop()
    if not gaps: st.info("No candidate gaps yet. They require an analysis with ≥2 papers and the Research Gap Agent enabled."); st.stop()
    st.warning("These are **candidate** gaps inferred from the retrieved papers only. They are not proof of novelty; verify with a systematic search.")
    counts = pd.Series([g["confidence"] for g in gaps]).value_counts().reindex(["Strong", "Moderate", "Weak"]).fillna(0).reset_index()
    counts.columns = ["Evidence strength", "Gaps"]
    c = st.columns([1, 2])
    fig = px.bar(counts, x="Evidence strength", y="Gaps", color="Evidence strength", color_discrete_map={"Strong": "#2a9d57", "Moderate": "#d49a1a", "Weak": "#7a8ba3"})
    fig.update_layout(height=230, showlegend=False, plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0)); c[0].plotly_chart(fig, use_container_width=True)
    rep = [g for g in gaps if g["kind"] in ("recurring_limitation",)]
    c[1].subheader("Repeated limitations")
    c[1].dataframe(pd.DataFrame([{"Limitation": g["title"], "Papers": len(g["supporting_papers"]), "Strength": g["confidence"]} for g in rep]) if rep else pd.DataFrame({"info": ["No limitation recurs across ≥2 papers."]}), hide_index=True, use_container_width=True)
    level = st.multiselect("Filter by evidence strength", ["Strong", "Moderate", "Weak"], default=["Strong", "Moderate", "Weak"])
    for g in [g for g in gaps if g["confidence"] in level]:
        with st.expander(f"{g['id']} · {g['title']}  —  {g['confidence']} evidence ({len(g['supporting_papers'])} papers)", expanded=g["confidence"] == "Strong"):
            st.markdown(C.badge(g["confidence"]) + f" &nbsp; <span class='lm-muted'>{g['kind'].replace('_', ' ')}</span>", unsafe_allow_html=True)
            st.write(g["description"]); st.write("**Why it may matter:** " + g["why_it_matters"])
            st.write("**Supporting papers:** " + "; ".join(p["title"] for p in g["supporting_papers"]))
            for e in g["evidence"]:
                loc = ", ".join(x for x in [f"p.{e['page']}" if e["page"] else "", e["section"] or ""] if x)
                st.markdown(f"<div class='lm-quote'>“{e['passage'] or e['text']}”<br><span class='lm-muted'>{e['paper_id']} · {loc}</span></div>", unsafe_allow_html=True)
            st.success("**Suggested future direction:** " + g["future_direction"]); st.caption(g["caveat"])
