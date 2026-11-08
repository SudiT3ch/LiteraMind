"""Page 5 — Paper Comparison."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import plotly.express as px
import streamlit as st
import common as C
from visualization.plotting import json_to_graph, to_plotly

C.setup("Paper Comparison", "⚖️")
st.title("⚖️ Paper Comparison")
pid = C.need_project()
if pid:
    comp, err = C.api("GET", f"/projects/{pid}/comparison")
    if err: st.error(err); st.stop()
    if not comp or not comp.get("table"):
        st.info("No comparison available yet. Run the analysis (with the Comparison Agent enabled) on at least two papers."); st.stop()
    rows = comp["table"]
    names = {r["paper_id"]: f"{r['title'][:60]} ({r['year'] or 'n.d.'})" for r in rows}
    chosen = st.multiselect("Papers to compare", list(names), default=list(names), format_func=lambda i: names[i])
    sel = [r for r in rows if r["paper_id"] in chosen]
    df = pd.DataFrame([{"Paper": r["title"][:70], "Year": r["year"], "Objective": r["objective"], "Methodology": r["methodology"], "Algorithms": ", ".join(r["algorithms"]),
                        "Dataset": ", ".join(r["datasets"]), "Results": " | ".join(r["results"]), "Limitations": " | ".join(r["limitations"]), "Coverage": r["coverage"]} for r in sel])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(comp.get("metric_note", ""))
    st.subheader("Directly comparable results")
    comparable = [g for g in comp.get("comparable_metrics", []) if sum(e["paper_id"] in chosen for e in g["entries"]) >= 2]
    if not comparable: st.info("No metric is reported by two or more selected papers on the same dataset, so no numeric comparison is shown.")
    for g in comparable:
        d = pd.DataFrame([e for e in g["entries"] if e["paper_id"] in chosen])
        fig = px.bar(d, x="title", y="value", text="raw", color_discrete_sequence=["#1f3a5f"], title=f"{g['metric']} on {g['dataset']}")
        fig.update_layout(height=280, plot_bgcolor="white", xaxis_title=None, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Non-comparable metrics (different datasets/metrics — not mixed)"):
        for g in comp.get("non_comparable_metrics", []):
            st.write(f"**{g['metric']}** on {g['dataset'] or 'unspecified dataset'} — {g['reason']}: " + "; ".join(f"{e['title']} = {e['raw']}" for e in g["entries"]))
    c = st.columns(2)
    with c[0]:
        st.subheader("Similarities"); [st.write(f"- {s}") for s in comp.get("similarities", [])] or st.write("—")
        st.subheader("Methodological trends"); [st.write(f"- **{t['family']}** — {len(t['papers'])} paper(s)") for t in comp.get("methodological_trends", [])]
    with c[1]:
        st.subheader("Differences"); [st.write(f"- {s}") for s in comp.get("differences", [])] or st.write("—")
        st.subheader("Divergent findings"); [st.write(f"- {x['metric']} on {x['dataset']}: spread {x['spread']} — {x['note']}") for x in comp.get("contradictory_findings", [])] or st.write("None detected.")
    st.subheader("Strengths and weaknesses")
    for s in comp.get("strengths_weaknesses", []):
        if s["paper_id"] in chosen:
            with st.expander(s["title"]): st.write("**Strength (reported):** " + (" ".join(s["strengths"]) or "—")); st.write("**Weaknesses (stated limitations):** " + (" ".join(s["weaknesses"]) or "—"))
    if comp.get("narrative"): st.subheader("Narrative (LLM-written from extracted facts)"); st.write(comp["narrative"])
    with st.expander("🕸️ Research knowledge graph"):
        g, err = C.api("GET", f"/projects/{pid}/graph")
        if g and g["nodes"]: st.plotly_chart(to_plotly(json_to_graph(g)), use_container_width=True)
        else: st.info(err or "No graph data yet.")
