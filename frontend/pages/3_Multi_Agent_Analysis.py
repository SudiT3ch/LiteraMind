"""Page 4 — Multi-Agent Analysis (shows real execution status from the database; nothing is animated)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
import common as C

C.setup("Multi-Agent Analysis", "🤖")
st.title("🤖 Multi-Agent Analysis")
pid = C.need_project()
EXPECTED = ["Research Planner", "Paper Discovery", "Academic Paper Retrieval", "Research Analyst", "Comparison Agent", "Research Gap Agent", "Evidence Verification Agent", "Report Generation"]
if pid:
    proj, err = C.api("GET", f"/projects/{pid}")
    if err: st.error(err); st.stop()
    st.markdown(f"**Question:** {proj['query']}")
    with st.expander("Run configuration", expanded=not proj["runs"]):
        c = st.columns(4)
        ec = c[0].checkbox("Comparison Agent", True); eg = c[1].checkbox("Research Gap Agent", True)
        ev = c[2].checkbox("Evidence Verification", True); dv = c[3].checkbox("Validate DOIs (Crossref)", False)
        st.caption("Disabling agents is used for ablation experiments; skipped agents are recorded as skipped.")
        if st.button("▶ Start analysis", type="primary", disabled=proj["running"]):
            res, err = C.api("POST", f"/projects/{pid}/run", json={"background": True, "config": {"enable_comparison": ec, "enable_gap": eg, "enable_verification": ev, "validate_dois": dv}})
            st.error(err) if err else st.rerun()
    if proj.get("plan"):
        with st.expander("Research plan"): st.json(proj["plan"])

    @st.fragment(run_every=2 if proj["running"] else None)
    def monitor():
        s, err = C.api("GET", f"/projects/{pid}/runs", timeout=15)
        if err: st.error(err); return
        runs = s["runs"]
        done = {r["agent"] for r in runs if r["status"] in ("completed", "skipped", "failed")}
        current = [r["agent"] for r in runs if r["status"] == "running"]
        if not runs: st.info("No analysis has been run for this project yet."); return
        st.progress(min(1.0, len([a for a in EXPECTED if a in done]) / len(EXPECTED)), text=("Currently executing: " + ", ".join(current)) if current else f"Status: {s['status']}")
        for r in runs:
            icon = C.STATUS_ICON.get(r["status"], "•")
            with st.expander(f"{icon} {r['agent']} — {r['status']}" + (f" · {r['duration_s']}s" if r["duration_s"] is not None else "") + (f" · attempt {r['attempts']}" if r["attempts"] > 1 else ""), expanded=r["status"] in ("running", "failed")):
                if r["summary"]: st.write(r["summary"])
                for w in r["warnings"]: st.warning(w)
                if r["error"]: st.error(r["error"])
                if r["output"]: st.json(r["output"], expanded=False)
        if not s["running"] and s["status"] in ("completed", "insufficient_evidence", "failed"):
            st.success("Analysis finished.") if s["status"] == "completed" else st.warning(f"Run finished with status: {s['status']}. See warnings above.")
    monitor()
    f, err = C.api("GET", f"/projects/{pid}/findings")
    if f:
        st.subheader("Extracted findings")
        df = pd.DataFrame(f)
        titles = {p["paper_id"]: p["title"][:50] for p in proj["papers"]}
        df["paper"] = df["paper_id"].map(titles)
        c = st.columns(2)
        fp = c[0].multiselect("Paper", sorted(df["paper"].dropna().unique())); ff = c[1].multiselect("Field", sorted(df["field"].unique()))
        if fp: df = df[df["paper"].isin(fp)]
        if ff: df = df[df["field"].isin(ff)]
        st.dataframe(df[["paper", "field", "text", "page", "section", "method"]], use_container_width=True, hide_index=True)
