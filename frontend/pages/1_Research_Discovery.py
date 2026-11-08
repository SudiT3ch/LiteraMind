"""Page 2 — Research Discovery."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
import common as C

C.setup("Research Discovery", "🔎")
st.title("🔎 Research Discovery")
pid = C.need_project()
if pid:
    proj, err = C.api("GET", f"/projects/{pid}")
    if err: st.error(err); st.stop()
    st.markdown(f"**Question:** {proj['query']}")
    if proj.get("plan"):
        p = proj["plan"]
        with st.expander("Research plan", expanded=False):
            st.write(f"**Domain:** {p.get('domain')}  ·  planner: `{p.get('method')}`")
            st.write("**Keywords:** " + ", ".join(p.get("keywords", [])))
            st.write("**Related concepts:** " + ", ".join(p.get("concepts", [])))
            st.write("**Search queries:**"); [st.write(f"- {q}") for q in p.get("search_queries", [])]
    if st.button("Run / refresh discovery", type="primary"):
        with st.spinner("Querying arXiv, Semantic Scholar and OpenAlex…"):
            d, err = C.api("POST", f"/projects/{pid}/discover", timeout=180, json={})
        if err: st.error(err)
        else:
            for w in d.get("warnings", []): st.warning(w)
            st.rerun()
    papers = [x for x in proj["papers"] if not x["is_uploaded"]]
    uploaded = [x for x in proj["papers"] if x["is_uploaded"]]
    if not papers:
        st.info("No discovered papers yet. Click **Run / refresh discovery**, or upload PDFs on the Document Management page.")
    else:
        st.subheader(f"{len(papers)} candidate papers")
        for x in papers:
            with st.container():
                links = []
                if x["doi"]: links.append(f"[DOI](https://doi.org/{x['doi']})")
                if x["url"]: links.append(f"[Source]({x['url']})")
                if x["pdf_url"]: links.append(f"[PDF]({x['pdf_url']})")
                st.markdown(f'<div class="lm-card"><b>{x["title"]}</b><br><span class="lm-muted">{", ".join(x["authors"][:5]) or "Unknown authors"} · {x["year"] or "n.d."} · {x["venue"] or "venue n/a"} · sources: {", ".join(x["sources"])}</span></div>', unsafe_allow_html=True)
                c = st.columns([1, 1, 3])
                c[0].progress(min(1.0, float(x["relevance"])), text=f"Relevance {x['relevance']:.2f}")
                c[1].write("📄 PDF available" if x["pdf_url"] else "📄 No open PDF (abstract only)")
                c[2].markdown(" · ".join(links) or "_no links_")
                with st.expander("Abstract"):
                    st.write(x["abstract"] or "_No abstract available from the source._")
        st.subheader("Select papers for analysis")
        df = pd.DataFrame([{"Select": x["selected"], "Title": x["title"][:90], "Year": x["year"], "Relevance": round(x["relevance"], 3), "paper_id": x["paper_id"]} for x in papers])
        edited = st.data_editor(df, hide_index=True, use_container_width=True, disabled=["Title", "Year", "Relevance", "paper_id"], key="sel")
        chosen = edited[edited["Select"]]["paper_id"].tolist()
        st.caption(f"{len(chosen)} discovered + {len(uploaded)} uploaded document(s) will be analysed.")
        if st.button("Analyse selected papers", type="primary", disabled=not (chosen or uploaded)):
            C.api("POST", f"/projects/{pid}/select", json={"paper_ids": chosen + [u['paper_id'] for u in uploaded]})
            res, err = C.api("POST", f"/projects/{pid}/run", json={"paper_ids": chosen, "background": True})
            if err: st.error(err)
            else: st.switch_page("pages/3_Multi_Agent_Analysis.py")
