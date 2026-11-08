"""Page 8 — Generated Research Report."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import streamlit as st
import common as C

C.setup("Generated Report", "📝")
st.title("📝 Generated Research Report")
pid = C.need_project()
if pid:
    r, err = C.api("GET", f"/projects/{pid}/report")
    if err: st.info("No report yet — run an analysis first." if "No report" in err else err); st.stop()
    st.caption(f"Generated {r['created_at']} · evidence status: **{r['payload'].get('evidence_status')}** · papers: {r['payload'].get('n_papers')}")
    cols = st.columns(3)
    for col, (fmt, mime) in zip(cols, [("pdf", "application/pdf"), ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"), ("md", "text/markdown")]):
        data, e = C.api("GET", f"/projects/{pid}/report/download", params={"format": fmt}, timeout=120)
        if e: col.error(e)
        else: col.download_button(f"⬇ Download {fmt.upper()}", data, file_name=f"literamind_report_{pid}.{fmt}", mime=mime, use_container_width=True)
    st.divider()
    tabs = st.tabs([s["heading"] for s in r["payload"]["sections"]])
    for t, s in zip(tabs, r["payload"]["sections"]):
        with t: st.markdown(s["body"])
