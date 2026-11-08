"""Page 3 — Document Management."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
import common as C

C.setup("Document Management", "🗂️")
st.title("🗂️ Document Management")
pid = C.sidebar()
st.caption("Uploaded PDFs are treated as untrusted data: validated, size-limited, and never allowed to alter agent instructions.")
attach = st.checkbox("Attach uploads to the active project", value=bool(pid), disabled=not pid)
files = st.file_uploader("Upload research PDFs", type=["pdf"], accept_multiple_files=True)
if files and st.button("Process uploads", type="primary"):
    for f in files:
        with st.spinner(f"Processing {f.name}…"):
            res, err = C.api("POST", "/documents", timeout=180, files={"file": (f.name, f.getvalue(), "application/pdf")},
                             data={"project_id": str(pid)} if (attach and pid) else {})
        if err: st.error(f"{f.name}: {err}")
        elif res["status"] == "failed": st.warning(f"{f.name}: {res.get('error')}")
        else:
            st.success(f"{f.name}: {res['pages']} pages, {res['chunks']} chunks indexed.")
            if res.get("injection_flags"): st.warning(f"{f.name} contains instruction-like text on {res['injection_flags']} page(s); it is treated as data only.")
docs, err = C.api("GET", "/documents")
if err: st.error(err)
elif not docs: st.info("No documents uploaded yet.")
else:
    df = pd.DataFrame(docs)
    st.dataframe(df[["filename", "title", "pages", "chunks", "size_bytes", "status", "created_at"]], use_container_width=True, hide_index=True)
    sel = st.selectbox("Inspect / delete document", [d["paper_id"] for d in docs], format_func=lambda i: next(d["filename"] for d in docs if d["paper_id"] == i))
    d = next(x for x in docs if x["paper_id"] == sel)
    st.write(f"**Status:** {C.STATUS_ICON.get(d['status'], '')} {d['status']}" + (f" — {d['error']}" if d["error"] else ""))
    c1, c2 = st.columns([3, 1])
    with c1:
        t, err = C.api("GET", f"/documents/{sel}/text", params={"limit": 15000})
        with st.expander("Extracted text (page and section tagged)", expanded=False):
            st.text(t["text"] if t else err)
    with c2:
        st.json({k: d[k] for k in ("paper_id", "pages", "chunks", "size_bytes", "created_at")})
        if st.button("🗑️ Delete document"):
            _, err = C.api("DELETE", f"/documents/{sel}")
            st.error(err) if err else st.rerun()
