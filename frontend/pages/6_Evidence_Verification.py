"""Page 7 — Evidence Verification."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
import common as C

C.setup("Evidence Verification", "🔬")
st.title("🔬 Evidence Verification")
pid = C.need_project()
if pid:
    v, err = C.api("GET", f"/projects/{pid}/verifications")
    if err: st.error(err); st.stop()
    if not v: st.info("No verification results yet. Run the analysis with the Evidence Verification Agent enabled."); st.stop()
    df = pd.DataFrame(v)
    st.caption("Heuristic judge: content-word coverage, numeric and negation consistency against passages re-retrieved from the cited paper. Not a substitute for human fact-checking.")
    order = ["Supported", "Partially supported", "Contradicted", "Insufficient evidence"]
    cols = st.columns(4)
    for c, s in zip(cols, order): c.metric(s, int((df["status"] == s).sum()))
    f = st.columns(2)
    fs = f[0].multiselect("Status", order, default=order); fsrc = f[1].multiselect("Claim origin", sorted(df["claim_source"].unique()), default=sorted(df["claim_source"].unique()))
    view = df[df["status"].isin(fs) & df["claim_source"].isin(fsrc)]
    st.dataframe(view[["status", "claim_source", "claim", "paper_id", "page", "section", "score", "method"]], use_container_width=True, hide_index=True)
    st.subheader("Source passages")
    for _, r in view.iterrows():
        with st.expander(f"{r['status']} · {r['claim'][:110]}"):
            st.markdown(C.badge(r["status"].split()[0] if r["status"] != "Insufficient evidence" else "Insufficient", r["status"].split()[0]), unsafe_allow_html=True)
            st.write("**Claim:** " + r["claim"])
            loc = ", ".join(x for x in [f"p.{int(r['page'])}" if pd.notna(r["page"]) and r["page"] else "", r["section"] or ""] if x)
            st.markdown(f"<div class='lm-quote'>{r['passage'] or '<i>No supporting passage found.</i>'}<br><span class='lm-muted'>{r['paper_id'] or 'no source'} · {loc}</span></div>", unsafe_allow_html=True)
            for c in r["contradicting"] or []:
                st.warning(f"Conflicting evidence in {c['paper_id']} (p.{c['page']}, {c['section']}): {c['passage']}")
    s, _ = C.api("GET", f"/projects/{pid}/runs")
    ver = next((r for r in (s or {}).get("runs", []) if r["agent"].startswith("Evidence")), None)
    if ver and ver["summary"]: st.caption("Agent summary: " + ver["summary"])
