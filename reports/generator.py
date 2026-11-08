"""Report Generation component (not an agent). Assembles the literature review from verified pipeline outputs only —
no free-form LLM prose, so every sentence traces to an extracted finding with paper/page/section. Exports Markdown, PDF, DOCX."""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
from agents.base import short_title
from database import repository as repo


def _cite(idx: dict, pid: str) -> str:
    return f"[{idx[pid]}]" if pid in idx else "[?]"


def _loc(page, section) -> str:
    bits = []
    if page: bits.append(f"p.{page}")
    if section: bits.append(str(section))
    return f" ({', '.join(bits)})" if bits else ""


def _reference(n: int, p: dict) -> str:
    authors = ", ".join(p["authors"][:6]) + (" et al." if len(p["authors"]) > 6 else "") if p.get("authors") else "Unknown authors"
    s = f"[{n}] {authors} ({p.get('year') or 'n.d.'}). {p['title']}."
    if p.get("venue"): s += f" {p['venue']}."
    if p.get("doi"): s += f" https://doi.org/{p['doi']}"
    elif p.get("url"): s += f" {p['url']}"
    if p.get("is_uploaded"): s += " (user-uploaded document)"
    return s


def build_report(state: dict) -> dict:
    pid_project = state["project_id"]
    q = state.get("question", "")
    plan = state.get("plan") or {}
    analyses = state.get("analyses") or {}
    findings = state.get("findings") or []
    comp = state.get("comparison") or {}
    gaps = state.get("gaps") or []
    vers = state.get("verifications") or []
    refs_chk = state.get("reference_checks") or []
    cov = state.get("coverage") or {}
    cfg = state.get("config") or {}
    papers = {p["paper_id"]: p for p in repo.project_papers(pid_project, selected_only=True) if p["paper_id"] in analyses}
    order = sorted(papers.values(), key=lambda p: (p.get("year") or 0, p["title"]))
    idx = {p["paper_id"]: i + 1 for i, p in enumerate(order)}
    status_of = {(v["paper_id"], v.get("check")): v["status"] for v in vers}
    verified_on = bool(vers)

    def tag(pid, text) -> str:
        if not verified_on: return " _[not verified]_"
        s = status_of.get((pid, text))
        return "" if s == "Supported" else f" _[{s.lower()}]_" if s else " _[not individually verified]_"

    n = len(order)
    counts = Counter(v["status"] for v in vers)
    gstr = Counter(g["confidence"] for g in gaps)
    title = f"Literature Review: {q.strip().rstrip('?')}"
    insufficient = state.get("evidence_status") == "insufficient" or n == 0
    # ---------------- sections ----------------
    abstract = (f"This automatically generated review addresses the question \"{q}\". It analyses {n} paper(s) retrieved from "
                f"{', '.join(cov.get('sources') or ['uploaded documents']) or 'the supplied sources'} "
                f"({cov.get('fulltext', 0)} full text, {cov.get('abstract_only', 0)} abstract-only). ")
    if comp.get("table"): abstract += f"Methods, datasets and reported metrics were compared across papers. "
    if gaps: abstract += f"{len(gaps)} candidate research gap(s) were identified ({gstr.get('Strong',0)} strong, {gstr.get('Moderate',0)} moderate, {gstr.get('Weak',0)} weak evidence). "
    if vers: abstract += f"{len(vers)} generated claim(s) were checked against source passages: {counts.get('Supported',0)} supported, {counts.get('Partially supported',0)} partially supported, {counts.get('Contradicted',0)} contradicted, {counts.get('Insufficient evidence',0)} with insufficient evidence. "
    if insufficient: abstract += "**Sufficient evidence could not be found; conclusions below are tentative.**"
    intro = (f"The research question concerns *{q}*. The planner classified the domain as **{plan.get('domain', 'General Research')}** and derived the keywords "
             f"{', '.join(plan.get('keywords', [])[:8]) or 'n/a'}. Search queries used: {'; '.join(plan.get('search_queries', [])) or 'n/a'}. "
             "All statements below are extracted from the cited papers and carry page/section locations where available; "
             "findings from abstract-only papers have no page information.")
    lit = []
    for p in order:
        a = analyses[p["paper_id"]]; pid = p["paper_id"]
        get = lambda fld, k=1: [f for f in findings if f["paper_id"] == pid and f["field"] == fld][:k]
        lines = [f"**{short_title(p['title'], 120)}** ({p.get('year') or 'n.d.'}) {_cite(idx, pid)}"]
        for lbl, fld, k in (("Objective", "objectives", 1), ("Method", "methodology", 1), ("Key result", "results", 2)):
            for f in get(fld, k):
                lines.append(f"- {lbl}: {f['text']}{_loc(f['page'], f['section'])}{tag(pid, f['text'])}")
        if a["datasets"]: lines.append(f"- Datasets: {', '.join(a['datasets'])}")
        if a["algorithms"]: lines.append(f"- Algorithms/models: {', '.join(a['algorithms'])}")
        if p.get("fulltext_status") == "abstract_only": lines.append("- _Analysed from abstract only._")
        lit.append("\n".join(lines))
    # comparison
    comp_md = ["No comparison available (fewer than two analysed papers, or the Comparison Agent was disabled)."]
    if comp.get("table"):
        rows = ["| Paper | Methods | Datasets | Reported metrics |", "|---|---|---|---|"]
        for r in comp["table"]:
            ms = "; ".join(f"{m['name']} {m['raw']}" + (f" ({m['dataset']})" if m.get("dataset") else "") for m in r["metrics"][:4]) or "—"
            rows.append(f"| {short_title(r['title'], 50)} {_cite(idx, r['paper_id'])} | {', '.join(r['algorithms'][:4]) or '—'} | {', '.join(r['datasets'][:3]) or '—'} | {ms} |")
        comp_md = rows + ["", comp.get("metric_note", "")]
        if comp.get("comparable_metrics"):
            comp_md += ["", "**Directly comparable results (same metric and dataset):**"]
            for g in comp["comparable_metrics"]:
                comp_md.append(f"- {g['metric']} on {g['dataset']}: " + "; ".join(f"{e['title']} {_cite(idx, e['paper_id'])} = {e['raw']}" for e in g["entries"]))
        if comp.get("similarities"): comp_md += ["", "**Similarities:**"] + [f"- {s}" for s in comp["similarities"][:8]]
        if comp.get("differences"): comp_md += ["", "**Differences:**"] + [f"- {s}" for s in comp["differences"][:8]]
        if comp.get("methodological_trends"): comp_md += ["", "**Methodological trends:**"] + [f"- {t['family']}: {len(t['papers'])} paper(s)" for t in comp["methodological_trends"][:8]]
        if comp.get("contradictory_findings"): comp_md += ["", "**Divergent findings:**"] + [f"- {c['metric']} on {c['dataset']}: spread {c['spread']} — {c['note']}" for c in comp["contradictory_findings"]]
    lims = []
    for p in order:
        for f in [f for f in findings if f["paper_id"] == p["paper_id"] and f["field"] == "limitations"][:3]:
            lims.append(f"- {f['text']} {_cite(idx, p['paper_id'])}{_loc(f['page'], f['section'])}{tag(p['paper_id'], f['text'])}")
    gap_md = [f"_{(gaps[0]['caveat'] if gaps else 'No candidate gaps could be derived from the available evidence.')}_"] if gaps else ["No candidate research gaps could be derived (insufficient or non-recurring evidence)."]
    for g in gaps:
        gap_md += ["", f"**{g['id']}. {g['title']}** — evidence strength: **{g['confidence']}** ({len(g['supporting_papers'])} of {n} papers)", g["description"], f"- Why it may matter: {g['why_it_matters']}"]
        for e in g["evidence"][:4]:
            gap_md.append(f"- Evidence {_cite(idx, e['paper_id'])}{_loc(e['page'], e['section'])}: “{e['passage'][:220]}”")
        gap_md.append(f"- Suggested direction: {g['future_direction']}")
    future = [f"- {g['future_direction']} (from {g['id']}, {g['confidence']} evidence)" for g in gaps]
    for p in order:
        for f in [f for f in findings if f["paper_id"] == p["paper_id"] and f["field"] == "future_work"][:2]:
            future.append(f"- {f['text']} {_cite(idx, p['paper_id'])}{_loc(f['page'], f['section'])} (author-stated)")
    ver_md = ["Claim verification was not run." if not vers else
              f"{len(vers)} claims checked — Supported: {counts.get('Supported',0)}, Partially supported: {counts.get('Partially supported',0)}, "
              f"Contradicted: {counts.get('Contradicted',0)}, Insufficient evidence: {counts.get('Insufficient evidence',0)}."]
    for v in [v for v in vers if v["status"] in ("Contradicted", "Insufficient evidence")][:10]:
        ver_md.append(f"- ⚠ {v['status']}: {v['claim']}")
    if refs_chk:
        ver_md.append("")
        ver_md.append("Reference metadata checks: " + "; ".join(f"{_cite(idx, r['paper_id'])} DOI {r['doi_status'].replace('_',' ')}" + (f" (missing {', '.join(r['metadata_missing'])})" if r["metadata_missing"] else "") for r in refs_chk if r["paper_id"] in idx))
    notes = ["Findings come from " + ("an LLM-assisted pipeline with verbatim-quote validation." if cfg.get("llm_used") else "a rule-based extraction pipeline (no LLM configured); extraction is cue-phrase based and may miss nuance."),
             "Verification is a lexical-entailment approximation, not a human fact-check.",
             "Candidate research gaps are suggestions grounded in a limited paper set, not proof of novelty."] + [f"Warning: {w}" for w in (state.get("warnings") or [])[:8]]
    sections = [("Abstract", abstract), ("Introduction", intro), ("Literature Review", "\n\n".join(lit) or "No papers were analysed."),
                ("Comparative Analysis", "\n".join(comp_md)), ("Research Limitations", "\n".join(lims) or "No limitations were extracted."),
                ("Candidate Research Gaps", "\n".join(gap_md)), ("Future Directions", "\n".join(future) or "No future directions were derived."),
                ("Evidence Verification Summary", "\n".join(ver_md)), ("Methodology and Coverage Notes", "\n".join(f"- {x}" for x in notes)),
                ("References", "\n".join(_reference(idx[p["paper_id"]], p) for p in order) or "No references.")]
    md = f"# {title}\n\n" + "\n\n".join(f"## {h}\n\n{b}" for h, b in sections)
    payload = {"title": title, "sections": [{"heading": h, "body": b} for h, b in sections], "n_papers": n, "evidence_status": state.get("evidence_status", "not_checked")}
    rid = repo.save_report(pid_project, title, md, payload)
    return {"report_id": rid, "title": title, "markdown": md, "payload": payload}


# ---------------- exporters ----------------
def _inline(s: str) -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])[_*]([^_*]+?)[_*](?![\w*])", r"<i>\1</i>", s)
    return s.replace("⚠", "[!]")


def to_pdf(markdown: str, path: str | Path) -> str:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
    ss = getSampleStyleSheet(); body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    cell = ParagraphStyle("c", parent=body, fontSize=8, leading=10)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    story, bullets, table = [], [], []

    def flush():
        nonlocal bullets, table
        if bullets:
            story.append(ListFlowable([ListItem(Paragraph(_inline(b), body)) for b in bullets], bulletType="bullet", leftIndent=14)); bullets = []
        if table:
            rows = [[Paragraph(_inline(c.strip()), cell) for c in r.strip().strip("|").split("|")] for r in table if not re.match(r"^\|?\s*-{2,}", r.strip())]
            w = (A4[0] - 4*cm) / len(rows[0]); t = Table(rows, colWidths=[w] * len(rows[0]), repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story.append(t); story.append(Spacer(1, 6)); table = []
    for line in markdown.splitlines():
        if line.startswith("|"): table.append(line); continue
        if table: flush()
        if line.startswith("- "): bullets.append(line[2:]); continue
        if bullets: flush()
        if line.startswith("# "): story.append(Paragraph(_inline(line[2:]), ss["Title"]))
        elif line.startswith("## "): story += [Spacer(1, 8), Paragraph(_inline(line[3:]), ss["Heading2"])]
        elif line.strip(): story.append(Paragraph(_inline(line), body)); story.append(Spacer(1, 3))
    flush()
    doc.build(story)
    return str(path)


def to_docx(markdown: str, path: str | Path) -> str:
    from docx import Document
    d = Document()
    def runs(par, text):
        for part in re.split(r"(\*\*.+?\*\*)", text):
            if part.startswith("**") and part.endswith("**"): par.add_run(part[2:-2]).bold = True
            else: par.add_run(re.sub(r"(?<![\w*])[_*]([^_*]+?)[_*](?![\w*])", r"\1", part))
    table: list[str] = []
    def flush():
        nonlocal table
        rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in table if not re.match(r"^\|?\s*-{2,}", r.strip())]
        if rows:
            t = d.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Table Grid"
            for i, r in enumerate(rows):
                for j, c in enumerate(r[: len(rows[0])]): t.cell(i, j).text = re.sub(r"\*\*|_", "", c)
        table = []
    for line in markdown.splitlines():
        if line.startswith("|"): table.append(line); continue
        if table: flush()
        if line.startswith("# "): d.add_heading(line[2:], 0)
        elif line.startswith("## "): d.add_heading(line[3:], 1)
        elif line.startswith("- "): runs(d.add_paragraph(style="List Bullet"), line[2:])
        elif line.strip(): runs(d.add_paragraph(), line)
    if table: flush()
    d.save(str(path)); return str(path)
