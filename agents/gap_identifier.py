"""Agent 5 — Research Gap Identification. Produces *candidate* gaps supported by retrieved literature only.
Signals: (1) recurring limitation themes across papers, (2) recurring un-themed limitations (embedding clusters),
(3) dataset concentration, (4) methodological imbalance, (5) sparse coverage of planned concepts.
It never claims novelty: every gap carries a coverage caveat and an evidence-strength class (Strong / Moderate / Weak)."""
from __future__ import annotations
import re
from collections import Counter, defaultdict
from .base import Agent, short_title
from .lexicon import THEMES, themes_of, family_of
from database import repository as repo
from retrieval.embeddings import get_embedder, tokenize, cosine
from retrieval.vector_store import get_store

CAVEAT = ("Candidate gap derived only from the {n} retrieved paper(s) ({ft} full text, {ab} abstract-only). Absence of work in this set does "
          "not prove the area is unexplored or novel; a systematic search is required before claiming novelty.")


def strength(n_papers: int) -> str:
    return "Strong" if n_papers >= 3 else "Moderate" if n_papers == 2 else "Weak"


class GapAgent(Agent):
    name = "Research Gap Identification"

    def _mk(self, gid, kind, title, desc, ev, rationale, direction, n_total, cov, extra=None):
        papers = list(dict.fromkeys(e["paper_id"] for e in ev))
        return {"id": gid, "kind": kind, "title": title, "description": desc,
                "supporting_papers": [{"paper_id": p, "title": cov["titles"].get(p, p)} for p in papers],
                "evidence": ev, "why_it_matters": rationale, "future_direction": direction,
                "confidence": strength(len(papers)), "evidence_strength": round(len(papers) / max(n_total, 1), 3),
                "caveat": CAVEAT.format(n=cov["n"], ft=cov["fulltext"], ab=cov["abstract_only"]), **(extra or {})}

    def run(self, state: dict) -> dict:
        analyses = state.get("analyses") or {}
        findings = state.get("findings") or []
        n = len(analyses)
        papers = {p["paper_id"]: p for p in repo.project_papers(state["project_id"], selected_only=True)}
        cov = {"n": n, "fulltext": sum(1 for a in analyses.values() if a.get("fulltext_status") == "fulltext"),
               "abstract_only": sum(1 for a in analyses.values() if a.get("fulltext_status") == "abstract_only"),
               "titles": {pid: short_title(a["title"]) for pid, a in analyses.items()}}
        cov["sources"] = sorted({s for p in papers.values() for s in p.get("sources", [])})
        if n < 2:
            return {"gaps": [], "coverage": cov, "warnings": ["Gap analysis needs at least two analysed papers; insufficient literature coverage."]}
        ev = lambda f: {"paper_id": f["paper_id"], "text": f["text"], "page": f["page"], "section": f["section"], "passage": f["passage"], "field": f["field"]}
        gaps: list[dict] = []
        # ---- 1. themed limitations ----
        by_theme: dict[str, dict[str, dict]] = defaultdict(dict)
        leftovers: list[dict] = []
        for f in findings:
            if f["field"] not in ("limitations", "future_work"):
                continue
            ts = themes_of(f["text"])
            if not ts and f["field"] == "limitations":
                leftovers.append(f)
            for t in ts:
                by_theme[t].setdefault(f["paper_id"], ev(f))
        for t, per_paper in sorted(by_theme.items(), key=lambda x: -len(x[1])):
            label, _, direction = THEMES[t]
            names = ", ".join(cov["titles"].get(p, p) for p in per_paper)
            ps = len(per_paper)
            desc = (f"{label} is reported as a limitation or future need in {ps} of {n} analysed papers ({names})." if ps > 1
                    else f"{label} is noted in 1 of {n} analysed papers ({names}); no corroboration within the retrieved set.")
            why = ("The same weakness is reported independently by multiple papers, which suggests a recurring obstacle to reliable use."
                   if ps > 1 else "Reported by a single paper; worth checking whether wider literature shares the concern.")
            gaps.append(self._mk(f"G{len(gaps)+1}", "recurring_limitation" if ps > 1 else "single_paper_observation", label, desc,
                                 list(per_paper.values()), why, direction, n, cov, {"theme": t}))
        # ---- 2. recurring un-themed limitations (embedding clusters) ----
        if len(leftovers) >= 2:
            emb = get_embedder(); thr = 0.55 if emb.name.startswith("st-") else 0.4
            vecs = emb.encode([f["text"] for f in leftovers]); clusters: list[list[int]] = []
            for i, v in enumerate(vecs):
                for c in clusters:
                    if cosine(vecs[c[0]], v) >= thr: c.append(i); break
                else: clusters.append([i])
            for c in clusters:
                pids = {leftovers[i]["paper_id"] for i in c}
                if len(pids) >= 2:
                    terms = [w for w, _ in Counter(w for i in c for w in tokenize(leftovers[i]["text"])).most_common(4)]
                    first_by_paper = {}
                    for i in c: first_by_paper.setdefault(leftovers[i]["paper_id"], ev(leftovers[i]))
                    title = "Recurring limitation: " + ", ".join(terms)
                    gaps.append(self._mk(f"G{len(gaps)+1}", "recurring_limitation", title,
                                         f"Similar limitation statements appear in {len(pids)} of {n} papers (shared terms: {', '.join(terms)}).",
                                         list(first_by_paper.values()), "Semantically similar limitations recur across papers.",
                                         "Investigate whether this shared limitation can be addressed systematically.", n, cov, {"theme": None}))
        # ---- 3. dataset concentration ----
        ds_papers: dict[str, list[str]] = defaultdict(list)
        for pid, a in analyses.items():
            for d in a["datasets"]: ds_papers[d].append(pid)
        if n >= 3 and ds_papers:
            top, ids = max(ds_papers.items(), key=lambda x: len(x[1]))
            if len(ids) >= 2 and len(ids) / n >= 0.5:
                evs = [ev(f) for f in findings if f["field"] == "dataset" and f["text"] == top][:6]
                gaps.append(self._mk(f"G{len(gaps)+1}", "dataset_concentration", f"Heavy reliance on the {top} dataset",
                                     f"{len(ids)} of {n} papers evaluate on the {top} dataset, which may limit how well findings generalise.",
                                     evs, "Conclusions that rest on a single dataset may not transfer to other populations or settings.",
                                     "Replicate key results on additional independent datasets.", n, cov, {"theme": "dataset_concentration"}))
        # ---- 4. methodological imbalance ----
        fam_papers: dict[str, set] = defaultdict(set)
        for pid, a in analyses.items():
            for al in a["algorithms"]: fam_papers[family_of(al)].add(pid)
        if n >= 4 and len(fam_papers) >= 2:
            dom, dp = max(fam_papers.items(), key=lambda x: len(x[1]))
            rare = [(f, p) for f, p in fam_papers.items() if len(p) == 1 and f != dom]
            if len(dp) / n >= 0.6 and rare:
                evs = [ev(f) for f in findings if f["field"] == "algorithm" and any(family_of(f["text"]) == r[0] and f["paper_id"] in r[1] for r in rare)][:4]
                gaps.append(self._mk(f"G{len(gaps)+1}", "method_imbalance", "Methodological families represented by a single paper",
                                     f"{dom} approaches appear in {len(dp)} of {n} papers, whereas {', '.join(r[0] for r in rare)} appear in only one paper each within this set.",
                                     evs or [ev(f) for f in findings if f["field"] == "algorithm"][:2],
                                     "Approaches studied by one paper cannot be compared or corroborated within this literature set.",
                                     "Benchmark the less-studied approaches against the dominant family on shared datasets.", n, cov, {"theme": "method_imbalance"}))
        # ---- 5. sparse concept coverage ----
        concepts = [c for c in ((state.get("plan") or {}).get("keywords") or []) if len(c) > 4 and " " in c][:6]
        texts = {pid: " ".join([papers.get(pid, {}).get("title", ""), papers.get(pid, {}).get("abstract", "")] +
                               [f["text"] for f in findings if f["paper_id"] == pid]).lower() for pid in analyses}
        for c in concepts:
            toks = set(tokenize(c))
            hit = [pid for pid, t in texts.items() if toks and toks <= set(tokenize(t))]
            if n >= 4 and len(hit) <= 1:
                evs = [{"paper_id": p, "text": f"Mentions '{c}'", "page": None, "section": "Title/Abstract/Findings", "passage": papers[p]["title"], "field": "coverage"} for p in hit] or \
                      [{"paper_id": next(iter(analyses)), "text": f"No analysed paper mentions all terms of '{c}'.", "page": None, "section": "Coverage check", "passage": "", "field": "coverage"}]
                g = self._mk(f"G{len(gaps)+1}", "sparse_coverage", f"Sparse coverage of '{c}' in retrieved literature",
                             f"Only {len(hit)} of {n} retrieved papers address '{c}' together with the rest of the query.",
                             evs, "May indicate an under-explored combination OR a limitation of the search; treat as a prompt for follow-up searching.",
                             f"Run targeted searches on '{c}' to determine whether this is a genuine gap.", n, cov, {"theme": "sparse_coverage"})
                g["confidence"] = "Weak"; gaps.append(g)
        # ---- optional LLM refinement of wording only (evidence untouched) ----
        warnings = []
        if self.use_llm and gaps:
            brief = [{"id": g["id"], "title": g["title"], "evidence": [e["passage"][:200] for e in g["evidence"][:4]]} for g in gaps[:8]]
            out = self.llm.complete_json("For each candidate research gap write 'future_direction' (1 sentence) supported ONLY by the evidence given. "
                                         'Return {"G1": "...", ...}.\n' + repr(brief), max_tokens=900)
            if isinstance(out, dict):
                for g in gaps:
                    v = out.get(g["id"])
                    if isinstance(v, str) and 10 < len(v) < 400: g["future_direction"] = v.strip()
            else:
                warnings.append(f"Gap wording refinement skipped ({self.llm.last_error}).")
        order = {"Strong": 0, "Moderate": 1, "Weak": 2}
        gaps.sort(key=lambda g: (order[g["confidence"]], -g["evidence_strength"]))
        for i, g in enumerate(gaps, 1): g["id"] = f"G{i}"
        if cov["abstract_only"]: warnings.append(f"{cov['abstract_only']} of {n} papers were analysed from abstracts only; gap evidence may be incomplete.")
        repo.replace_gaps(state["project_id"], gaps)
        return {"gaps": gaps, "coverage": cov, "warnings": warnings}
