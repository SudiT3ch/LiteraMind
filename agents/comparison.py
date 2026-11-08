"""Agent 4 — Comparison: structured cross-paper comparison. Metrics are only called 'directly comparable' when metric name AND dataset match;
everything else is listed as non-comparable with the reason. Deterministic by design (LLM only adds an optional narrative)."""
from __future__ import annotations
from collections import Counter, defaultdict
from .base import Agent, short_title
from .lexicon import family_of
from database import repository as repo
from llm.provider import wrap_untrusted


def _first(findings: list[dict], pid: str, field: str) -> str:
    return next((f["text"] for f in findings if f["paper_id"] == pid and f["field"] == field), "")


def _all(findings: list[dict], pid: str, field: str, n: int = 3) -> list[str]:
    return [f["text"] for f in findings if f["paper_id"] == pid and f["field"] == field][:n]


class ComparisonAgent(Agent):
    name = "Comparison"

    def run(self, state: dict) -> dict:
        analyses = state.get("analyses") or {}
        findings = state.get("findings") or []
        if len(analyses) < 2:
            return {"comparison": {"table": [], "note": "At least two analysed papers are required for comparison."},
                    "warnings": ["Comparison skipped: fewer than two analysed papers."]}
        table = []
        for pid, a in analyses.items():
            table.append({"paper_id": pid, "title": a["title"], "year": a.get("year"), "coverage": a.get("fulltext_status"),
                          "objective": _first(findings, pid, "objectives") or _first(findings, pid, "problem"),
                          "methodology": _first(findings, pid, "methodology"), "algorithms": a["algorithms"], "datasets": a["datasets"],
                          "setup": _first(findings, pid, "setup"),
                          "metrics": [{"name": m["name"], "value": m["value"], "raw": m["raw"], "dataset": m.get("dataset"), "page": m["page"]} for m in a["metrics"]],
                          "results": _all(findings, pid, "results", 2), "limitations": _all(findings, pid, "limitations", 3)})
        # ---- metric comparability (never mix datasets/metrics) ----
        groups: dict[tuple, list] = defaultdict(list)
        for row in table:
            for m in row["metrics"]:
                groups[(m["name"], m["dataset"])].append({"paper_id": row["paper_id"], "title": short_title(row["title"]), "value": m["value"], "raw": m["raw"], "page": m["page"]})
        comparable, noncomparable = [], []
        for (metric, ds), items in groups.items():
            papers = {i["paper_id"] for i in items}
            if ds and len(papers) >= 2:
                vals = [i["value"] for i in items]
                comparable.append({"metric": metric, "dataset": ds, "entries": items, "spread": round(max(vals) - min(vals), 4),
                                   "best": max(items, key=lambda i: i["value"])["paper_id"] if metric not in ("RMSE", "MAE", "Perplexity") else min(items, key=lambda i: i["value"])["paper_id"]})
            else:
                reason = "dataset not identified" if not ds else "metric reported by only one paper on this dataset"
                noncomparable.append({"metric": metric, "dataset": ds, "entries": items, "reason": reason})
        metric_names = {m["name"] for r in table for m in r["metrics"]}
        cross_note = (f"Metrics reported across papers: {', '.join(sorted(metric_names))}. Values are only compared when metric and dataset are identical; "
                      "other values are not directly comparable (different datasets, splits or protocols)." if metric_names else "No numeric metrics were extracted.")
        # ---- similarities / differences / trends ----
        algo_count = Counter(a for r in table for a in r["algorithms"]); ds_count = Counter(d for r in table for d in r["datasets"])
        shared_algos = {a: c for a, c in algo_count.items() if c >= 2}; shared_ds = {d: c for d, c in ds_count.items() if c >= 2}
        families = defaultdict(list)
        for r in table:
            for fam in {family_of(a) for a in r["algorithms"]}:
                families[fam].append(r["paper_id"])
        n = len(table)
        similarities = [f"{a} is used in {c} of {n} papers." for a, c in sorted(shared_algos.items(), key=lambda x: -x[1])] + \
                       [f"The {d} dataset is used in {c} of {n} papers." for d, c in sorted(shared_ds.items(), key=lambda x: -x[1])]
        differences = [f"{short_title(r['title'], 50)} uniquely uses {', '.join(a for a in r['algorithms'] if algo_count[a] == 1)}." for r in table if any(algo_count[a] == 1 for a in r["algorithms"])]
        trends = [{"family": fam, "papers": ids, "years": sorted({analyses[i].get('year') for i in ids if analyses[i].get('year')})} for fam, ids in sorted(families.items(), key=lambda x: -len(x[1]))]
        # ---- contradictory / divergent findings (same metric+dataset, divergent values) ----
        contradictions = [{"metric": g["metric"], "dataset": g["dataset"], "spread": g["spread"], "entries": g["entries"],
                           "note": "Same metric and dataset but different reported values; may reflect protocol differences rather than a true contradiction."}
                          for g in comparable if g["spread"] >= 0.03]
        sw = [{"paper_id": r["paper_id"], "title": short_title(r["title"]), "strengths": r["results"][:1], "weaknesses": r["limitations"][:2]} for r in table]
        comp = {"table": table, "comparable_metrics": comparable, "non_comparable_metrics": noncomparable, "metric_note": cross_note,
                "similarities": similarities, "differences": differences, "methodological_trends": trends,
                "contradictory_findings": contradictions, "strengths_weaknesses": sw, "narrative": None}
        warnings = []
        if self.use_llm:
            facts = repr({k: comp[k] for k in ("similarities", "differences", "comparable_metrics", "strengths_weaknesses")})[:6000]
            txt = None
            try:
                txt = self.llm.complete("Write a 120-word neutral comparison paragraph using ONLY these extracted facts; do not compare "
                                        "metrics that are not listed as comparable.\n" + wrap_untrusted(facts), max_tokens=400)
            except Exception as e:  # LLMError
                warnings.append(f"Comparison narrative failed: {e}")
            comp["narrative"] = txt
        repo.save_comparison(state["project_id"], comp)
        return {"comparison": comp, "warnings": warnings}
