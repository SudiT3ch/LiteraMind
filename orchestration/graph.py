"""LangGraph orchestration of the six agents.
START -> planner -> discovery -> [expand?] -> retrieval -> analyst -> comparison -> gap -> verifier -> [expand? | report] -> END
Every node: DB-logged execution record, retries, graceful degradation, ablation flags, controlled (bounded) expansion loop."""
from __future__ import annotations
import time
from typing import Any
from langgraph.graph import StateGraph, START, END
from agents.planner import PlannerAgent
from agents.discovery import DiscoveryAgent
from agents.analyst import AnalystAgent
from agents.comparison import ComparisonAgent
from agents.gap_identifier import GapAgent
from agents.verifier import VerifierAgent
from config import get_settings
from database import repository as repo
from document_processing.ingest import fetch_and_index
from llm.provider import get_provider
from logging_utils import get_logger, redact
from orchestration.state import WorkflowState
from reports.generator import build_report

log = get_logger("graph")
ABLATION_FLAG = {"comparison": "enable_comparison", "gap": "enable_gap", "verifier": "enable_verification"}


def _summary(name: str, r: dict) -> tuple[str, Any]:
    if name == "planner": p = r.get("plan") or {}; return f"domain={p.get('domain')}; {len(p.get('search_queries', []))} queries; method={p.get('method')}", p
    if name == "discovery": return f"{len(r.get('candidates', []))} candidates, {len(r.get('selected_ids', []))} selected", {"selected": r.get("selected_ids"), "n_candidates": len(r.get("candidates", []))}
    if name == "retrieval": return r.get("_summary", ""), r.get("_out")
    if name == "analyst": a = r.get("analyses", {}); return f"{len(a)} papers analysed, {len(r.get('findings', []))} findings", {pid: {"findings": len(x["findings"]), "datasets": x["datasets"], "algorithms": x["algorithms"]} for pid, x in a.items()}
    if name == "comparison": c = r.get("comparison", {}); return f"{len(c.get('table', []))} papers compared, {len(c.get('comparable_metrics', []))} comparable metric groups", {k: c.get(k) for k in ("similarities", "differences", "metric_note")}
    if name == "gap": return f"{len(r.get('gaps', []))} candidate gaps", [{"id": g["id"], "title": g["title"], "confidence": g["confidence"]} for g in r.get("gaps", [])]
    if name == "verifier":
        from collections import Counter
        return f"{len(r.get('verifications', []))} claims; supported ratio {r.get('supported_ratio')}; evidence {r.get('evidence_status')}", dict(Counter(v["status"] for v in r.get("verifications", [])))
    return "", None


def make_node(name: str, label: str, fn):
    def node(state: WorkflowState) -> dict:
        cfg = state.get("config") or {}
        pid = state["project_id"]
        flag = ABLATION_FLAG.get(name)
        if flag and not cfg.get(flag, True):
            rid = repo.start_run(pid, label); repo.finish_run(rid, "skipped", "Disabled by configuration (ablation).")
            return {"warnings": [f"{label} skipped (disabled)."]}
        retries = get_settings().max_retries
        last = ""
        for attempt in range(1, retries + 2):
            rid = repo.start_run(pid, label, attempt)
            try:
                res = fn(state) or {}
                if res.get("status") == "failed":
                    repo.finish_run(rid, "failed", "", None, "; ".join(res.get("errors", [])) or "failed", attempts=attempt)
                    return res
                s, out = _summary(name, res)
                repo.finish_run(rid, "completed", s, out, None, res.get("warnings", []), attempt)
                return {k: v for k, v in res.items() if not k.startswith("_")}
            except Exception as e:  # noqa: BLE001 — a failing agent must not crash the workflow
                last = redact(f"{type(e).__name__}: {e}")
                log.exception("agent %s failed (attempt %d)", label, attempt)
                repo.finish_run(rid, "failed", "", None, last, attempts=attempt)
                if attempt <= retries: time.sleep(0.2)
        return {"errors": [f"{label} failed: {last}"], "warnings": [f"{label} failed after retries; downstream results may be incomplete."]}
    return node


def retrieval_fn(state: WorkflowState) -> dict:
    """Academic Paper Retrieval: open-access PDF download + indexing, abstract fallback."""
    out, warns = {}, []
    for pid in state.get("selected_ids", []):
        st = fetch_and_index(pid); out[pid] = st
        if st != "fulltext": warns.append(f"{pid}: full text unavailable; using {st.replace('_', ' ')}.")
    ft = sum(1 for v in out.values() if v == "fulltext")
    return {"warnings": warns, "_summary": f"{ft}/{len(out)} full text, {len(out) - ft} fallback", "_out": out}


def expand_fn(state: WorkflowState) -> dict:
    plan = state.get("plan") or {}; used = set(plan.get("search_queries", [])) | set(state.get("extra_queries", []))
    cands = [f"{k} {plan.get('keywords', [''])[0]}".strip() for k in (plan.get("concepts") or [])[:6]] + [f"{k} survey" for k in (plan.get("keywords") or [])[:2]]
    extra = [q for q in dict.fromkeys(cands) if q and q not in used][:4]
    cfg = dict(state.get("config") or {}); cfg["max_papers"] = int(cfg.get("max_papers") or get_settings().max_papers) + 3
    n = state.get("expansion_loops", 0) + 1
    return {"extra_queries": list(state.get("extra_queries", [])) + extra, "expansion_loops": n, "config": cfg,
            "warnings": [f"Evidence insufficient: expanding search (round {n}) with {len(extra)} additional queries."]}


def planner_fn(state: WorkflowState) -> dict:
    return {"plan": state["plan"]} if state.get("plan") else PlannerAgent().run(state)


def after_planner(state: WorkflowState) -> str:
    return "end" if state.get("status") == "failed" or not state.get("plan") else "discovery"


def after_discovery(state: WorkflowState) -> str:
    st = get_settings(); n = len(state.get("selected_ids") or [])
    if n >= int(((state.get("config") or {}).get("min_papers")) or st.min_papers): return "retrieval"
    if state.get("expansion_loops", 0) < st.max_expansion_loops and not (state.get("config") or {}).get("skip_search"): return "expand"
    return "retrieval" if n else "report"


def after_verifier(state: WorkflowState) -> str:
    st = get_settings()
    if state.get("evidence_status") == "insufficient" and state.get("expansion_loops", 0) < st.max_expansion_loops and not (state.get("config") or {}).get("skip_search"):
        return "expand"
    return "report"


def report_fn(state: WorkflowState) -> dict:
    cfg = dict(state.get("config") or {}); cfg["llm_used"] = get_provider().available
    r = build_report({**state, "config": cfg})
    return {"report_id": r["report_id"]}


def build_graph():
    g = StateGraph(WorkflowState)
    g.add_node("planner", make_node("planner", "Research Planner", planner_fn))
    g.add_node("discovery", make_node("discovery", "Paper Discovery", lambda s: DiscoveryAgent().run(s)))
    g.add_node("expand", make_node("expand", "Search Expansion", expand_fn))
    g.add_node("retrieval", make_node("retrieval", "Academic Paper Retrieval", retrieval_fn))
    g.add_node("analyst", make_node("analyst", "Research Analyst", lambda s: AnalystAgent().run(s)))
    g.add_node("comparison", make_node("comparison", "Comparison Agent", lambda s: ComparisonAgent().run(s)))
    g.add_node("gap", make_node("gap", "Research Gap Agent", lambda s: GapAgent().run(s)))
    g.add_node("verifier", make_node("verifier", "Evidence Verification Agent", lambda s: VerifierAgent().run(s)))
    g.add_node("report", make_node("report", "Report Generation", report_fn))
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", after_planner, {"discovery": "discovery", "end": END})
    g.add_conditional_edges("discovery", after_discovery, {"retrieval": "retrieval", "expand": "expand", "report": "report"})
    g.add_edge("expand", "discovery")
    for a, b in (("retrieval", "analyst"), ("analyst", "comparison"), ("comparison", "gap"), ("gap", "verifier")):
        g.add_edge(a, b)
    g.add_conditional_edges("verifier", after_verifier, {"expand": "expand", "report": "report"})
    g.add_edge("report", END)
    return g.compile()
