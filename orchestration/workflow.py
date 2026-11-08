"""High-level workflow API used by FastAPI/Streamlit/evaluation. Includes background execution with DB-backed progress."""
from __future__ import annotations
import threading
from typing import Any
from config import get_settings
from database import repository as repo
from llm.provider import get_provider
from logging_utils import get_logger, redact
from orchestration.graph import build_graph, make_node, planner_fn
from agents.discovery import DiscoveryAgent

log = get_logger("workflow")
DEFAULT_CONFIG = {"enable_comparison": True, "enable_gap": True, "enable_verification": True, "validate_dois": False}
_threads: dict[int, threading.Thread] = {}


def new_project(question: str, title: str | None = None) -> int:
    q = (question or "").strip()
    if len(q) < 5: raise ValueError("Research question must be at least 5 characters.")
    if len(q) > 1000: raise ValueError("Research question is too long (max 1000 characters).")
    return repo.create_project(title or q[:80], q)


def discover(project_id: int, config: dict | None = None, uploaded_ids: list[str] | None = None) -> dict:
    """Planner + Discovery only (human-review step: user then selects papers for analysis)."""
    q = repo.get_query(project_id)
    state: dict[str, Any] = {"project_id": project_id, "question": q.text, "config": {**DEFAULT_CONFIG, **(config or {})}, "uploaded_ids": uploaded_ids or []}
    state.update(make_node("planner", "Research Planner", planner_fn)(state))
    if state.get("status") == "failed": raise ValueError("; ".join(state.get("errors", [])) or "Planning failed")
    repo.save_plan(project_id, state["plan"])
    state.update(make_node("discovery", "Paper Discovery", lambda s: DiscoveryAgent().run(s))(state))
    repo.set_status(project_id, "discovered")
    return state


def run_analysis(project_id: int, selected_ids: list[str] | None = None, config: dict | None = None, uploaded_ids: list[str] | None = None) -> dict:
    """Run the full LangGraph workflow. If selected_ids is given they are used as the reviewed selection (no re-ranking override)."""
    q = repo.get_query(project_id)
    if q is None: raise ValueError("Unknown project")
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    state = {"project_id": project_id, "question": q.text, "config": cfg, "uploaded_ids": uploaded_ids or [], "expansion_loops": 0,
             "warnings": [], "errors": []}
    if q.plan: state["plan"] = q.plan
    if selected_ids:
        state["preselected_ids"] = selected_ids; cfg["skip_search"] = True
    repo.set_status(project_id, "running")
    try:
        final = build_graph().invoke(state, {"recursion_limit": 40})
    except Exception as e:  # noqa: BLE001
        repo.set_status(project_id, "failed"); log.exception("workflow crashed")
        raise RuntimeError(redact(str(e))) from e
    final["usage"] = get_provider().usage()
    n = len(final.get("analyses") or {})
    repo.set_status(project_id, "failed" if final.get("status") == "failed" else "insufficient_evidence" if (n == 0 or final.get("evidence_status") == "insufficient") else "completed")
    return final


def run_in_background(project_id: int, **kw) -> bool:
    if project_id in _threads and _threads[project_id].is_alive(): return False
    def target():
        try: run_analysis(project_id, **kw)
        except Exception as e: log.error("background run failed: %s", redact(str(e)))
    t = threading.Thread(target=target, daemon=True, name=f"run-{project_id}"); _threads[project_id] = t; t.start(); return True


def is_running(project_id: int) -> bool:
    return project_id in _threads and _threads[project_id].is_alive()
