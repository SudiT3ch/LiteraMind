"""Persistence helpers used by agents, workflow and API (keeps SQL out of agent code)."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from sqlalchemy import func
from .database import session_scope
from .models import (Project, ResearchQuery, Paper, ProjectPaper, Document, Chunk, AgentRun, Finding, Comparison, Gap,
                     Verification, Report)


def create_project(title: str, query: str) -> int:
    with session_scope() as s:
        p = Project(title=title[:300] or "Untitled", status="created")
        s.add(p); s.flush()
        s.add(ResearchQuery(project_id=p.id, text=query))
        return p.id


def get_query(project_id: int) -> ResearchQuery | None:
    with session_scope() as s:
        return s.query(ResearchQuery).filter_by(project_id=project_id).order_by(ResearchQuery.id.desc()).first()


def save_plan(project_id: int, plan: dict) -> None:
    with session_scope() as s:
        q = s.query(ResearchQuery).filter_by(project_id=project_id).order_by(ResearchQuery.id.desc()).first()
        if q: q.plan = plan


def set_status(project_id: int, status: str) -> None:
    with session_scope() as s:
        p = s.get(Project, project_id)
        if p: p.status = status


def upsert_paper(rec: dict) -> str:
    """rec: PaperRecord.to_dict(). Merges missing fields; never overwrites with empty values."""
    pid = rec["paper_id"]
    with session_scope() as s:
        p = s.get(Paper, pid)
        if p is None:
            p = Paper(paper_id=pid, title=rec["title"])
            s.add(p)
        for f in ("year", "doi", "arxiv_id", "s2_id", "openalex_id", "venue", "url", "pdf_url", "citation_count"):
            if rec.get(f) not in (None, "") and getattr(p, f) in (None, ""):
                setattr(p, f, rec[f])
        if len(rec.get("abstract") or "") > len(p.abstract or ""): p.abstract = rec["abstract"]
        if len(rec.get("authors") or []) > len(p.authors or []): p.authors = rec["authors"]
        p.sources = sorted(set(p.sources or []) | set(rec.get("sources") or []))
    return pid


def link_paper(project_id: int, paper_id: str, relevance: float, selected: bool = True) -> None:
    with session_scope() as s:
        l = s.query(ProjectPaper).filter_by(project_id=project_id, paper_id=paper_id).first()
        if l is None:
            s.add(ProjectPaper(project_id=project_id, paper_id=paper_id, relevance=relevance, selected=selected))
        else:
            l.relevance = max(l.relevance, relevance); l.selected = selected or l.selected


def set_selection(project_id: int, selected_ids: list[str]) -> None:
    with session_scope() as s:
        for l in s.query(ProjectPaper).filter_by(project_id=project_id).all():
            l.selected = l.paper_id in selected_ids


def paper_to_dict(p: Paper) -> dict:
    return {c: getattr(p, c) for c in ("paper_id", "title", "authors", "year", "abstract", "doi", "arxiv_id", "venue", "url",
                                       "pdf_url", "sources", "citation_count", "is_uploaded", "fulltext_status")}


def project_papers(project_id: int, selected_only: bool = False) -> list[dict]:
    with session_scope() as s:
        q = s.query(Paper, ProjectPaper).join(ProjectPaper, ProjectPaper.paper_id == Paper.paper_id).filter(ProjectPaper.project_id == project_id)
        if selected_only:
            q = q.filter(ProjectPaper.selected.is_(True))
        rows = q.order_by(ProjectPaper.relevance.desc()).all()
        return [{**paper_to_dict(p), "relevance": l.relevance, "selected": l.selected} for p, l in rows]


def get_paper(paper_id: str) -> dict | None:
    with session_scope() as s:
        p = s.get(Paper, paper_id)
        return paper_to_dict(p) if p else None


def start_run(project_id: int, agent: str, attempts: int = 1) -> int:
    with session_scope() as s:
        r = AgentRun(project_id=project_id, agent=agent, status="running", attempts=attempts)
        s.add(r); s.flush()
        return r.id


def finish_run(run_id: int, status: str, summary: str = "", output: Any = None, error: str | None = None,
               warnings: list[str] | None = None, attempts: int = 1) -> None:
    with session_scope() as s:
        r = s.get(AgentRun, run_id)
        if not r: return
        r.status, r.summary, r.output, r.error = status, summary[:1000], output, (error or None)
        r.warnings, r.attempts = warnings or [], attempts
        r.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        r.duration_s = round((r.finished_at - r.started_at).total_seconds(), 3)


def list_runs(project_id: int) -> list[dict]:
    with session_scope() as s:
        rows = s.query(AgentRun).filter_by(project_id=project_id).order_by(AgentRun.id).all()
        return [{"id": r.id, "agent": r.agent, "status": r.status, "attempts": r.attempts, "duration_s": r.duration_s,
                 "summary": r.summary, "error": r.error, "warnings": r.warnings or [], "output": r.output,
                 "started_at": r.started_at.isoformat() if r.started_at else None} for r in rows]


def replace_findings(project_id: int, findings: list[dict]) -> None:
    with session_scope() as s:
        s.query(Finding).filter_by(project_id=project_id).delete()
        for f in findings:
            s.add(Finding(project_id=project_id, paper_id=f["paper_id"], field=f["field"], text=f["text"], page=f.get("page"),
                          section=f.get("section"), passage=f.get("passage", ""), method=f.get("method", "heuristic")))


def list_findings(project_id: int) -> list[dict]:
    with session_scope() as s:
        return [{"paper_id": f.paper_id, "field": f.field, "text": f.text, "page": f.page, "section": f.section,
                 "passage": f.passage, "method": f.method} for f in s.query(Finding).filter_by(project_id=project_id).order_by(Finding.id)]


def save_comparison(project_id: int, payload: dict) -> None:
    with session_scope() as s:
        s.query(Comparison).filter_by(project_id=project_id).delete()
        s.add(Comparison(project_id=project_id, payload=payload))


def get_comparison(project_id: int) -> dict | None:
    with session_scope() as s:
        c = s.query(Comparison).filter_by(project_id=project_id).order_by(Comparison.id.desc()).first()
        return c.payload if c else None


def replace_gaps(project_id: int, gaps: list[dict]) -> None:
    with session_scope() as s:
        s.query(Gap).filter_by(project_id=project_id).delete()
        for g in gaps:
            s.add(Gap(project_id=project_id, title=g["title"], payload=g, confidence=g.get("confidence", "Weak")))


def list_gaps(project_id: int) -> list[dict]:
    with session_scope() as s:
        return [g.payload for g in s.query(Gap).filter_by(project_id=project_id).order_by(Gap.id)]


def replace_verifications(project_id: int, items: list[dict]) -> None:
    with session_scope() as s:
        s.query(Verification).filter_by(project_id=project_id).delete()
        for v in items:
            s.add(Verification(project_id=project_id, claim=v["claim"], claim_source=v.get("claim_source", ""), paper_id=v.get("paper_id"),
                               passage=v.get("passage", ""), page=v.get("page"), section=v.get("section"), status=v["status"],
                               score=v.get("score", 0.0), contradicting=v.get("contradicting", []), method=v.get("method", "heuristic")))


def list_verifications(project_id: int) -> list[dict]:
    with session_scope() as s:
        return [{"claim": v.claim, "claim_source": v.claim_source, "paper_id": v.paper_id, "passage": v.passage, "page": v.page,
                 "section": v.section, "status": v.status, "score": v.score, "contradicting": v.contradicting or [],
                 "method": v.method} for v in s.query(Verification).filter_by(project_id=project_id).order_by(Verification.id)]


def save_report(project_id: int, title: str, markdown: str, payload: dict) -> int:
    with session_scope() as s:
        r = Report(project_id=project_id, title=title, markdown=markdown, payload=payload)
        s.add(r); s.flush(); return r.id


def latest_report(project_id: int) -> dict | None:
    with session_scope() as s:
        r = s.query(Report).filter_by(project_id=project_id).order_by(Report.id.desc()).first()
        return {"id": r.id, "title": r.title, "markdown": r.markdown, "payload": r.payload, "created_at": r.created_at.isoformat()} if r else None


def list_projects() -> list[dict]:
    with session_scope() as s:
        out = []
        for p in s.query(Project).order_by(Project.id.desc()).all():
            q = s.query(ResearchQuery).filter_by(project_id=p.id).first()
            n = s.query(func.count(ProjectPaper.id)).filter_by(project_id=p.id, selected=True).scalar()
            out.append({"id": p.id, "title": p.title, "status": p.status, "created_at": p.created_at.isoformat(),
                        "query": q.text if q else "", "n_papers": n})
        return out


def list_documents() -> list[dict]:
    with session_scope() as s:
        out = []
        for d, p in s.query(Document, Paper).join(Paper, Paper.paper_id == Document.paper_id).order_by(Document.id.desc()).all():
            n = s.query(func.count(Chunk.id)).filter_by(paper_id=p.paper_id).scalar()
            out.append({"document_id": d.id, "paper_id": p.paper_id, "filename": d.filename, "title": p.title, "pages": d.n_pages,
                        "size_bytes": d.size_bytes, "status": d.status, "error": d.error, "chunks": n,
                        "created_at": d.created_at.isoformat()})
        return out


def document_text(paper_id: str, limit: int = 20000) -> str:
    with session_scope() as s:
        rows = s.query(Chunk).filter_by(paper_id=paper_id).order_by(Chunk.idx).all()
        return "\n\n".join(f"[p.{c.page} · {c.section}] {c.text}" for c in rows)[:limit]


def stats() -> dict:
    with session_scope() as s:
        return {"projects": s.query(func.count(Project.id)).scalar(),
                "completed": s.query(func.count(Project.id)).filter(Project.status == "completed").scalar(),
                "papers": s.query(func.count(func.distinct(ProjectPaper.paper_id))).scalar(),
                "documents": s.query(func.count(Document.id)).scalar(),
                "claims_verified": s.query(func.count(Verification.id)).scalar()}
