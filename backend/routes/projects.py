from io import BytesIO
from pathlib import Path
import tempfile
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from database import repository as repo
from orchestration import workflow
from reports import generator
from visualization import knowledge_graph as kg

router = APIRouter(prefix="/projects", tags=["projects"])


class NewProject(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    title: str | None = Field(None, max_length=300)


class RunRequest(BaseModel):
    paper_ids: list[str] | None = None
    uploaded_ids: list[str] | None = None
    config: dict | None = None
    background: bool = True


def _need(pid: int):
    if repo.get_query(pid) is None:
        raise HTTPException(404, "Unknown project")


@router.post("")
def create(body: NewProject):
    return {"project_id": workflow.new_project(body.question, body.title)}


@router.get("")
def list_projects():
    return repo.list_projects()


@router.get("/{pid}")
def get(pid: int):
    _need(pid)
    q = repo.get_query(pid)
    proj = next(p for p in repo.list_projects() if p["id"] == pid)
    return {**proj, "plan": q.plan, "papers": repo.project_papers(pid), "runs": repo.list_runs(pid), "running": workflow.is_running(pid)}


@router.post("/{pid}/discover")
def discover(pid: int, config: dict | None = None):
    _need(pid)
    st = workflow.discover(pid, config)
    return {"plan": st["plan"], "candidates": st["candidates"], "selected_ids": st["selected_ids"], "warnings": st.get("warnings", [])}


class Selection(BaseModel):
    paper_ids: list[str]


@router.post("/{pid}/select")
def select(pid: int, body: Selection):
    _need(pid)
    repo.set_selection(pid, body.paper_ids)
    return {"selected": body.paper_ids}


@router.post("/{pid}/run")
def run(pid: int, body: RunRequest):
    _need(pid)
    sel = body.paper_ids if body.paper_ids is not None else [p["paper_id"] for p in repo.project_papers(pid, selected_only=True) if not p["is_uploaded"]] or None
    kw = dict(selected_ids=sel, config=body.config, uploaded_ids=body.uploaded_ids or [p["paper_id"] for p in repo.project_papers(pid) if p["is_uploaded"]])
    if body.background:
        started = workflow.run_in_background(pid, **kw)
        return {"started": started, "message": "Run started" if started else "A run is already in progress"}
    final = workflow.run_analysis(pid, **kw)
    return {"status": final.get("status", "completed"), "warnings": final.get("warnings", []), "errors": final.get("errors", []), "usage": final.get("usage")}


@router.get("/{pid}/runs")
def runs(pid: int):
    _need(pid)
    return {"running": workflow.is_running(pid), "runs": repo.list_runs(pid), "status": next(p["status"] for p in repo.list_projects() if p["id"] == pid)}


@router.get("/{pid}/findings")
def findings(pid: int):
    _need(pid); return repo.list_findings(pid)


@router.get("/{pid}/comparison")
def comparison(pid: int):
    _need(pid); return repo.get_comparison(pid) or {}


@router.get("/{pid}/gaps")
def gaps(pid: int):
    _need(pid); return repo.list_gaps(pid)


@router.get("/{pid}/verifications")
def verifications(pid: int):
    _need(pid); return repo.list_verifications(pid)


@router.get("/{pid}/report")
def report(pid: int):
    _need(pid)
    r = repo.latest_report(pid)
    if not r: raise HTTPException(404, "No report generated yet")
    return r


@router.get("/{pid}/report/download")
def download(pid: int, format: str = "md"):
    _need(pid)
    r = repo.latest_report(pid)
    if not r: raise HTTPException(404, "No report generated yet")
    fmt = format.lower()
    if fmt == "md":
        return Response(r["markdown"], media_type="text/markdown", headers={"Content-Disposition": f'attachment; filename="literamind_report_{pid}.md"'})
    if fmt not in ("pdf", "docx"):
        raise HTTPException(400, "format must be md, pdf or docx")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / f"report.{fmt}"
        (generator.to_pdf if fmt == "pdf" else generator.to_docx)(r["markdown"], path)
        mt = "application/pdf" if fmt == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        return Response(path.read_bytes(), media_type=mt, headers={"Content-Disposition": f'attachment; filename="literamind_report_{pid}.{fmt}"'})


@router.get("/{pid}/graph")
def graph(pid: int):
    _need(pid); return kg.to_json(kg.build_graph(pid))
