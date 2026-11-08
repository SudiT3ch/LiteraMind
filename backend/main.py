"""FastAPI backend. Run: uvicorn backend.main:app --reload"""
from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from database.database import init_db, is_initialised
from document_processing.pdf_extractor import PDFError
from logging_utils import get_logger, redact
from backend.routes import projects, documents, system

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not is_initialised():
        init_db()
    yield


app = FastAPI(title="LiteraMind API", version="1.0.0", lifespan=lifespan,
              description="Evidence-grounded multi-agent literature analysis. Interactive docs at /docs.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(ValueError)
async def _value_error(_: Request, e: ValueError):
    return JSONResponse(status_code=400, content={"detail": redact(str(e))})


@app.exception_handler(PDFError)
async def _pdf_error(_: Request, e: PDFError):
    return JSONResponse(status_code=422, content={"detail": redact(str(e))})


@app.exception_handler(Exception)
async def _unhandled(_: Request, e: Exception):
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error. See server logs."})


for r in (system.router, projects.router, documents.router):
    app.include_router(r, prefix="/api")
