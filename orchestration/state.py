"""Shared workflow state (agents communicate through this structured state, not free-form chat)."""
from __future__ import annotations
import operator
from typing import Annotated, Any, TypedDict


class WorkflowState(TypedDict, total=False):
    project_id: int
    question: str
    config: dict[str, Any]          # enable_comparison / enable_gap / enable_verification / max_papers / validate_dois ...
    plan: dict[str, Any]
    candidates: list[dict]          # discovered + ranked papers
    selected_ids: list[str]         # papers analysed
    preselected_ids: list[str]      # human-reviewed selection (from UI), optional
    uploaded_ids: list[str]         # user-uploaded documents to include
    extra_queries: list[str]        # added by the expansion loop
    analyses: dict[str, Any]        # paper_id -> structured analysis
    findings: list[dict]
    comparison: dict[str, Any]
    gaps: list[dict]
    verifications: list[dict]
    reference_checks: list[dict]
    coverage: dict[str, Any]
    expansion_loops: int
    supported_ratio: float
    evidence_status: str            # sufficient | insufficient | not_checked
    report_id: int
    status: str
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    usage: dict[str, Any]
