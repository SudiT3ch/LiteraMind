"""Multi-source search: queries every enabled source, tolerates per-source failures, de-duplicates."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from config import get_settings
from logging_utils import get_logger
from . import arxiv_client, semantic_scholar, openalex
from .base import PaperRecord, deduplicate, SourceError

log = get_logger("aggregator")
REGISTRY = {"arxiv": arxiv_client.search, "semantic_scholar": semantic_scholar.search, "openalex": openalex.search}


def search_all(queries: list[str], per_source: int = 10, sources: list[str] | None = None) -> tuple[list[PaperRecord], list[str]]:
    """Returns (deduplicated records, warnings). Never raises for single-source failure."""
    sources = sources or get_settings().sources()
    warnings: list[str] = []
    jobs = [(s, q) for s in sources if s in REGISTRY for q in queries if q.strip()]
    if not jobs:
        return [], ["No enabled academic sources or empty queries."]

    def run(job):
        s, q = job
        try:
            return s, q, REGISTRY[s](q, per_source), None
        except SourceError as e:
            return s, q, [], str(e)
        except Exception as e:  # defensive: parser bugs must not crash the workflow
            return s, q, [], f"{type(e).__name__}: {e}"

    records: list[PaperRecord] = []
    failed: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for s, q, recs, err in ex.map(run, jobs):
            records.extend(recs)
            if err:
                failed[s] = err
    for s, err in failed.items():
        warnings.append(f"Source '{s}' failed: {err}")
        log.warning("source %s failed: %s", s, err)
    return deduplicate(records), warnings
