"""Agent 3 — Research Analyst: structured extraction (problem, objectives, methods, datasets, metrics, results, limitations, future work).
Every finding keeps paper_id + page + section + source passage. Heuristic path = cue-phrase/section scoring over retrieved chunks;
LLM path = one JSON call per paper whose quotes are verified to occur verbatim in the retrieved text (hallucinated evidence is dropped)."""
from __future__ import annotations
import re
from .base import Agent, quote_in
from .lexicon import find_algorithms, find_datasets, find_metrics
from database import repository as repo
from document_processing.chunker import split_sentences
from llm.provider import wrap_untrusted
from retrieval.retriever import retrieve
from retrieval.vector_store import get_store

FIELDS = ["problem", "objectives", "methodology", "setup", "results", "limitations", "future_work"]
CUES = {
    "problem": (r"\b(problem|challenge|opaque|lack|requires?|restrict|essential|need)", {"abstract", "introduction", "front matter", "background"}),
    "objectives": (r"\b(we (propose|present|introduce|study|apply|investigate)|this (work|study|paper|article)|aim(s)? to|objective|goal)", {"abstract", "introduction"}),
    "methodology": (r"\b(we (train|use|employ|apply|implement|develop|compute|pair|simulate)|is paired|trained|architecture|algorithm|approach|method)", {"methodology", "methods", "method", "approach", "model", "materials and methods"}),
    "setup": (r"\b(dataset|split|partition|simulat|sites?|epochs?|cross-validation|learning rate|patients|images|records|hospitals|clinics|experiments? use)", {"methodology", "methods", "method", "experiments", "experimental setup", "dataset", "datasets", "data"}),
    "results": (r"\b(achiev|reach|attain|outperform|improv|drops?|identified|preferred|rated|valid in|compared with|accuracy|auc|bleu|f1)", {"results", "results and discussion", "evaluation", "experimental results", "conclusion", "conclusions", "discussion"}),
    "limitations": (r"\b(limitation|lack|not (been )?(evaluated|examined|conducted|performed|analy[sz]ed|reflect)|small|may (not|suggest)|degrad|drops?|high|only|single|no user|relies)", {"limitations"}),
    "future_work": (r"\b(future work|we plan|should|needed|further|future research|will)", {"future work", "conclusion", "conclusions", "limitations"}),
}
CAP = {"limitations": 5, "future_work": 3, "results": 3, "methodology": 2, "setup": 2, "problem": 2, "objectives": 2}
FIELD_QUERIES = {"problem": "research problem challenge motivation", "objectives": "we propose aim objective contribution",
                 "methodology": "method approach algorithm model architecture", "setup": "dataset experimental setup evaluation",
                 "results": "results accuracy performance findings", "limitations": "limitations weaknesses lack not evaluated",
                 "future_work": "future work further research needed"}


def paper_sentences(paper_id: str) -> list[dict]:
    """De-duplicated sentences of a paper with page/section (ordered by chunk index)."""
    hits = get_store().get_all([paper_id])
    key = lambda h: int(h.chunk_uid.rsplit("::c", 1)[-1]) if "::c" in h.chunk_uid else 0
    out, seen = [], set()
    for h in sorted(hits, key=key):
        for s in split_sentences(h.text):
            k = re.sub(r"\W+", " ", s.lower()).strip()
            if len(s.split()) >= 4 and k not in seen:
                seen.add(k); out.append({"text": s, "page": h.page, "section": h.section, "chunk_uid": h.chunk_uid})
    return out


def score(field: str, s: dict) -> float:
    cue, secs = CUES[field]
    sc = len(re.findall(cue, s["text"], re.I))
    in_sec = (s["section"] or "").lower() in secs
    if in_sec: sc += 1.5
    if field == "results" and re.search(r"\d", s["text"]): sc += 1
    if field == "limitations" and (s["section"] or "").lower() != "limitations" and sc < 2: return 0
    return sc if (in_sec or sc >= 2) else 0


def finding(pid: str, field: str, text: str, s: dict, method: str = "heuristic") -> dict:
    return {"paper_id": pid, "field": field, "text": text, "page": s.get("page") or None, "section": s.get("section"),
            "passage": s["text"], "method": method}


class AnalystAgent(Agent):
    name = "Research Analyst"

    def analyse_heuristic(self, pid: str) -> dict:
        sents = paper_sentences(pid)
        findings: list[dict] = []
        for f in FIELDS:
            if f in ("limitations", "future_work"):
                sec = "limitations" if f == "limitations" else "future work"
                chosen = [s for s in sents if (s["section"] or "").lower() == sec][: CAP[f]]
                if f == "future_work" and not chosen:
                    chosen = [s for s in sents if re.search(r"future work|future research|we plan|further (work|research)", s["text"], re.I)][: CAP[f]]
                if f == "limitations" and not chosen:
                    chosen = [s for sc, s in sorted(((score(f, s), s) for s in sents), key=lambda x: -x[0]) if sc >= 2][: CAP[f]]
            else:
                ranked = sorted(((score(f, s), i, s) for i, s in enumerate(sents)), key=lambda x: (-x[0], x[1]))
                chosen = [s for sc, _, s in ranked if sc > 0][: CAP[f]]
            findings += [finding(pid, f, s["text"], s) for s in chosen]
        return self._entities(pid, sents, findings)

    def _entities(self, pid: str, sents: list[dict], findings: list[dict]) -> dict:
        algos: dict[str, dict] = {}; datasets: dict[str, dict] = {}; metrics: list[dict] = []
        for s in sents:
            if (s["section"] or "").lower() in ("references", "acknowledgments"):
                continue
            for a in find_algorithms(s["text"]):
                algos.setdefault(a, s)
            for d in find_datasets(s["text"]):
                datasets.setdefault(d, s)
            for m in find_metrics(s["text"]):
                ds = next((d for d in find_datasets(s["text"])), None)
                metrics.append({**m, "dataset": ds, "page": s["page"], "section": s["section"], "passage": s["text"]})
        for a, s in algos.items(): findings.append(finding(pid, "algorithm", a, s))
        for d, s in datasets.items(): findings.append(finding(pid, "dataset", d, s))
        only_ds = list(datasets)[0] if len(datasets) == 1 else None
        for m in metrics:
            if m["dataset"] is None and only_ds: m["dataset"] = only_ds; m["dataset_inferred"] = True
            findings.append({"paper_id": pid, "field": "metric", "text": f'{m["name"]} = {m["raw"]}' + (f' on {m["dataset"]}' if m["dataset"] else ""),
                             "page": m["page"] or None, "section": m["section"], "passage": m["passage"], "method": "heuristic"})
        return {"paper_id": pid, "findings": findings, "algorithms": list(algos), "datasets": list(datasets), "metrics": metrics}

    def analyse_llm(self, pid: str, title: str, warnings: list[str]) -> dict | None:
        chunks, seen = [], set()
        for f, q in FIELD_QUERIES.items():
            for r in retrieve(q, 3, paper_ids=[pid]):
                if r.chunk_uid not in seen:
                    seen.add(r.chunk_uid); chunks.append(r)
        chunks = chunks[:10]
        if not chunks:
            return None
        ctx = "\n".join(f"[chunk {i}] (page {c.page}, section {c.section}) {c.text}" for i, c in enumerate(chunks))
        prompt = (f"Extract structured information from the paper excerpts. Paper title: {title}\n"
                  f"JSON object with keys {FIELDS}; each value is a list (max 3) of objects "
                  '{"text": concise finding, "quote": EXACT sentence copied from the excerpt, "chunk": chunk index}. '
                  "Also keys algorithms (list of strings), datasets (list of strings). Omit anything not stated.\n" + wrap_untrusted(ctx))
        out = self.llm.complete_json(prompt, max_tokens=1800)
        if not isinstance(out, dict):
            warnings.append(f"{pid}: LLM extraction failed ({self.llm.last_error}); used heuristics.")
            return None
        findings, dropped = [], 0
        for f in FIELDS:
            for item in (out.get(f) or [])[:3]:
                try:
                    c = chunks[int(item.get("chunk"))]
                except (TypeError, ValueError, IndexError):
                    c = None
                quote = str(item.get("quote", ""))
                if c is None or not quote_in(quote, c.text):  # wrong/missing chunk index: search all retrieved chunks
                    c = next((x for x in chunks if quote_in(quote, x.text)), None)
                if c is None:  # quote not present anywhere in the evidence -> hallucinated, drop
                    dropped += 1; continue
                findings.append({"paper_id": pid, "field": f, "text": str(item.get("text") or quote)[:500], "page": c.page, "section": c.section,
                                 "passage": quote, "method": "llm"})
        if dropped:
            warnings.append(f"{pid}: dropped {dropped} LLM finding(s) whose quoted evidence was not found in the source text.")
        if len(findings) < 3:
            return None
        base = self._entities(pid, paper_sentences(pid), [])  # entities still rule-based for traceable values
        return {**base, "findings": findings + base["findings"]}

    def run(self, state: dict) -> dict:
        pid_project = state["project_id"]
        papers = {p["paper_id"]: p for p in repo.project_papers(pid_project, selected_only=True)}
        warnings: list[str] = []
        analyses: dict = {}; all_findings: list[dict] = []
        for pid, p in papers.items():
            if not get_store().has_paper(pid):
                warnings.append(f"{pid}: no indexed text available; skipped."); continue
            a = self.analyse_llm(pid, p["title"], warnings) if self.use_llm else None
            a = a or self.analyse_heuristic(pid)
            a["title"] = p["title"]; a["year"] = p["year"]; a["fulltext_status"] = p["fulltext_status"]
            analyses[pid] = a; all_findings += a["findings"]
            if p["fulltext_status"] == "abstract_only":
                warnings.append(f"{pid}: analysed from abstract only (full text unavailable); limitations/methods may be incomplete.")
            if not a["findings"]:
                warnings.append(f"{pid}: no structured findings could be extracted.")
        repo.replace_findings(pid_project, all_findings)
        return {"analyses": analyses, "findings": all_findings, "warnings": warnings}
