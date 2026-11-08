"""Agent 1 — Research Planner: question -> domain, keywords, related terms, search queries, information needs."""
from __future__ import annotations
import re
from .base import Agent
from .lexicon import DOMAINS, SYNONYMS
from retrieval.embeddings import STOP

QSTOP = STOP | set("how what why which who when where does do did can could should would will improve improves improving impact effect "
                   "effects role using use approach approaches methods method study studies review literature research about between "
                   "versus vs over under across among through towards toward current recent state art".split())
INFO_NEEDS = ["research problem and objectives", "methodology, algorithms and models", "datasets and experimental setup",
              "evaluation metrics and results", "stated limitations", "future research directions"]


def _phrases(question: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", question)
    phrases, cur = [], []
    for w in words:
        if w.lower() in QSTOP:
            if cur: phrases.append(" ".join(cur)); cur = []
        else:
            cur.append(w)
    if cur: phrases.append(" ".join(cur))
    out, seen = [], set()
    for p in phrases:
        k = p.lower()
        if k not in seen and len(k) > 2:
            seen.add(k); out.append(p)
    return out


class PlannerAgent(Agent):
    name = "Research Planner"

    def heuristic(self, question: str) -> dict:
        phrases = _phrases(question)
        low = question.lower()
        scores = {d: sum(1 for k in kws if k in low) for d, kws in DOMAINS.items()}
        domains = [d for d, s in sorted(scores.items(), key=lambda x: -x[1]) if s > 0][:3]
        related: list[str] = []
        for key, syns in SYNONYMS.items():
            if re.search(rf"\b{re.escape(key)}\b", low):
                related += [s for s in syns if s.lower() not in low]
        related = list(dict.fromkeys(related))[:8]
        keywords = list(dict.fromkeys([p for p in phrases] + [w for p in phrases for w in p.split() if len(w) > 3 and w.lower() not in QSTOP]))[:10]
        base = " ".join(phrases) or question
        queries = [question.strip().rstrip("?")]
        if base.lower() != queries[0].lower(): queries.append(base)
        for r in related[:3]:
            queries.append(f"{phrases[0] if phrases else base} {r}")
        if len(phrases) >= 2: queries.append(" ".join(phrases[:2]) + " literature review")
        queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))[:6]
        return {"question": question, "domain": domains[0] if domains else "General Research", "domains": domains,
                "keywords": keywords, "concepts": list(dict.fromkeys(phrases + related))[:12], "search_queries": queries,
                "information_needs": INFO_NEEDS, "method": "heuristic"}

    def llm_plan(self, question: str) -> dict | None:
        prompt = ("Create a literature research plan for the question below. JSON keys: domain (string), keywords (5-10 strings), "
                  "concepts (related concepts, 5-10), search_queries (3-6 concise academic search strings), information_needs (list).\n"
                  f"Question: {question}")
        out = self.llm.complete_json(prompt, max_tokens=700)
        if not isinstance(out, dict) or not out.get("search_queries"):
            return None
        h = self.heuristic(question)
        clean = lambda v: [str(x).strip() for x in (v or []) if str(x).strip()][:12]
        return {"question": question, "domain": str(out.get("domain") or h["domain"]), "domains": [str(out.get("domain") or h["domain"])],
                "keywords": clean(out.get("keywords")) or h["keywords"], "concepts": clean(out.get("concepts")) or h["concepts"],
                "search_queries": clean(out.get("search_queries"))[:6], "information_needs": clean(out.get("information_needs")) or INFO_NEEDS,
                "method": "llm"}

    def run(self, state: dict) -> dict:
        q = (state.get("question") or "").strip()
        warnings = []
        if len(re.findall(r"[A-Za-z]{3,}", q)) < 2:
            return {"plan": {}, "errors": ["Research question is too short or unsupported; provide a descriptive research question."],
                    "status": "failed"}
        plan = None
        if self.use_llm:
            plan = self.llm_plan(q)
            if plan is None:
                warnings.append(f"Planner LLM failed ({self.llm.last_error}); used rule-based planning.")
        plan = plan or self.heuristic(q)
        return {"plan": plan, "warnings": warnings}
