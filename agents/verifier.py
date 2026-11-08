"""Agent 6 — Evidence Verification. Independently re-retrieves passages from the cited paper for every claim and judges support.
Statuses: Supported | Partially supported | Contradicted | Insufficient evidence.
Heuristic judge = stemmed content-word coverage + numeric consistency + negation consistency (+ embedding similarity for ranking).
This is a lexical entailment approximation (documented limitation); an LLM judge, when configured, may revise non-Supported verdicts but
must quote a passage that really exists in the evidence."""
from __future__ import annotations
import re
from .base import Agent, short_title, quote_in
from .lexicon import THEMES, family_of, find_metrics
from database import repository as repo
from document_processing.chunker import split_sentences
from retrieval.embeddings import STOP, stem, get_embedder, cosine
from retrieval.retriever import retrieve
from llm.provider import wrap_untrusted

NEG = {"no", "not", "without", "lack", "lacks", "lacking", "lacked", "never", "none", "absent", "missing", "neither", "nor", "cannot", "unable"}
THEME_CLAIM = {"external_validation": "lacks external validation", "small_data": "small dataset", "user_study": "no user study",
               "faithfulness": "explanation faithfulness not evaluated", "privacy_utility": "accuracy degrades privacy",
               "cost": "high communication cost", "heterogeneity": "heterogeneous data generalisation",
               "plausibility": "clinically unrealistic explanations", "bias": "bias fairness not assessed"}
STATUSES = ["Supported", "Partially supported", "Contradicted", "Insufficient evidence"]


def ctokens(text: str) -> tuple[set[str], bool]:
    """Content tokens (stemmed, stopwords removed) and whether the text contains negation."""
    toks, neg = [], False
    for w in re.findall(r"[a-z0-9]+(?:[\-\.][a-z0-9]+)*", (text or "").lower()):
        if w in NEG: neg = True; continue
        if w in STOP or len(w) < 2: continue
        toks.append(stem(w))
    return set(toks), neg


def numbers(text: str) -> set[str]:
    out = set()
    for m in re.findall(r"\d+(?:\.\d+)?", text or ""):
        try: out.add(str(float(m)).rstrip("0").rstrip(".") if "." in m else str(int(m)))
        except ValueError: pass
    return out


def judge(claim: str, sentence: str) -> dict:
    ct, cneg = ctokens(claim); st, sneg = ctokens(sentence)
    if not ct: return {"status": "Insufficient evidence", "score": 0.0, "coverage": 0.0}
    cov = len(ct & st) / len(ct)
    cn, sn = numbers(claim), numbers(sentence)
    nums_ok = cn <= sn if cn else True
    num_conflict = bool(cn) and bool(sn) and not (cn & sn) and cov >= 0.5
    # Metric-aware check: if both texts report the same metric, the claimed value must equal a value the sentence reports FOR THAT METRIC
    # (so "accuracy 90%" does not pass against "accuracy drops from 90% to 83%", where 83% is the reported value).
    cm = {(m["name"], round(m["value"], 4)) for m in find_metrics(claim)}
    sm = {(m["name"], round(m["value"], 4)) for m in find_metrics(sentence)}
    shared = {n for n, _ in cm} & {n for n, _ in sm}
    if shared and cov >= 0.5 and not any((n, v) in sm for n, v in cm if n in shared):
        num_conflict = True
    if shared:
        nums_ok = nums_ok and all((n, v) in sm for n, v in cm if n in shared)
    neg_mismatch = cneg != sneg
    if num_conflict: status = "Contradicted"
    elif neg_mismatch and cov >= 0.6: status = "Contradicted" if sneg and not cneg else "Partially supported" if cov >= 0.8 else "Insufficient evidence"
    elif cov >= 0.7 and nums_ok: status = "Supported"
    elif cov >= 0.4 and nums_ok: status = "Partially supported"
    else: status = "Insufficient evidence"
    return {"status": status, "score": round(cov, 3), "coverage": cov, "neg_mismatch": neg_mismatch}


def best_sentence(claim: str, paper_ids: list[str] | None, k: int = 5):
    """Re-retrieve evidence for the claim and return (best candidate dict, judge result)."""
    best, bj, bs = None, None, -1.0
    emb = get_embedder(); cv = emb.encode([claim])[0]
    for r in retrieve(claim, k, paper_ids=paper_ids):
        for s in split_sentences(r.text):
            if len(s.split()) < 4: continue
            j = judge(claim, s)
            sc = j["coverage"] + 0.2 * max(0.0, cosine(cv, emb.encode([s])[0])) + (0.15 if j["status"] == "Supported" else 0)
            if sc > bs: best, bj, bs = {"paper_id": r.paper_id, "passage": s, "page": r.page, "section": r.section}, j, sc
    return best, bj


class VerifierAgent(Agent):
    name = "Evidence Verification"

    # ---------- claim construction ----------
    def build_claims(self, state: dict) -> list[dict]:
        claims: list[dict] = []
        titles = {pid: short_title(a["title"], 60) for pid, a in (state.get("analyses") or {}).items()}
        for g in state.get("gaps") or []:
            for e in g["evidence"]:
                if e["field"] == "coverage": continue
                t = g.get("theme")
                if t in THEME_CLAIM:
                    phr = THEME_CLAIM[t]
                    claims.append({"claim": f"{titles.get(e['paper_id'], e['paper_id'])}: {phr}", "check": phr, "paper_id": e["paper_id"], "claim_source": "gap", "ref": g["id"]})
                elif t == "dataset_concentration":
                    claims.append({"claim": f"{titles.get(e['paper_id'], e['paper_id'])} uses the {e['text']} dataset", "check": f"{e['text']} dataset", "paper_id": e["paper_id"], "claim_source": "gap", "ref": g["id"]})
        for row in (state.get("comparison") or {}).get("table", []):
            for m in row["metrics"][:3]:
                chk = f"{m['name']} {m['raw']}" + (f" {m['dataset']}" if m.get("dataset") else "")
                claims.append({"claim": f"{short_title(row['title'], 60)} reports {m['name']} of {m['raw']}" + (f" on {m['dataset']}" if m.get("dataset") else ""),
                               "check": chk, "paper_id": row["paper_id"], "claim_source": "comparison"})
        seen = set()
        for f in state.get("findings") or []:
            if f["field"] in ("results", "methodology", "objectives") and (f["paper_id"], f["field"]) not in seen:
                seen.add((f["paper_id"], f["field"]))
                claims.append({"claim": f"{titles.get(f['paper_id'], f['paper_id'])}: {f['text']}", "check": f["text"], "paper_id": f["paper_id"], "claim_source": "finding"})
        narrative = (state.get("comparison") or {}).get("narrative")
        if narrative:
            for s in split_sentences(narrative):
                if len(s.split()) >= 6: claims.append({"claim": s, "check": s, "paper_id": None, "claim_source": "narrative"})
        return claims

    # ---------- verification ----------
    def verify_claim(self, c: dict, all_ids: list[str]) -> dict:
        pid = c.get("paper_id")
        best, j = best_sentence(c["check"], [pid] if pid else all_ids)
        note = None
        if best is None:
            return {**c, "status": "Insufficient evidence", "score": 0.0, "passage": "", "page": None, "section": None, "contradicting": [], "method": "heuristic",
                    "note": "No retrievable source text." if pid else "Missing citation: no source text found."}
        if not pid:
            if j["status"] in ("Supported", "Partially supported"):
                pid, note = best["paper_id"], "Citation attributed by verifier (claim carried no citation)."
            else:
                note = "Missing citation and no supporting passage found."
        contra = []
        if j["status"] in ("Supported", "Partially supported", "Contradicted"):
            others = [p for p in all_ids if p != pid]
            if others:
                for r in retrieve(c["check"], 4, paper_ids=others):
                    for s in split_sentences(r.text):
                        jj = judge(c["check"], s)
                        if jj["coverage"] >= 0.6 and jj["neg_mismatch"] and len(contra) < 3:
                            contra.append({"paper_id": r.paper_id, "passage": s, "page": r.page, "section": r.section})
        return {**c, "paper_id": pid or best["paper_id"] if (pid or j["status"] != "Insufficient evidence") else None, "status": j["status"], "score": j["score"],
                "passage": best["passage"], "page": best["page"], "section": best["section"], "contradicting": contra, "method": "heuristic", "note": note}

    def llm_review(self, items: list[dict], warnings: list[str]) -> None:
        todo = [i for i in items if i["status"] != "Supported" and i["passage"]][:15]
        if not todo: return
        payload = [{"i": n, "claim": t["check"], "passage": t["passage"]} for n, t in enumerate(todo)]
        out = self.llm.complete_json("For each item decide if the passage supports the claim. Allowed verdicts: Supported, Partially supported, "
                                     'Contradicted, Insufficient evidence. Return [{"i":0,"verdict":"...","evidence":"exact words copied from the passage"}].\n'
                                     + wrap_untrusted(repr(payload)), max_tokens=1200)
        if not isinstance(out, list):
            warnings.append(f"LLM verification review skipped ({self.llm.last_error})."); return
        for r in out:
            try: t = todo[int(r["i"])]
            except (KeyError, ValueError, TypeError, IndexError): continue
            v = r.get("verdict")
            if v in STATUSES and quote_in(str(r.get("evidence", "")), t["passage"]):
                t["status"], t["method"] = v, "llm"

    def reference_checks(self, state: dict) -> list[dict]:
        from academic_sources import crossref
        out = []
        validate = (state.get("config") or {}).get("validate_dois", False)
        for p in repo.project_papers(state["project_id"], selected_only=True):
            row = {"paper_id": p["paper_id"], "title": p["title"], "doi": p["doi"], "doi_status": "no_doi" if not p["doi"] else "not_checked",
                   "metadata_missing": [f for f in ("authors", "year", "abstract") if not p.get(f)]}
            if p["is_uploaded"]: row["doi_status"] = "uploaded_document"
            elif p["doi"] and validate:
                v = crossref.validate_doi(p["doi"])
                row["doi_status"] = "registered" if v else "not_found" if v is False else "unchecked_network"
            out.append(row)
        return out

    def run(self, state: dict) -> dict:
        analyses = state.get("analyses") or {}
        all_ids = list(analyses)
        if not all_ids:
            return {"verifications": [], "supported_ratio": 0.0, "evidence_status": "insufficient", "reference_checks": [],
                    "warnings": ["Verification skipped: no analysed papers."]}
        warnings: list[str] = []
        items = [self.verify_claim(c, all_ids) for c in self.build_claims(state)]
        if self.use_llm: self.llm_review(items, warnings)
        total = len(items)
        ok = sum(1 for i in items if i["status"] in ("Supported", "Partially supported"))
        ratio = round(ok / total, 3) if total else 0.0
        cfg_min = float((state.get("config") or {}).get("min_supported_ratio", 0.5))
        sufficient = total > 0 and ratio >= cfg_min and len(all_ids) >= 2
        repo.replace_verifications(state["project_id"], items)
        refs = self.reference_checks(state)
        bad = [i for i in items if i["status"] in ("Contradicted", "Insufficient evidence")]
        if bad: warnings.append(f"{len(bad)} claim(s) are contradicted or lack sufficient evidence and are flagged in the report.")
        return {"verifications": items, "supported_ratio": ratio, "evidence_status": "sufficient" if sufficient else "insufficient",
                "reference_checks": refs, "warnings": warnings}
