"""Experiment runner:  python -m evaluation.experiments [--only retrieval,summarization,baseline,ablation,verifier,external]

IMPORTANT: offline experiments run on the SYNTHETIC fixture corpus (evaluation/fixtures/synthetic_corpus.py). They validate the pipeline
and relative behaviour of configurations; they are NOT benchmarks of real-world scientific performance. Benchmarks on real data
(QASPER, SciTLDR) run only when the datasets can be downloaded and are otherwise reported as PENDING. Nothing here is hardcoded:
all numbers are computed at run time and written to evaluation/results/."""
from __future__ import annotations
import argparse, json, os, sys, tempfile, time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "evaluation" / "results"
FIELDS = ["datasets", "algorithms", "metrics", "results", "limitations", "future_work"]


def setup_environment():
    """Isolated DB / vector store / uploads; ingest the synthetic corpus as real PDFs. Returns {corpus_key: paper_id}."""
    tmp = Path(tempfile.mkdtemp(prefix="lm_eval_"))
    os.environ.update({"DATABASE_URL": f"sqlite:///{tmp}/eval.db", "UPLOAD_DIR": str(tmp / "up"), "EMBEDDING_BACKEND": os.getenv("EMBEDDING_BACKEND", "hashing")})
    from config import reset_settings
    from database.database import init_db
    from retrieval import embeddings, vector_store
    from llm import provider
    reset_settings(); embeddings.reset_embedder(); provider.reset_provider(); init_db(os.environ["DATABASE_URL"])
    vector_store.set_store(vector_store.VectorStore(ephemeral=True, embedder=embeddings.get_embedder()))
    from tests.helpers import make_pdf
    from evaluation.fixtures.synthetic_corpus import DOCS
    from document_processing.ingest import ingest_pdf_bytes
    ids = {}
    for k, d in DOCS.items():
        pdf = make_pdf(tmp / f"{k}.pdf", d["pages"])
        r = ingest_pdf_bytes(open(pdf, "rb").read(), f"{k}.pdf", paper_id=k, meta={"title": d["title"], "year": d["year"], "authors": d["authors"]})
        assert r["status"] == "processed"; ids[k] = r["paper_id"]
    return ids


def _offline():
    return mock.patch("agents.discovery.search_all", lambda q, per_source=10: ([], []))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _prf(pred: set, gold: set) -> dict:
    tp = len(pred & gold); p = tp / len(pred) if pred else 0.0; r = tp / len(gold) if gold else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def comparable_groups_from_state(final: dict) -> set:
    return {(g["metric"], g["dataset"]) for g in (final.get("comparison") or {}).get("comparable_metrics", [])}


def gap_themes_from_state(final: dict) -> dict:
    return {g["theme"]: {p["paper_id"] for p in g["supporting_papers"]} for g in final.get("gaps", []) if g.get("theme") in
            ("external_validation", "user_study", "cost", "small_data", "faithfulness", "privacy_utility", "heterogeneity", "plausibility", "bias")}


def run_workflow(question: str, ids: dict, config: dict | None = None) -> tuple[dict, float]:
    from orchestration import workflow
    with _offline():
        pid = workflow.new_project(question)
        t = time.perf_counter()
        final = workflow.run_analysis(pid, uploaded_ids=list(ids.values()), config=config)
    final["project_id"] = pid
    return final, time.perf_counter() - t


# ------------------------------------------------------------------ experiments
def exp_retrieval(ids) -> dict:
    from evaluation.fixtures.synthetic_corpus import QUERIES
    from evaluation.retrieval_metrics import evaluate_retrieval
    from retrieval.retriever import retrieve
    out = {"corpus": "synthetic fixture (6 papers) — P@5/R@10 are near-trivial on a corpus this small; use for regression only"}
    for name, kw in {"semantic": dict(mode="semantic", rerank=False), "keyword (BM25)": dict(mode="keyword", rerank=False),
                     "hybrid (RRF)": dict(mode="hybrid", rerank=False), "hybrid + rerank": dict(mode="hybrid", rerank=True)}.items():
        res = evaluate_retrieval(QUERIES, lambda q: [r.paper_id for r in retrieve(q, 30, **kw)], 5, 10)
        res.pop("per_query"); out[name] = res
    # stricter variant: P@3 (informative on 6 docs)
    for name, kw in {"semantic": dict(mode="semantic", rerank=False), "hybrid + rerank": dict(mode="hybrid", rerank=True)}.items():
        out[name]["P@3"] = evaluate_retrieval(QUERIES, lambda q: [r.paper_id for r in retrieve(q, 30, **kw)], 3, 10)["P@3"]
    return out


def exp_summarization(ids) -> dict:
    from evaluation.fixtures.synthetic_corpus import DOCS
    from evaluation.summarization_metrics import rouge_l, factual_consistency
    from evaluation.baseline import baseline_rag
    final, _ = run_workflow("explainable AI for medical diagnosis reliability", ids)
    ag, bl, fc_a, fc_b = [], [], [], []
    for k, d in DOCS.items():
        ref = d["pages"][0].split("Introduction")[0].replace("Abstract", "").strip()
        fnd = [f for f in final["findings"] if f["paper_id"] == k and f["field"] in ("objectives", "methodology", "results")]
        summ = " ".join(f["text"] for f in fnd)
        base = baseline_rag(f"summary of {d['title']}", [k], 4)["answer"]
        ag.append(rouge_l(summ, ref)["f1"]); bl.append(rouge_l(base, ref)["f1"])
        fc_a.append(factual_consistency(summ, [k])["consistency"]); fc_b.append(factual_consistency(base, [k])["consistency"])
    return {"reference": "paper abstracts (synthetic)", "rouge_impl": rouge_l("a b c", "a b c")["impl"],
            "multi_agent_structured_summary": {"ROUGE-L F1": _mean(ag), "factual_consistency": _mean(fc_a)},
            "baseline_extractive": {"ROUGE-L F1": _mean(bl), "factual_consistency": _mean(fc_b)},
            "human_evaluation": "PENDING — see evaluation/results/human_eval_template.csv", "n_papers": len(DOCS)}


def _field_presence(paper_id: str, text_items: list[dict]) -> set:
    """Which structured fields can be recovered for a paper from a list of {text, section} items (same lexicon for both configs)."""
    from agents.lexicon import find_algorithms, find_datasets, find_metrics
    got = set()
    for s in text_items:
        t, sec = s["text"], (s.get("section") or "").lower()
        if find_datasets(t): got.add("datasets")
        if find_algorithms(t): got.add("algorithms")
        if find_metrics(t): got.add("metrics")
        if sec in ("results", "conclusion"): got.add("results")
        if sec == "limitations": got.add("limitations")
        if sec == "future work": got.add("future_work")
    return got


def exp_baseline_vs_multiagent(ids) -> dict:
    from agents.lexicon import find_metrics, find_datasets
    from evaluation.baseline import baseline_rag, baseline_claims
    from evaluation.citation_metrics import citation_metrics
    from evaluation.fixtures.synthetic_corpus import QUERIES, GOLD_COMPARABLE, GOLD_GAP_THEMES
    from agents.verifier import VerifierAgent
    from llm.provider import get_provider
    qs = {"How can explainable AI improve reliability of medical diagnosis?": QUERIES["How can explainable AI improve reliability of medical diagnosis?"],
          "federated learning privacy hospitals": QUERIES["federated learning privacy hospitals"]}
    rows = []
    for q, relevant in qs.items():
        # ---- BASELINE ----
        t = time.perf_counter(); b = baseline_rag(q, None, 8); bt = time.perf_counter() - t
        claims = baseline_claims(b)
        ver = VerifierAgent(); allids = list(ids.values())
        bver = [ver.verify_claim({**c, "claim_source": "baseline"}, allids) for c in claims]
        cited = {c["paper_id"] for c in claims}
        by_pair: dict = {}
        for c in claims:
            for m in find_metrics(c["claim"]):
                ds = next(iter(find_datasets(c["claim"])), None)
                if ds: by_pair.setdefault((m["name"], ds), set()).add(c["paper_id"])
        b_cmp = {k for k, v in by_pair.items() if len(v) >= 2}
        b_items = {p: [{"text": c["text"], "section": c.get("section")} for c in b["citations"] if c.get("paper_id") == p and "text" in c] for p in relevant}
        b_complete = _mean([len(_field_presence(p, b_items[p])) / len(FIELDS) for p in relevant])
        # ---- PROPOSED ----
        final, mt = run_workflow(q, ids)
        a_items = {p: [{"text": f["passage"], "section": f["section"]} for f in final["findings"] if f["paper_id"] == p] for p in relevant}
        a_complete = _mean([len(_field_presence(p, a_items[p])) / len(FIELDS) for p in relevant])
        gold_cmp = set(GOLD_COMPARABLE)
        gt = gap_themes_from_state(final)
        a_cov = len({v["paper_id"] for v in final["verifications"] if v.get("paper_id")} & relevant) / len(relevant)
        rows.append({"question": q, "n_relevant_papers": len(relevant),
                     "baseline": {"mode": b["mode"], "time_s": round(bt, 3), "tokens": b["usage"]["prompt_tokens"] + b["usage"]["completion_tokens"] or "n/a (no LLM)",
                                  **citation_metrics(bver), "evidence_coverage": round(len(cited & relevant) / len(relevant), 4), "analysis_completeness": b_complete,
                                  "comparison_f1": _prf(b_cmp, gold_cmp)["f1"], "gap_theme_f1": 0.0, "note": "baseline produces no gap analysis by design"},
                     "multi_agent": {"time_s": round(mt, 3), "tokens": "n/a (no LLM)" if not get_provider().available else final["usage"],
                                     **citation_metrics(final["verifications"]), "evidence_coverage": round(a_cov, 4), "analysis_completeness": a_complete,
                                     "comparison_f1": _prf(comparable_groups_from_state(final), gold_cmp)["f1"],
                                     "gap_theme_f1": _prf(set(gt), set(GOLD_GAP_THEMES))["f1"], "n_gaps": len(final["gaps"])}})
    return {"configs": "BASELINE = single-pass extractive RAG (no LLM available in this run); PROPOSED = six-agent pipeline (rule-based path)",
            "caveat": "answer correctness/faithfulness vs reference answers and LLM-mode comparison: PENDING (needs QASPER download and an LLM key)",
            "per_question": rows,
            "mean": {cfg: {k: _mean([r[cfg][k] for r in rows if isinstance(r[cfg].get(k), (int, float))]) for k in
                           ("time_s", "citation_support_rate", "unsupported_claim_rate", "citation_completeness", "source_traceability", "evidence_coverage",
                            "analysis_completeness", "comparison_f1", "gap_theme_f1")} for cfg in ("baseline", "multi_agent")}}


def exp_ablation(ids) -> dict:
    from evaluation.citation_metrics import citation_metrics
    from evaluation.fixtures.synthetic_corpus import GOLD_COMPARABLE, GOLD_GAP_THEMES
    from database import repository as repo
    q = "How can explainable AI improve reliability of medical diagnosis?"
    cfgs = {"full system": {}, "− Comparison Agent": {"enable_comparison": False}, "− Research Gap Agent": {"enable_gap": False},
            "− Evidence Verification Agent": {"enable_verification": False}}
    out = {}
    for name, cfg in cfgs.items():
        final, t = run_workflow(q, ids, cfg)
        cm = citation_metrics(final.get("verifications") or [])
        md = repo.latest_report(final["project_id"])["markdown"]
        gt = gap_themes_from_state(final)
        out[name] = {"time_s": round(t, 3), "comparison_f1": _prf(comparable_groups_from_state(final), set(GOLD_COMPARABLE))["f1"],
                     "gap_theme_f1": _prf(set(gt), set(GOLD_GAP_THEMES))["f1"], "n_gaps": len(final.get("gaps") or []),
                     "claims_checked": cm["n_claims"], "citation_support_rate": cm["citation_support_rate"], "unsupported_claim_rate": cm["unsupported_claim_rate"],
                     "report_statements_marked_unverified": md.count("[not verified]") + md.count("[not individually verified]"),
                     "report_has_comparison_table": "| Paper |" in md, "report_has_gap_section_content": "No candidate research gaps" not in md}
    return {"question": q, "results": out, "reading_guide": "Claims-checked / unsupported rates are None when verification is disabled: nothing is verified, and the report marks statements as '[not verified]'."}


def exp_verifier_detection(ids) -> dict:
    """Fault injection: real metric claims vs numerically perturbed (false) claims. Measures whether the verifier separates them."""
    from agents.verifier import VerifierAgent
    final, _ = run_workflow("explainable AI for medical diagnosis reliability", ids)
    ver = VerifierAgent(); allids = list(ids.values()); real, fake = [], []
    for row in final["comparison"]["table"]:
        for m in row["metrics"]:
            if not m.get("dataset"): continue
            true_raw = m["raw"]; num = float(true_raw.rstrip("%")); pct = true_raw.endswith("%")
            wrong = f"{num + (7 if pct else 0.07):g}{'%' if pct else ''}"
            mk = lambda raw: {"claim": f"{row['title']} reports {m['name']} of {raw}", "check": f"{m['name']} {raw} {m['dataset']}", "paper_id": row["paper_id"], "claim_source": "injection"}
            real.append(ver.verify_claim(mk(true_raw), allids)["status"]); fake.append(ver.verify_claim(mk(wrong), allids)["status"])
    ok = lambda s: s in ("Supported", "Partially supported")
    return {"n_true_claims": len(real), "n_false_claims_injected": len(fake),
            "true_claims_accepted (recall)": round(sum(map(ok, real)) / len(real), 4) if real else None,
            "false_claims_flagged (detection rate)": round(sum(1 for s in fake if not ok(s)) / len(fake), 4) if fake else None,
            "false_claims_accepted_as_supported": sum(1 for s in fake if s == "Supported"), "note": "false claims are real claims with the number changed; synthetic fixture, so a sanity check of the numeric-consistency rule only"}


def exp_external() -> dict:
    from evaluation.datasets import load_scitldr, load_qasper, DatasetUnavailable
    out = {}
    for name, fn in (("SciTLDR", load_scitldr), ("QASPER", load_qasper)):
        try:
            data = fn(10); out[name] = {"status": "dataset loaded", "n": len(data), "note": "runner for full evaluation not executed in this offline environment"}
        except DatasetUnavailable as e:
            out[name] = {"status": "PENDING", "reason": str(e)}
    return out


def write_human_eval_template(ids) -> str:
    import csv
    final, _ = run_workflow("explainable AI for medical diagnosis reliability", ids)
    path = RESULTS / "human_eval_template.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item_type", "id", "text", "rating_relevance_1to5", "rating_evidence_support_1to5", "rating_usefulness_1to5", "comments"])
        for g in final["gaps"]: w.writerow(["candidate_gap", g["id"], f"{g['title']} — {g['description']}", "", "", "", ""])
        for k in ids: w.writerow(["summary", k, " ".join(x["text"] for x in final["findings"] if x["paper_id"] == k and x["field"] in ("objectives", "results")), "", "", "", ""])
    return str(path)


EXPERIMENTS = {"retrieval": exp_retrieval, "summarization": exp_summarization, "baseline": exp_baseline_vs_multiagent, "ablation": exp_ablation, "verifier": exp_verifier_detection}


def _md_table(d: dict) -> str:
    keys = list(next(iter(d.values())).keys())
    rows = ["| config | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    rows += [f"| {n} | " + " | ".join(str(v.get(k)) for k in keys) + " |" for n, v in d.items()]
    return "\n".join(rows)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=",".join(list(EXPERIMENTS) + ["external"])); a = ap.parse_args(argv)
    want = set(a.only.split(","))
    RESULTS.mkdir(parents=True, exist_ok=True)
    ids = setup_environment()
    from retrieval.embeddings import get_embedder
    from llm.provider import get_provider
    res = {"meta": {"corpus": "SYNTHETIC fixture corpus (not real publications)", "embedder": get_embedder().name, "llm": get_provider().name,
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0]}}
    for name, fn in EXPERIMENTS.items():
        if name in want:
            t = time.perf_counter(); res[name] = fn(ids); res[name]["_runtime_s"] = round(time.perf_counter() - t, 2); print(f"[done] {name} ({res[name]['_runtime_s']}s)")
    if "external" in want: res["external_benchmarks"] = exp_external()
    res["human_evaluation"] = {"status": "PENDING", "template": write_human_eval_template(ids)}
    (RESULTS / "results.json").write_text(json.dumps(res, indent=2, default=str))
    md = [f"# LiteraMind experiment results\n\nGenerated {res['meta']['date']} · embedder `{res['meta']['embedder']}` · LLM `{res['meta']['llm']}`\n",
          "> **Read this first.** Measured on a *synthetic fixture corpus* of 6 hand-written documents with author-assigned gold labels. These runs validate the pipeline and the relative effect of components; "
          "they are **not** evidence of real-world accuracy. Real-data benchmarks and LLM-mode runs are marked PENDING.\n"]
    if "retrieval" in res: md += ["## Retrieval (paper-level)\n", _md_table({k: v for k, v in res["retrieval"].items() if isinstance(v, dict)}), ""]
    if "summarization" in res: md += ["## Summarisation\n", "```json\n" + json.dumps({k: v for k, v in res["summarization"].items()}, indent=1) + "\n```", ""]
    if "baseline" in res: md += ["## Baseline RAG vs Multi-Agent RAG (means over questions)\n", _md_table(res["baseline"]["mean"]), "", res["baseline"]["caveat"], ""]
    if "ablation" in res: md += ["## Ablation\n", _md_table(res["ablation"]["results"]), "", res["ablation"]["reading_guide"], ""]
    if "verifier" in res: md += ["## Verifier fault injection\n", "```json\n" + json.dumps(res["verifier"], indent=1) + "\n```", ""]
    if "external_benchmarks" in res: md += ["## External benchmarks\n", "```json\n" + json.dumps(res["external_benchmarks"], indent=1) + "\n```", ""]
    (RESULTS / "results.md").write_text("\n".join(md))
    print("\n".join(md))
    return res


if __name__ == "__main__":
    main()
