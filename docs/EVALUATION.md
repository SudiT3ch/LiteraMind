# Evaluation methodology

Run: `python -m evaluation.experiments [--only retrieval,summarization,baseline,ablation,verifier,external]` → `evaluation/results/results.{json,md}` (+ `human_eval_template.csv`).

| Area | Metrics implemented | Where |
|---|---|---|
| Retrieval | Precision@5, Recall@10, MRR (paper level, de-duplicated), P@3 | `retrieval_metrics.py` |
| Summarisation | ROUGE-L (rouge-score), factual consistency (verifier-based) | `summarization_metrics.py` |
| Citation verification | support rate, partial rate, unsupported-claim rate, citation completeness, source traceability (passage found verbatim in stored chunks with page+section) | `citation_metrics.py` |
| Baseline vs proposed | citation metrics, evidence coverage, analysis completeness, comparison F1, gap-theme F1, time, tokens | `experiments.py::exp_baseline_vs_multiagent` |
| Ablation | removing Comparison / Gap / Verification | `experiments.py::exp_ablation` |
| Verifier fault injection | true claims accepted vs numerically-perturbed claims flagged | `experiments.py::exp_verifier_detection` |
| Benchmarks | SciTLDR, QASPER loaders (`datasets.py`) | **PENDING** — need network + `pip install datasets` |
| Human evaluation | template generated | **PENDING** |
| RAGAS | **not integrated** — it requires an LLM judge/embeddings API; custom metrics above are used instead |

## What the committed results are (and are not)
Computed offline on a **synthetic 6-document fixture corpus** with **author-assigned** gold labels (`evaluation/fixtures/synthetic_corpus.py`), rule-based mode (no LLM), hashing embedder. They demonstrate that the pipeline runs end-to-end and how components behave relative to each other; they are not benchmark scores and must not be quoted as real-world accuracy. Known confounds:
* The **baseline's citation-support rate (1.0) is tautological**: in extractive mode its "claims" are the retrieved sentences themselves. The multi-agent system verifies harder claims (paraphrased gap statements, numeric comparisons), so its lower rate is not evidence the baseline is better *or* worse.
* Gold gap themes list only 4 themes; the system reports more valid-looking themes, which lowers gap precision.
* With 6 papers, P@5/R@10 saturate; BM25 outscored hybrid on P@5 here — no claim of hybrid superiority is made.
* To produce meaningful numbers: put your key in `.env` (`LLM_PROVIDER=anthropic|openai|ollama`), install `sentence-transformers` and `datasets`, and re-run.
