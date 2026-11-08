# LiteraMind experiment results

Generated 2026-11-07 16:13:14 · embedder `hashing-512` · LLM `heuristic`

> **Read this first.** Measured on a *synthetic fixture corpus* of 6 hand-written documents with author-assigned gold labels. These runs validate the pipeline and the relative effect of components; they are **not** evidence of real-world accuracy. Real-data benchmarks and LLM-mode runs are marked PENDING.

## Retrieval (paper-level)

| config | P@5 | R@10 | MRR | n_queries | P@3 |
|---|---|---|---|---|---|
| semantic | 0.3333 | 1.0 | 1.0 | 6 | 0.5 |
| keyword (BM25) | 0.4611 | 1.0 | 1.0 | 6 | None |
| hybrid (RRF) | 0.3667 | 1.0 | 1.0 | 6 | None |
| hybrid + rerank | 0.3667 | 1.0 | 1.0 | 6 | 0.5 |

## Summarisation

```json
{
 "reference": "paper abstracts (synthetic)",
 "rouge_impl": "rouge-score",
 "multi_agent_structured_summary": {
  "ROUGE-L F1": 0.4863,
  "factual_consistency": 1.0
 },
 "baseline_extractive": {
  "ROUGE-L F1": 0.4931,
  "factual_consistency": 1.0
 },
 "human_evaluation": "PENDING \u2014 see evaluation/results/human_eval_template.csv",
 "n_papers": 6,
 "_runtime_s": 2.59
}
```

## Baseline RAG vs Multi-Agent RAG (means over questions)

| config | time_s | citation_support_rate | unsupported_claim_rate | citation_completeness | source_traceability | evidence_coverage | analysis_completeness | comparison_f1 | gap_theme_f1 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.0055 | 1.0 | 0.0 | 1.0 | 1.0 | 0.75 | 0.2292 | 0.0 | 0.0 |
| multi_agent | 0.4355 | 0.8636 | 0.1136 | 1.0 | 1.0 | 1.0 | 0.8958 | 1.0 | 0.6667 |

answer correctness/faithfulness vs reference answers and LLM-mode comparison: PENDING (needs QASPER download and an LLM key)

## Ablation

| config | time_s | comparison_f1 | gap_theme_f1 | n_gaps | claims_checked | citation_support_rate | unsupported_claim_rate | report_statements_marked_unverified | report_has_comparison_table | report_has_gap_section_content |
|---|---|---|---|---|---|---|---|---|---|---|
| full system | 0.468 | 1.0 | 0.6667 | 10 | 44 | 0.8636 | 0.1136 | 18 | True | True |
| − Comparison Agent | 0.377 | 0.0 | 0.6667 | 10 | 38 | 0.8421 | 0.1316 | 18 | False | True |
| − Research Gap Agent | 0.316 | 1.0 | 0.0 | 0 | 24 | 1.0 | 0.0 | 18 | True | False |
| − Evidence Verification Agent | 0.105 | 1.0 | 0.6667 | 10 | 0 | None | None | 36 | True | True |

Claims-checked / unsupported rates are None when verification is disabled: nothing is verified, and the report marks statements as '[not verified]'.

## Verifier fault injection

```json
{
 "n_true_claims": 5,
 "n_false_claims_injected": 5,
 "true_claims_accepted (recall)": 1.0,
 "false_claims_flagged (detection rate)": 1.0,
 "false_claims_accepted_as_supported": 0,
 "note": "false claims are real claims with the number changed; synthetic fixture, so a sanity check of the numeric-consistency rule only",
 "_runtime_s": 0.63
}
```

## External benchmarks

```json
{
 "SciTLDR": {
  "status": "PENDING",
  "reason": "`datasets` package not installed (pip install datasets)"
 },
 "QASPER": {
  "status": "PENDING",
  "reason": "`datasets` package not installed (pip install datasets)"
 }
}
```
