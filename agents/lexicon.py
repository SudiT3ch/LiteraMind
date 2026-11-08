"""Shared rule-based resources for heuristic (no-LLM) analysis. Transparent regex lexicons, not hidden model behaviour."""
from __future__ import annotations
import re

ALGORITHMS: list[tuple[str, str, str]] = [  # (canonical, regex, family)
    ("CNN", r"convolutional neural network|(?-i:\bCNNs?\b)", "Deep learning"),
    ("ResNet", r"(?-i:\bResNet-?\d*)", "Deep learning"),
    ("Vision Transformer", r"vision transformer|(?-i:\bViT\b)", "Deep learning"),
    ("Transformer", r"\btransformers?\b", "Deep learning"),
    ("LSTM", r"(?-i:\bLSTMs?\b)|long short-term memory", "Deep learning"),
    ("BERT", r"(?-i:\bBERT\b)", "Deep learning"),
    ("GPT / LLM", r"(?-i:\bGPT-?\d*)|large language models?|(?-i:\bLLMs?\b)", "Deep learning"),
    ("Graph Neural Network", r"graph neural network|(?-i:\bGNNs?\b)", "Deep learning"),
    ("Autoencoder", r"auto-?encoders?", "Deep learning"),
    ("GAN", r"generative adversarial|(?-i:\bGANs?\b)", "Deep learning"),
    ("Random Forest", r"random forests?", "Tree-based ML"),
    ("XGBoost / Gradient Boosting", r"xgboost|gradient boost", "Tree-based ML"),
    ("SVM", r"support vector|(?-i:\bSVMs?\b)", "Classical ML"),
    ("Logistic Regression", r"logistic regression", "Classical ML"),
    ("k-means", r"k-means", "Classical ML"),
    ("Reinforcement Learning", r"reinforcement learning", "Reinforcement learning"),
    ("Grad-CAM", r"grad-?cam", "Post-hoc attribution"),
    ("Saliency Maps", r"saliency", "Post-hoc attribution"),
    ("SHAP", r"(?-i:\bSHAP\b)", "Post-hoc attribution"),
    ("LIME", r"(?-i:\bLIME\b)", "Post-hoc attribution"),
    ("Counterfactual Explanations", r"counterfactual", "Counterfactual explanation"),
    ("Attention Visualisation", r"attention (?:weights|visuali[sz]ation|maps?)", "Attention-based explanation"),
    ("Federated Averaging", r"federated averaging|fedavg|federated (?:learning|training)", "Privacy-preserving learning"),
    ("DP-SGD / Differential Privacy", r"dp-sgd|differential(?:ly)? privacy|differentially private", "Privacy-preserving learning"),
    ("Retrieval-Augmented Generation", r"retrieval-augmented|(?-i:\bRAG\b)", "Retrieval-augmented"),
]
KNOWN_DATASETS = ["ImageNet", "MNIST", "CIFAR-10", "CIFAR-100", "MIMIC-III", "MIMIC-IV", "CheXpert", "ChestX-ray14", "SQuAD", "GLUE",
                  "QASPER", "SciTLDR", "PubMed", "COCO", "S2ORC", "UCI", "PhysioNet", "ISIC"]
METRIC_NAMES = {"accuracy": "Accuracy", "auc": "AUC", "auroc": "AUC", "f1": "F1", "f1-score": "F1", "f1 score": "F1",
                "precision": "Precision", "recall": "Recall", "bleu": "BLEU", "rouge-l": "ROUGE-L", "rouge": "ROUGE",
                "sensitivity": "Sensitivity", "specificity": "Specificity", "rmse": "RMSE", "mae": "MAE", "map": "mAP",
                "perplexity": "Perplexity"}
_MN = "|".join(sorted((re.escape(k) for k in METRIC_NAMES), key=len, reverse=True))
METRIC_FROM_TO = re.compile(rf"\b({_MN})\b[^.\d]{{0,30}}?from\s+(\d+(?:\.\d+)?)\s*%?\s+to\s+(\d+(?:\.\d+)?)\s*(%?)", re.I)
METRIC_VALUE = re.compile(rf"\b({_MN})\b(?:\s+score)?[^.\d]{{0,40}}?(\d+(?:\.\d+)?)\s*(%?)", re.I)
DATASET_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*[A-Za-z0-9\-]*(?: [A-Z][A-Za-z0-9\-]+)?)\s+(?:dataset|corpus|benchmark|data set)s?\b")
DATASET_STOP = {"The", "This", "Our", "Each", "Both", "That", "These", "A", "An", "One", "Single", "Small", "Large", "Another", "Same",
                "Training", "Test", "Validation", "Public", "Private", "Real", "Medical", "Clinical", "Larger", "Benchmark"}

# Limitation themes -> (label, regex, suggested future direction)
THEMES: dict[str, tuple[str, str, str]] = {
    "external_validation": ("Limited external / multi-site validation",
        r"external validation|independent (?:test|validation|cohort)|single[- ](?:hospital|site|cent(?:re|er)|dataset|institution)|only (?:one|a single) (?:hospital|dataset|site)|multi-?cent|simulated sites?|real multi-hospital|relies on a single",
        "Validate methods on independent, multi-site or real-world deployments with external test sets."),
    "small_data": ("Small or limited datasets",
        r"small (?:dataset|sample|cohort)|only \d+ (?:patients|images|samples|records)|limited (?:data|sample)|larger datasets?",
        "Build and evaluate on larger, more diverse datasets."),
    "user_study": ("Missing clinician / user studies",
        r"user stud(?:y|ies)|clinician (?:evaluation|user stud)|human evaluation|no .{0,20}user study|(?:was|were) limited in size",
        "Run controlled user studies with domain experts to measure real decision impact."),
    "faithfulness": ("Explanation faithfulness / stability not validated",
        r"faithful|(?:may )?not reflect|stability|not evaluated quantitatively|explanations? (?:was|were) not|quantitative(?:ly)? evaluat",
        "Develop and apply quantitative faithfulness and stability metrics for explanations."),
    "privacy_utility": ("Privacy–utility trade-off / unproven privacy guarantees",
        r"accuracy (?:degrades|drops)|privacy[- ]utility|privacy budget|privacy guarantees?|performance degrades",
        "Study methods that reduce the accuracy cost of formal privacy guarantees."),
    "cost": ("High communication / computational cost",
        r"communication cost|computational cost|high cost|computationally (?:expensive|intensive)",
        "Design communication- and compute-efficient training and inference."),
    "heterogeneity": ("Data heterogeneity / generalisation",
        r"non-identical|non-iid|heterogene|distribution shift|generali[sz]",
        "Evaluate robustness to heterogeneous data distributions across sites."),
    "plausibility": ("Plausibility of generated explanations",
        r"unrealistic|implausible|clinically (?:unrealistic|invalid)",
        "Constrain generated explanations with domain knowledge to keep them clinically plausible."),
    "bias": ("Bias and fairness not assessed", r"\bbias(?:es)?\b|fairness", "Assess subgroup performance and fairness."),
}
THEME_RES = {k: re.compile(v[1], re.I) for k, v in THEMES.items()}


def themes_of(sentence: str) -> list[str]:
    return [k for k, r in THEME_RES.items() if r.search(sentence)]


def find_algorithms(text: str) -> list[str]:
    return [n for n, rx, _ in ALGORITHMS if re.search(rx, text, re.I)]


def family_of(name: str) -> str:
    return next((f for n, _, f in ALGORITHMS if n == name), "Other")


def find_datasets(text: str) -> list[str]:
    found = [d for d in KNOWN_DATASETS if re.search(rf"\b{re.escape(d)}\b", text)]
    for m in DATASET_RE.finditer(text):
        name = m.group(1).strip()
        first = name.split()[0]
        if first in DATASET_STOP or len(name) < 3:
            continue
        if name not in found:
            found.append(name)
    return found


def find_metrics(sentence: str) -> list[dict]:
    """Extract (metric, value) pairs; values in % are converted to fractions. 'from X to Y' keeps the final value Y."""
    out, spans = [], []
    for m in METRIC_FROM_TO.finditer(sentence):
        name = METRIC_NAMES[m.group(1).lower()]
        v = float(m.group(3)); pct = m.group(4) == "%"
        out.append({"name": name, "value": v / 100 if pct else v, "raw": m.group(3) + ("%" if pct else ""), "note": f"changed from {m.group(2)}"})
        spans.append(m.span())
    for m in METRIC_VALUE.finditer(sentence):
        if any(s <= m.start() < e for s, e in spans):
            continue
        name = METRIC_NAMES[m.group(1).lower()]
        v = float(m.group(2)); pct = m.group(3) == "%"
        if v > 1 and not pct and name not in ("RMSE", "MAE", "Perplexity", "BLEU"):
            pct = True  # "accuracy of 91" -> percent
        out.append({"name": name, "value": v / 100 if pct else v, "raw": m.group(2) + ("%" if m.group(3) else "")})
    seen, uniq = set(), []
    for x in out:
        k = (x["name"], round(x["value"], 6))
        if k not in seen:
            seen.add(k); uniq.append(x)
    return uniq


DOMAINS = {
    "Medicine & Healthcare": ["medical", "clinical", "diagnosis", "patient", "health", "healthcare", "disease", "radiology", "hospital", "drug"],
    "Computer Vision": ["image", "vision", "visual", "object detection", "segmentation", "video"],
    "Natural Language Processing": ["language", "text", "nlp", "translation", "summarization", "question answering", "llm", "retrieval"],
    "Machine Learning": ["learning", "neural", "classification", "model", "training", "deep"],
    "Privacy & Security": ["privacy", "security", "federated", "attack", "adversarial", "encryption"],
    "Robotics & Control": ["robot", "control", "navigation", "reinforcement"],
    "Education": ["education", "student", "learning analytics", "teaching"],
    "Finance & Economics": ["finance", "stock", "market", "economic", "fraud"],
    "Climate & Environment": ["climate", "weather", "environment", "energy", "emission"],
}
SYNONYMS = {
    "explainable ai": ["XAI", "model interpretability", "post-hoc explanation", "trustworthy AI"],
    "explainable": ["interpretable", "interpretability", "XAI"],
    "interpretability": ["explainability", "explainable AI"],
    "medical diagnosis": ["clinical decision support", "computer-aided diagnosis", "healthcare"],
    "diagnosis": ["clinical decision support", "medical diagnosis"],
    "reliability": ["trustworthiness", "robustness", "calibration"],
    "federated learning": ["privacy-preserving learning", "distributed training", "differential privacy"],
    "privacy": ["differential privacy", "data protection"],
    "large language model": ["LLM", "foundation model"], "llm": ["large language model", "foundation model"],
    "rag": ["retrieval-augmented generation"], "transformer": ["attention", "self-attention"],
    "deep learning": ["neural network"], "cancer": ["oncology", "tumor"],
}
