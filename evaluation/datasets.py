"""Loaders for public benchmark subsets (QASPER, SciTLDR). Require network + `pip install datasets`; otherwise raise DatasetUnavailable
so experiments are reported as PENDING rather than faked."""
from __future__ import annotations


class DatasetUnavailable(Exception):
    pass


def _load(name: str, split: str, n: int):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise DatasetUnavailable("`datasets` package not installed (pip install datasets)") from e
    try:
        ds = load_dataset(name, split=f"{split}[:{n}]")
        return ds
    except Exception as e:  # network / hub errors
        raise DatasetUnavailable(f"Could not download {name}: {type(e).__name__}") from e


def load_scitldr(n: int = 50):
    """allenai/scitldr (AIC config): returns [{'source': str, 'target': [tldr,...]}]."""
    ds = _load("allenai/scitldr", "test", n)
    return [{"source": " ".join(r["source"]), "target": r["target"]} for r in ds]


def load_qasper(n: int = 20):
    """allenai/qasper subset: returns papers with full text and question/answer annotations."""
    return list(_load("allenai/qasper", "validation", n))
