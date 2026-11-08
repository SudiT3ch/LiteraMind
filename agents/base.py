"""Common agent plumbing."""
from __future__ import annotations
import re
from typing import Any
from llm.provider import get_provider


class Agent:
    name = "agent"

    @property
    def llm(self):
        return get_provider()

    @property
    def use_llm(self) -> bool:
        return self.llm.available

    def run(self, state: dict) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


def short_title(t: str, n: int = 70) -> str:
    t = re.sub(r"^\[SYNTHETIC\]\s*", "", t or "").strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def quote_in(quote: str, text: str) -> bool:
    """True if quote (whitespace/case-normalised) occurs verbatim in text. Used to reject hallucinated LLM evidence."""
    q = norm_ws(quote)
    return len(q) >= 12 and q in norm_ws(text)
