"""Embedding backends. Preferred: Sentence-Transformers. Fallback: deterministic hashing embedder (no download, CPU-only)."""
from __future__ import annotations
import hashlib, re
import numpy as np
from config import get_settings
from logging_utils import get_logger

log = get_logger("embeddings")
STOP = set("a an the of and or to in on for with by from is are was were be been being this that these those it its as at we our "
           "can may also which not but their than such into using use used based paper results show shown study".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", (text or "").lower())
    return [stem(t) for t in toks if t not in STOP and len(t) > 1]


def stem(w: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


class HashingEmbedder:
    """Signed feature-hashing of unigrams+bigrams with sublinear TF; L2-normalised. Lexical, not semantic."""
    name = "hashing-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _idx(self, tok: str) -> tuple[int, int]:
        h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
        return h % self.dim, 1 if (h >> 40) & 1 else -1

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            toks = tokenize(t)
            feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
            counts: dict[str, int] = {}
            for f in feats:
                counts[f] = counts.get(f, 0) + 1
            for f, c in counts.items():
                i, s = self._idx(f)
                out[r, i] += s * (1.0 + np.log(c)) * (0.6 if "_" in f else 1.0)
            n = np.linalg.norm(out[r])
            if n > 0:
                out[r] /= n
        return out


class STEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import
        self.model = SentenceTransformer(model_name)
        self.name = "st-" + re.sub(r"[^a-zA-Z0-9]+", "-", model_name.split("/")[-1])[:40]

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


_embedder = None


def get_embedder():
    """Singleton. 'auto' tries Sentence-Transformers and silently falls back (with a log warning) to hashing."""
    global _embedder
    if _embedder is not None:
        return _embedder
    st = get_settings()
    backend = st.embedding_backend
    if backend in ("auto", "sentence-transformers"):
        try:
            _embedder = STEmbedder(st.embedding_model)
            log.info("Using Sentence-Transformers model %s", st.embedding_model)
            return _embedder
        except Exception as e:
            if backend == "sentence-transformers":
                raise RuntimeError(f"Sentence-Transformers unavailable: {type(e).__name__}") from e
            log.warning("Sentence-Transformers unavailable (%s); using hashing embedder (lexical).", type(e).__name__)
    _embedder = HashingEmbedder()
    return _embedder


def reset_embedder() -> None:
    global _embedder
    _embedder = None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d else 0.0
