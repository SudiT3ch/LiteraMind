"""Central configuration. All secrets come from environment variables / .env (never hardcoded)."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # python-dotenv optional
    pass

BASE_DIR = Path(__file__).resolve().parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    # --- LLM (provider: heuristic | anthropic | openai | ollama) ---
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "heuristic").lower())
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    llm_timeout: int = field(default_factory=lambda: _int("LLM_TIMEOUT", 60))
    llm_max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 2000))
    # --- Academic APIs (all optional keys) ---
    s2_api_key: str = field(default_factory=lambda: os.getenv("S2_API_KEY", ""))
    contact_email: str = field(default_factory=lambda: os.getenv("CONTACT_EMAIL", ""))  # polite pool for OpenAlex/Crossref
    http_timeout: int = field(default_factory=lambda: _int("HTTP_TIMEOUT", 20))
    http_retries: int = field(default_factory=lambda: _int("HTTP_RETRIES", 2))
    enabled_sources: str = field(default_factory=lambda: os.getenv("ENABLED_SOURCES", "arxiv,semantic_scholar,openalex"))
    # --- Storage ---
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'literamind.db'}"))
    chroma_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", str(BASE_DIR / "data" / "chroma")))
    upload_dir: str = field(default_factory=lambda: os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads")))
    report_dir: str = field(default_factory=lambda: os.getenv("REPORT_DIR", str(BASE_DIR / "data" / "reports")))
    # --- Embeddings: 'auto' tries sentence-transformers then falls back to hashing embedder ---
    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "auto"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    # --- Chunking / retrieval ---
    chunk_words: int = field(default_factory=lambda: _int("CHUNK_WORDS", 220))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 40))
    top_k: int = field(default_factory=lambda: _int("TOP_K", 8))
    # --- Upload safeguards ---
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 25))
    max_pdf_pages: int = field(default_factory=lambda: _int("MAX_PDF_PAGES", 120))
    # --- Workflow ---
    max_papers: int = field(default_factory=lambda: _int("MAX_PAPERS", 8))
    min_papers: int = field(default_factory=lambda: _int("MIN_PAPERS", 3))
    max_retries: int = field(default_factory=lambda: _int("AGENT_RETRIES", 1))
    max_expansion_loops: int = field(default_factory=lambda: _int("MAX_EXPANSION_LOOPS", 1))
    min_supported_ratio: float = field(default_factory=lambda: _float("MIN_SUPPORTED_RATIO", 0.5))
    backend_url: str = field(default_factory=lambda: os.getenv("BACKEND_URL", "http://127.0.0.1:8000"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def sources(self) -> list[str]:
        return [s.strip() for s in self.enabled_sources.split(",") if s.strip()]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> Settings:
    """Re-read environment (used by tests)."""
    global _settings
    _settings = Settings()
    return _settings
