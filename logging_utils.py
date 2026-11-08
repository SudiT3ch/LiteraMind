"""Logging with secret redaction so API keys never reach logs or UI error messages."""
from __future__ import annotations
import logging, os, re

_SECRET_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "S2_API_KEY")


def redact(text: str) -> str:
    out = str(text)
    for name in _SECRET_ENV:
        v = os.getenv(name, "")
        if v and len(v) > 6:
            out = out.replace(v, "***")
    out = re.sub(r"(sk-[A-Za-z0-9_\-]{10,})", "***", out)
    out = re.sub(r"(?i)(x-api-key|authorization)\s*[:=]\s*\S+", r"\1: ***", out)
    return out


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        for h in logging.getLogger().handlers:
            h.addFilter(_RedactFilter())
    return logger
